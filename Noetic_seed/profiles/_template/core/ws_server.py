"""WebSocketサーバー（UIアプリとの双方向通信）"""
import asyncio
import base64
import json
import threading
import queue
import secrets
from datetime import datetime
from pathlib import Path

_ws_clients: set = set()
_ws_token: str = ""
_ws_log_buffer: list = []
_LOG_BUFFER_MAX = 100
_send_queue: queue.Queue = queue.Queue()
_current_state: dict = {}
_chat_queue: queue.Queue = queue.Queue()  # 外部入力キュー（main.pyが読む）
_approval_queue: queue.Queue = queue.Queue()  # 承認応答キュー
_pending_approval: dict = {}  # 現在承認待ちのリクエスト
_device_queue: queue.Queue = queue.Queue()  # デバイス応答キュー
_pending_device: dict = {}  # 現在デバイス応答待ちのリクエスト
_test_tool_queue: queue.Queue = queue.Queue()  # テストタブからの実行要求
_paused: bool = False  # サイクル一時停止フラグ（pause/resume で切替、メモリ内のみ）
_profile_running: bool = False  # main.py 稼働中フラグ（server.py 時は False のまま）


def is_paused() -> bool:
    """main.py が毎 tick 呼ぶ。pause 中は cycle break を抑制する判定に使う。"""
    return _paused


def set_paused(value: bool) -> None:
    """pause フラグを外部から設定する。

    承認 callback 等が `pause_on_await=True` で is_paused 発動させる経路。
    既存の WebSocket server_command (pause/resume) 経由とは別に、Noetic
    runtime 内部から直接呼び出せる API。
    """
    global _paused
    _paused = bool(value)


def set_profile_running(running: bool = True):
    """main.py 起動時に呼ぶ。接続時に state を常に送るようになる。"""
    global _profile_running
    _profile_running = running

# === camera_stream 非同期処理用 ===
_stream_frames_lock = threading.Lock()
_stream_frames: list = []  # (rel_path, meta) のリスト、ローリングバッファ
_stream_frame_counter: int = 0  # 新フレーム到着ごとにインクリメント（main.py が「新規判定」に使う）
_STREAM_BUFFER_MAX = 5


def _get_token() -> str:
    global _ws_token
    if not _ws_token:
        # settings.jsonにws_tokenがあれば固定トークンとして使う
        try:
            import json
            from core.config import LLM_SETTINGS
            with open(LLM_SETTINGS, encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("ws_token"):
                _ws_token = cfg["ws_token"]
                return _ws_token
        except Exception:
            pass
        _ws_token = secrets.token_urlsafe(16)
    return _ws_token


async def _ws_handler(websocket):
    """WebSocket接続ハンドラ"""
    try:
        auth_msg = await asyncio.wait_for(websocket.recv(), timeout=10)
        auth = json.loads(auth_msg)
        if auth.get("type") != "auth" or auth.get("token") != _get_token():
            await websocket.send(json.dumps({"type": "error", "message": "authentication failed"}))
            await websocket.close()
            return
    except Exception:
        await websocket.close()
        return

    _ws_clients.add(websocket)
    print(f"  [ws] client connected ({len(_ws_clients)} total)")

    # 接続時: バッファされたログ + 現在のstateを送信
    try:
        await websocket.send(json.dumps({
            "type": "sync",
            "recent_logs": _ws_log_buffer[-50:],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False))
        if _current_state or _profile_running:
            await websocket.send(json.dumps({
                "type": "state",
                "entropy": _current_state.get("entropy", 0.65),
                "energy": round(_current_state.get("energy", 50), 1),
                "cycle_id": _current_state.get("cycle_id", 0),
                "tool_level": _current_state.get("tool_level", 0),
                "pressure": _current_state.get("pressure", 0),
            }, ensure_ascii=False))
        if _current_state.get("self"):
            await websocket.send(json.dumps({
                "type": "self",
                "data": _current_state["self"],
            }, ensure_ascii=False))
    except Exception:
        pass

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type", "")
                if msg_type == "approve":
                    ap_id = data.get("id", "")
                    decision = data.get("decision", "no")
                    if ap_id:
                        _approval_queue.put({"id": ap_id, "decision": decision})
                        print(f"  [ws] approval: {ap_id} → {decision}")
                elif msg_type == "device_response":
                    dr_id = data.get("id", "")
                    if dr_id:
                        _device_queue.put(data)
                        print(f"  [ws] device response: {dr_id} (success={data.get('success')})")
                elif msg_type == "stream_frame":
                    # camera_stream の各フレーム到着（非同期）
                    b64 = data.get("data", "")
                    if b64:
                        try:
                            _save_stream_frame(b64, data.get("meta", {}))
                        except Exception as e:
                            print(f"  [ws] stream_frame 保存エラー: {e}")
                elif msg_type == "stream_end":
                    # ストリーム終了通知
                    print(f"  [ws] stream ended: frames={data.get('frame_count', '?')}")
                    _mark_stream_ended()
                elif msg_type == "test_tool":
                    # テストタブからのツール直接実行
                    tool_name = data.get("tool", "")
                    tool_args = data.get("args", {})
                    if tool_name:
                        _test_tool_queue.put({"tool": tool_name, "args": tool_args})
                        print(f"  [ws] test_tool: {tool_name} args={tool_args}")
                elif msg_type == "chat":
                    text = data.get("text", "").strip()
                    if text:
                        _chat_queue.put(text)
                        print(f"  [ws] chat received: {text[:50]}")
                elif msg_type == "select_profile":
                    name = data.get("name", "").strip()
                    if name:
                        _profile_queue.put(name)
                        print(f"  [ws] profile selected: {name}")
                elif msg_type == "get_llm_providers":
                    # プロバイダ一覧とメタ情報を返す（api_key は含めない）
                    try:
                        from core.auth import get_llm_provider_metadata
                        import json as _json
                        all_providers = ["lmstudio", "claude", "openai", "gemini"]
                        providers_info = []
                        for p in all_providers:
                            meta = get_llm_provider_metadata(p)
                            if meta:
                                providers_info.append(meta)
                            else:
                                providers_info.append({
                                    "provider": p,
                                    "base_url": "",
                                    "last_model": "",
                                    "has_key": False,
                                })
                        # 現在アクティブな provider/model を settings.json から
                        from core.config import LLM_SETTINGS
                        with open(LLM_SETTINGS, encoding="utf-8") as f:
                            active_cfg = _json.load(f)
                        broadcast({
                            "type": "llm_providers_list",
                            "providers": providers_info,
                            "active": {
                                "provider": active_cfg.get("provider", ""),
                                "model": active_cfg.get("model", ""),
                            },
                        })
                    except Exception as e:
                        print(f"  [ws] get_llm_providers error: {e}")
                elif msg_type == "set_llm":
                    # Android からの LLM 設定変更要求
                    try:
                        provider = data.get("provider", "").strip()
                        model = data.get("model", "").strip()
                        api_key = data.get("api_key", "")  # 空なら既存温存
                        base_url = data.get("base_url", "")
                        if not provider:
                            broadcast({"type": "set_llm_result", "ok": False, "error": "provider required"})
                        else:
                            # secrets.json を更新
                            from core.auth import save_llm_provider
                            err = save_llm_provider(provider, base_url=base_url, api_key=api_key, last_model=model)
                            if err:
                                broadcast({"type": "set_llm_result", "ok": False, "error": err})
                            else:
                                # settings.json を更新（active provider/model）
                                import json as _json
                                from core.config import LLM_SETTINGS
                                with open(LLM_SETTINGS, encoding="utf-8") as f:
                                    cfg = _json.load(f)
                                cfg["provider"] = provider
                                if model:
                                    cfg["model"] = model
                                with open(LLM_SETTINGS, "w", encoding="utf-8") as f:
                                    _json.dump(cfg, f, ensure_ascii=False, indent=2)
                                print(f"  [ws] set_llm: provider={provider} model={model} (next cycle で反映)")
                                broadcast({"type": "set_llm_result", "ok": True, "provider": provider, "model": model})
                    except Exception as e:
                        print(f"  [ws] set_llm error: {e}")
                        broadcast({"type": "set_llm_result", "ok": False, "error": str(e)[:200]})
                elif msg_type == "server_command":
                    # アプリからの pause/resume/stop コマンド
                    global _paused
                    action = data.get("action", "").strip()
                    if action == "pause":
                        _paused = True
                        print(f"  [ws] server_command: pause")
                        broadcast({"type": "server_command_result", "action": "pause", "ok": True})
                    elif action == "resume":
                        _paused = False
                        print(f"  [ws] server_command: resume")
                        broadcast({"type": "server_command_result", "action": "resume", "ok": True})
                    elif action == "stop":
                        print(f"  [ws] server_command: stop (graceful exit)")
                        broadcast({"type": "server_command_result", "action": "stop", "ok": True})
                        # 送信が flush されるよう少し待ってから exit
                        await asyncio.sleep(0.3)
                        import os as _os
                        _os._exit(0)
                    else:
                        print(f"  [ws] unknown server_command action: {action}")
                elif msg_type == "config":
                    pass
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    finally:
        _ws_clients.discard(websocket)
        print(f"  [ws] client disconnected ({len(_ws_clients)} total)")


_shutdown_flag = False


async def _send_loop():
    """キューからメッセージを取り出して全クライアントに送信するループ"""
    global _ws_clients
    while not _shutdown_flag:
        # キューをポーリング（ノンブロッキング）
        messages = []
        try:
            while True:
                messages.append(_send_queue.get_nowait())
        except queue.Empty:
            pass

        if messages and _ws_clients:
            dead = set()
            for text in messages:
                # snapshot を iterate (race 回避: await ws.send 中に別 coroutine が
                # _ws_clients から discard する可能性)。段階8 で asyncio.Lock 化予定。
                for ws in list(_ws_clients):
                    try:
                        await ws.send(text)
                    except Exception:
                        dead.add(ws)
            _ws_clients -= dead

        await asyncio.sleep(0.1)  # 100msごとにチェック


def stop_ws_server():
    """WebSocketサーバーをシャットダウンしてポートを解放する。
    server.py → main.py 引継ぎ時に呼び出す。"""
    global _shutdown_flag, _server_started
    _shutdown_flag = True
    _server_started = False
    # _send_loop が停止 → async with websockets.serve() が閉じる → ソケット解放
    # 呼び出し元で少し sleep して確実に解放を待つこと


def broadcast(msg: dict):
    """全接続クライアントにメッセージを送信（スレッドセーフ）"""
    if not _ws_clients:
        return
    text = json.dumps(msg, ensure_ascii=False)
    if msg.get("type") in ("log", "state", "e_values"):
        _ws_log_buffer.append(msg)
        if len(_ws_log_buffer) > _LOG_BUFFER_MAX:
            _ws_log_buffer.pop(0)
    _send_queue.put(text)


def broadcast_log(text: str):
    """ターミナルログ行をブロードキャスト"""
    broadcast({"type": "log", "text": text, "time": datetime.now().strftime("%H:%M:%S")})


def broadcast_state(state: dict):
    """状態スナップショットをブロードキャスト"""
    global _current_state
    _current_state = state
    broadcast({
        "type": "state",
        "entropy": state.get("entropy", 0.65),
        "energy": round(state.get("energy", 50), 1),
        "cycle_id": state.get("cycle_id", 0),
        "tool_level": state.get("tool_level", 0),
        "pressure": state.get("pressure", 0),
        "pending_count": len(state.get("pending", [])),
        "pending_items": [{"type": p.get("type",""), "content": p.get("content","")[:80], "id": p.get("id","")} for p in state.get("pending", [])[:5]],
        "paused": _paused,
    })


def broadcast_self(state: dict):
    """自己モデル + dispositionをブロードキャスト"""
    data = dict(state.get("self", {}))
    data["disposition"] = state.get("disposition", {})
    broadcast({
        "type": "self",
        "data": data,
    })


def get_pending_chats() -> list[str]:
    """未処理の外部入力を全て取得（main.pyが毎tick呼ぶ）"""
    messages = []
    try:
        while True:
            messages.append(_chat_queue.get_nowait())
    except queue.Empty:
        pass
    return messages


def request_approval(tool: str, preview: str, timeout_sec: int = 300) -> bool:
    """WebSocket経由で承認を要求し、応答を待つ。タイムアウトでdeny。
    スマートフォンが接続されてなければターミナルのinput()にフォールバック。"""
    import time as _time

    if not _ws_clients:
        # WebSocket未接続 → ターミナルフォールバック
        try:
            ans = input(f"  [{tool}] 実行しますか？ [y/N]: ").strip().lower()
            return ans == "y"
        except EOFError:
            return False

    ap_id = f"ap_{int(_time.time() * 1000) % 100000}"
    _pending_approval[ap_id] = {"tool": tool, "preview": preview}

    # 承認リクエスト送信
    broadcast({
        "type": "approval_request",
        "id": ap_id,
        "tool": tool,
        "preview": preview[:500],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"  [ws] 承認リクエスト送信: {ap_id} ({tool})")

    # 応答待ち（ポーリング）
    deadline = _time.time() + timeout_sec
    while _time.time() < deadline:
        try:
            resp = _approval_queue.get(timeout=1.0)
            if resp.get("id") == ap_id:
                _pending_approval.pop(ap_id, None)
                approved = resp.get("decision", "no").lower() in ("yes", "y", "approve")
                # 結果をブロードキャスト
                broadcast({
                    "type": "approval_result",
                    "id": ap_id,
                    "decision": "approved" if approved else "denied",
                })
                return approved
            else:
                _approval_queue.put(resp)  # 別のリクエストの応答なら戻す
        except queue.Empty:
            pass

    # タイムアウト
    _pending_approval.pop(ap_id, None)
    broadcast({"type": "approval_result", "id": ap_id, "decision": "timeout"})
    print(f"  [ws] 承認タイムアウト: {ap_id}")
    return False


_stream_ended_flag = False


def _save_stream_frame(b64: str, meta: dict):
    """WS から受け取ったフレームを保存し、ローリングバッファに追加する。
    WS ハンドラから呼ばれる（async context）"""
    global _stream_frames, _stream_frame_counter
    # BASE_DIR はランタイムで解決（import サイクル回避）
    from core.config import BASE_DIR
    captures_dir = BASE_DIR / "sandbox" / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    img_bytes = base64.b64decode(b64)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"stream_{ts}.jpg"
    path = captures_dir / filename
    path.write_bytes(img_bytes)
    rel = str(path.relative_to(BASE_DIR)).replace("\\", "/")

    with _stream_frames_lock:
        _stream_frames.append((rel, meta))
        if len(_stream_frames) > _STREAM_BUFFER_MAX:
            _stream_frames = _stream_frames[-_STREAM_BUFFER_MAX:]
        _stream_frame_counter += 1


def _mark_stream_ended():
    global _stream_ended_flag
    _stream_ended_flag = True


def get_stream_snapshot(consume_end: bool = True) -> tuple[list, int, bool]:
    """現在のバッファ内容・カウンタ・終了フラグを返す。
    戻り値: (frames: list[(rel_path, meta)], counter: int, ended: bool)
    consume_end=True のとき ended フラグを読み取り後にクリアする（main.py 用）。
    consume_end=False でフラグを保持（ツール内部から覗き見る用）。"""
    global _stream_ended_flag
    with _stream_frames_lock:
        frames = list(_stream_frames)
        counter = _stream_frame_counter
    ended = _stream_ended_flag
    if consume_end:
        _stream_ended_flag = False
    return frames, counter, ended


def clear_stream_buffer():
    """ストリーム終了後にバッファをクリア。"""
    global _stream_frames, _stream_frame_counter
    with _stream_frames_lock:
        _stream_frames = []
        _stream_frame_counter = 0


def send_device(action: str, params: dict = None) -> str:
    """デバイスにコマンドを送信する（fire-and-forget、応答待ちしない）。
    戻り値: 生成した device request id（必要なら応答追跡に使える）"""
    import time as _time
    dr_id = f"dr_{int(_time.time() * 1000) % 1000000}"
    broadcast({
        "type": "device_request",
        "id": dr_id,
        "action": action,
        "params": params or {},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"  [ws] device send (async): {dr_id} ({action})")
    return dr_id


def request_device(action: str, params: dict = None, timeout_sec: int = 60) -> dict | None:
    """スマートフォンに対してデバイス操作をリクエストし、応答を待つ。
    戻り値: {"success": bool, "data": ..., "meta": ..., "error": ...} or None(タイムアウト)"""
    import time as _time

    if not _ws_clients:
        return {"success": False, "error": "No client connected"}

    dr_id = f"dr_{int(_time.time() * 1000) % 1000000}"
    _pending_device[dr_id] = {"action": action, "params": params or {}}

    broadcast({
        "type": "device_request",
        "id": dr_id,
        "action": action,
        "params": params or {},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"  [ws] device request送信: {dr_id} ({action})")

    deadline = _time.time() + timeout_sec
    while _time.time() < deadline:
        try:
            resp = _device_queue.get(timeout=1.0)
            if resp.get("id") == dr_id:
                _pending_device.pop(dr_id, None)
                return resp
            else:
                _device_queue.put(resp)
        except queue.Empty:
            pass

    _pending_device.pop(dr_id, None)
    print(f"  [ws] device request タイムアウト: {dr_id}")
    return {"success": False, "error": "timeout"}


def broadcast_e_values(cycle_id: int, e1: float, e2: float, e3: float, e4: float, negentropy: float = 0):
    """E値をブロードキャスト"""
    broadcast({
        "type": "e_values",
        "cycle_id": cycle_id,
        "e1": e1, "e2": e2, "e3": e3, "e4": e4,
        "negentropy": negentropy,
    })


_server_started = False
_profile_queue: queue.Queue = queue.Queue()  # プロファイル選択キュー


def start_ws_server(host: str = "0.0.0.0", port: int = 8765):
    """WebSocketサーバーを別スレッドで起動。既に起動済みならスキップ。"""
    global _server_started
    if _server_started:
        print(f"  [ws] Server already running, skipping")
        return _get_token()
    _server_started = True

    import websockets

    async def _serve():
        # max_size を 64 MiB に拡張（camera_stream で複数フレームを受け取れるように。
        # デフォルト 1 MiB では 2 フレーム程度でオーバーする）
        async with websockets.serve(_ws_handler, host, port, max_size=2**26):
            print(f"  [ws] WebSocket server started on ws://{host}:{port}")
            print(f"  [ws] Token: {_get_token()}")
            await _send_loop()

    def _run():
        asyncio.run(_serve())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return _get_token()


def get_pending_profile() -> str | None:
    """プロファイル選択結果を取得（ノンブロッキング）"""
    try:
        return _profile_queue.get_nowait()
    except queue.Empty:
        return None


def get_pending_test_tools() -> list[dict]:
    """テストタブからのツール実行要求を全て取得（ノンブロッキング）"""
    results = []
    try:
        while True:
            results.append(_test_tool_queue.get_nowait())
    except queue.Empty:
        pass
    return results
