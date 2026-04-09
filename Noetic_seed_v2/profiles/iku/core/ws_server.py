"""WebSocketサーバー（UIアプリとの双方向通信）"""
import asyncio
import json
import threading
import queue
import secrets
from datetime import datetime

_ws_clients: set = set()
_ws_token: str = ""
_ws_log_buffer: list = []
_LOG_BUFFER_MAX = 100
_send_queue: queue.Queue = queue.Queue()
_current_state: dict = {}
_chat_queue: queue.Queue = queue.Queue()  # ユーザー入力キュー（main.pyが読む）


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
        if _current_state:
            await websocket.send(json.dumps({
                "type": "state",
                "entropy": _current_state.get("entropy", 0.65),
                "energy": round(_current_state.get("energy", 50), 1),
                "cycle_id": _current_state.get("cycle_id", 0),
                "tool_level": _current_state.get("tool_level", 0),
                "pressure": _current_state.get("pressure", 0),
            }, ensure_ascii=False))
            await websocket.send(json.dumps({
                "type": "self",
                "data": _current_state.get("self", {}),
            }, ensure_ascii=False))
    except Exception:
        pass

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type", "")
                if msg_type == "approve":
                    pass
                elif msg_type == "chat":
                    text = data.get("text", "").strip()
                    if text:
                        _chat_queue.put(text)
                        print(f"  [ws] chat received: {text[:50]}")
                elif msg_type == "config":
                    pass
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    finally:
        _ws_clients.discard(websocket)
        print(f"  [ws] client disconnected ({len(_ws_clients)} total)")


async def _send_loop():
    """キューからメッセージを取り出して全クライアントに送信するループ"""
    global _ws_clients
    while True:
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
                for ws in _ws_clients:
                    try:
                        await ws.send(text)
                    except Exception:
                        dead.add(ws)
            _ws_clients -= dead

        await asyncio.sleep(0.1)  # 100msごとにチェック


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
    })


def broadcast_self(state: dict):
    """自己モデルをブロードキャスト"""
    broadcast({
        "type": "self",
        "data": state.get("self", {}),
    })


def get_pending_chats() -> list[str]:
    """未処理のユーザー入力を全て取得（main.pyが毎tick呼ぶ）"""
    messages = []
    try:
        while True:
            messages.append(_chat_queue.get_nowait())
    except queue.Empty:
        pass
    return messages


def broadcast_e_values(cycle_id: int, e1: float, e2: float, e3: float, e4: float, negentropy: float = 0):
    """E値をブロードキャスト"""
    broadcast({
        "type": "e_values",
        "cycle_id": cycle_id,
        "e1": e1, "e2": e2, "e3": e3, "e4": e4,
        "negentropy": negentropy,
    })


def start_ws_server(host: str = "0.0.0.0", port: int = 8765):
    """WebSocketサーバーを別スレッドで起動"""
    import websockets

    async def _serve():
        async with websockets.serve(_ws_handler, host, port):
            print(f"  [ws] WebSocket server started on ws://{host}:{port}")
            print(f"  [ws] Token: {_get_token()}")
            # 送信ループも並行実行
            await _send_loop()

    def _run():
        asyncio.run(_serve())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return _get_token()
