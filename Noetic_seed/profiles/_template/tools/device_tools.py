"""デバイスツール — スマートフォン経由のカメラ・センサー等

WebSocket device_request/response プロトコルを使用。
camera_stream は非同期実行: 承認後 Android に fire-and-forget でコマンドを送り、
Android が各フレームを stream_frame メッセージで送信してくる（ws_server が蓄積）。
"""
import base64
import time
from datetime import datetime
from pathlib import Path
from core.config import BASE_DIR
from core.state import load_state, save_state
from core.ws_server import request_device, request_approval, send_device, clear_stream_buffer, get_stream_snapshot


CAPTURES_DIR = BASE_DIR / "sandbox" / "captures"
AUDIO_DIR = BASE_DIR / "sandbox" / "audio"


def _build_approval_preview(tool: str, args_summary: str, args: dict) -> str:
    """承認通知のプレビュー文を組み立てる。
    ツール固有の args 要約 + AI の intent + デバイス所有者へのメッセージ message を含める。
    intent と message は承認側が判断する上で必須情報。
    （message= は self_modify の content= との衝突を避けるため message に統一）"""
    intent = args.get("intent", "").strip()
    message = args.get("message", "").strip()
    lines = [f"[{tool}] {args_summary}"]
    if intent:
        lines.append(f"意図: {intent}")
    if message:
        lines.append(f"メッセージ: {message}")
    lines.append("承認しますか？")
    return "\n".join(lines)


def _camera_stream(args) -> str:
    """端末のカメラ経由で連続撮影を非同期に開始する。
    Android 側でカメラストリームが始まり、各フレームが到着するたびに AI の視覚入力に入る。
    AI は別サイクルで camera_stream_stop を呼んで停止できる。
    単発撮影は frames=1 で行える。

    動作:
    - 承認（同期ブロック、ここだけ main loop を待たせる）
    - 承認OK → Android に device_request を送る（応答待ちしない）
    - state['stream_active'] = True をセット
    - 即座に return（main loop は次サイクルへ）
    - Android が各フレームを stream_frame メッセージで送信
    - ws_server 側でローリングバッファに蓄積、main.py が次サイクルで pending_images として参照
    - AI は frames 数が満ちるか自分で camera_stream_stop を呼ぶまで観察を続ける
    """
    facing = args.get("facing", "back").strip().lower()
    if facing not in ("front", "back"):
        facing = "back"

    try:
        frames = int(args.get("frames", "5"))
    except (ValueError, TypeError):
        frames = 5
    try:
        interval_sec = float(args.get("interval_sec", "1.0"))
    except (ValueError, TypeError):
        interval_sec = 1.0

    if not (1 <= frames <= 30):
        return "エラー: frames は 1-30 の範囲で指定してください"
    if not (0.3 <= interval_sec <= 5.0):
        return "エラー: interval_sec は 0.3-5.0 の範囲で指定してください"

    state = load_state()
    if state.get("stream_active"):
        return "エラー: 既に camera_stream がアクティブです。camera_stream_stop で停止してから再開してください"

    estimated_sec = frames * interval_sec
    summary = f"facing={facing} frames={frames} interval={interval_sec}s (約{estimated_sec:.1f}秒)"
    preview = _build_approval_preview("camera_stream", summary, args)
    if not request_approval("camera_stream", preview, timeout_sec=60):
        return "キャンセル: 撮影は承認されませんでした"

    # 前回のストリームフレームが残ってたらクリア
    clear_stream_buffer()

    # Android に非同期で送信（応答を待たない）
    dr_id = send_device(
        "camera_stream",
        {"facing": facing, "frames": frames, "interval_sec": interval_sec},
    )

    state = load_state()  # 承認中に state が変わっている可能性があるので再読込
    state["stream_active"] = True
    state["stream_id"] = dr_id
    state["stream_params"] = {"facing": facing, "frames": frames, "interval_sec": interval_sec}
    save_state(state)

    # 最初の 1 フレームが到着するのを最大 timeout 秒待つ（ハイブリッド同期）
    # 到着したら LLM で描写を取得して結果に含める → E 値評価が意味を持つ
    first_frame_wait_sec = 12.0  # カメラ起動 + 1 フレームキャプチャ + 余裕
    poll_interval = 0.3
    waited = 0.0
    first_rel = None
    first_meta = None
    while waited < first_frame_wait_sec:
        stream_frames, _counter, _ended = get_stream_snapshot(consume_end=False)
        if stream_frames:
            first_rel, first_meta = stream_frames[0]
            break
        time.sleep(poll_interval)
        waited += poll_interval

    if first_rel is None:
        # タイムアウト。ストリーム自体は進行中の可能性あり（カメラ起動が遅い等）
        return (
            f"ストリーム送信済み: facing={facing} frames={frames} interval={interval_sec}s\n"
            f"最初のフレームが {first_frame_wait_sec}秒以内に到着しませんでした（カメラ起動中の可能性）。"
            f"後続サイクルで視覚入力として届く場合があります。"
        )

    # 最初のフレームを LLM に描写させる
    first_full = BASE_DIR / first_rel
    intent = args.get("intent", "").strip()
    if intent:
        describe_prompt = (
            f"カメラストリームの最初のフレームです。\n"
            f"目的: {intent}\n\n"
            f"この画像を 1-2 文で簡潔に描写してください。目的に関連する情報を優先してください。"
        )
    else:
        describe_prompt = (
            "カメラストリームの最初のフレームです。\n"
            "この画像を 1-2 文で簡潔に描写してください。"
        )

    try:
        from core.llm import call_llm
        description = call_llm(
            describe_prompt,
            max_tokens=300,
            temperature=0.7,
            image_paths=[str(first_full)],
        ).strip()
    except Exception as e:
        description = f"（描写取得失敗: {e}）"

    return (
        f"ストリーム開始成功: facing={facing} frames={frames} interval={interval_sec}s\n"
        f"最初のフレーム観察: {description}\n"
        f"観察は継続中（後続フレームは次サイクル以降で視覚入力に入る）。camera_stream_stop で能動停止できます。"
    )


def _camera_stream_stop(args) -> str:
    """アクティブな camera_stream を停止する。"""
    state = load_state()
    if not state.get("stream_active"):
        return "エラー: アクティブな camera_stream がありません"

    # 停止時点のバッファ状況を取得
    frames, counter, _ended = get_stream_snapshot(consume_end=False)
    frame_count = len(frames)

    send_device("camera_stream_stop", {"stream_id": state.get("stream_id", "")})
    state["stream_active"] = False
    state["stream_id"] = None
    state["stream_params"] = None
    save_state(state)

    return (
        f"camera_stream 停止命令を送信しました。"
        f"停止時点のバッファに {frame_count}フレーム 観察済み（累計 counter={counter}）。"
    )
