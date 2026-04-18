"""UI出力ツール (段階6-C v3: 動的 channel 対応)"""
from core.ws_server import broadcast, broadcast_log


def _output_display(args):
    content = args.get("content", "")
    if not content:
        return "エラー: contentを指定してください"
    channel = str(args.get("channel", "device")).strip() or "device"

    broadcast({"type": "reply", "content": content, "channel": channel})
    broadcast_log(f"  [output_display:{channel}] → {content[:100]}")
    return f"送信完了 ({channel}): {content[:80]}"
