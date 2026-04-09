"""UI出力ツール"""
from core.ws_server import broadcast, broadcast_log


def _output_display(args):
    content = args.get("content", "")
    if not content:
        return "エラー: contentを指定してください"
    target = args.get("target", "user").strip()

    if target == "user":
        broadcast({"type": "reply", "content": content})
        broadcast_log(f"  [output_display] → {content[:100]}")
        return f"送信完了: {content[:80]}"
    else:
        return f"エラー: 未対応のtarget '{target}'。現在はuser のみ対応。"
