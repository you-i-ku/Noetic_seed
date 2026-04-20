"""UI出力ツール (段階6-C v3: 動的 channel 対応)"""
from core.ws_server import broadcast, broadcast_log


def _output_display(args):
    content = args.get("content", "")
    if not content:
        return "エラー: contentを指定してください"
    channel = str(args.get("channel", "")).strip()
    if not channel:
        return ("エラー: channel を指定してください "
                "(WM.channels を観察して利用可能な channel を確認)")

    broadcast({"type": "reply", "content": content, "channel": channel})
    broadcast_log(f"  [output_display:{channel}] → {content[:100]}")
    # 段階9 fix 5: content truncation 撤去。log.result にも完全 content を保存
    # することで、iku が自分の発言を「途切れた」と誤認するバグを根治。prompt
    # 側の _pack_log_block tier cap が表示省略を担当し、超過時は [表示上 N/M字。
    # ツール実行時は完全取得済] marker で LLM に「省略は表示だけ」を伝える。
    return f"送信完了 ({channel}): {content}"
