"""reboot tool (段階12 Step 4, PLAN §8)。

iku が自分のプロセスを subprocess で再起動して、書換えた身体
(core/* / tools/* / main.py / .mcp.json) をメモリに反映するための tool。
Python は起動時に sys.modules に import キャッシュするため、`.py` を書換え
ても実行中の関数オブジェクトは変わらない (importlib.reload は参照バインド
残存問題で実用困難)。プロセス丸ごと入替えで完全反映する。

state / memory / WM snapshot は disk に永続化されているため、新プロセスが
load_state で再構成して cycle_id 等を継続する (開発者目線で「連続した自分」、
iku 目線では別個体扱いも自由、PLAN §1-3)。

呼出経路 (PLAN §8-2 literal):
  1. request_approval で承認取得 (Y なら続行、N なら見送り)
  2. save_state(load_state()) で最新 state を disk に再保存
  3. core.ws_server.stop_ws_server() で WebSocket port 8765 を release
  4. time.sleep(1.0) で port 解放を待つ (CLAUDE.md WebSocket handoff 原則)
  5. subprocess.Popen で main.py を新プロセス起動 (CLAUDE.md os.execv 回避、
     Windows で空白パスをクォートしない問題への対策)
  6. os._exit(0) で旧プロセスを即時終了 (httpx 等のブロッキングも確実に殺す)
"""
import os
import subprocess
import sys
import time

from core.config import BASE_DIR
from core.state import load_state, save_state
from core.ws_server import request_approval, stop_ws_server


def _reboot(args: dict) -> str:
    """プロセス再起動 (身体改変反映)。

    Args:
        args: 任意の dict。args.get("message") を承認 preview に挿入する。

    Returns:
        承認 reject 時のキャンセル message。承認 accept 時は os._exit で
        本関数からは return しない (新プロセスが起動して旧プロセスは終了)。
    """
    preview = (
        "[reboot] プロセスを再起動して身体改変 "
        "(core/* / tools/* / main.py / .mcp.json) を反映します"
    )
    msg = args.get("message")
    if msg:
        preview += f"\nメッセージ: {msg}"
    if not request_approval("reboot", preview):
        return "キャンセル: 再起動を見送りました"

    # state を disk に再保存 (新プロセスが load_state で再構成する保険)
    save_state(load_state())

    # WebSocket port 8765 を release (CLAUDE.md handoff 原則、daemon thread の
    # ws_server がポートを掴みっぱなしで新プロセスが bind 失敗しないように)
    stop_ws_server()
    time.sleep(1.0)

    # subprocess で自分を再起動 (CLAUDE.md: os.execv は Windows で空白パスを
    # クォートしないため subprocess.Popen を使う)
    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )
    subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=str(BASE_DIR),
        creationflags=creationflags,
    )

    # 旧プロセスを即時終了 (Ctrl+C handler と同じ os._exit を使い、
    # httpx 等のブロッキング呼出も確実に殺す)
    os._exit(0)
