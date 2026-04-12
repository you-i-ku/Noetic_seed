"""Noetic_seed サーバーモード — アプリからprofile選択して起動
WSサーバーを先に起動し、アプリ接続→profile選択→main.py実行の順で動作。
従来のrun.bat（ターミナル選択）も引き続き使用可能。
"""
import sys
import os
import json
import time
import signal
from pathlib import Path

# Windows: Ctrl+C 即時終了
def _force_exit_on_sigint(_signum, _frame):
    print("\n[Ctrl+C] 強制終了します。", flush=True)
    os._exit(0)
signal.signal(signal.SIGINT, _force_exit_on_sigint)

# venv ブートストラップ
_here = Path(__file__).parent
_venv = _here / ".venv"
_is_win = sys.platform == "win32"
_venv_python = _venv / ("Scripts/python.exe" if _is_win else "bin/python")
try:
    if Path(sys.executable).resolve() != _venv_python.resolve():
        if _venv_python.exists():
            os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)
except Exception:
    pass

# WSサーバーを起動するために_templateのcoreを使う
# profile選択前なので、_templateのws_serverを直接import
sys.path.insert(0, str(_here / "profiles" / "_template"))

from core.ws_server import start_ws_server, broadcast, get_pending_profile, _ws_clients, stop_ws_server

PROFILES_DIR = _here / "profiles"


def get_profiles() -> list[dict]:
    """利用可能なプロファイル一覧"""
    profiles = []
    for d in sorted(PROFILES_DIR.iterdir()):
        if not d.is_dir() or d.name == "_template" or not (d / "main.py").exists():
            continue
        info = {"name": d.name}
        state_file = d / "state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                info["cycle_id"] = data.get("cycle_id", 0)
                info["entropy"] = round(data.get("entropy", 0.65), 3)
                info["energy"] = round(data.get("energy", 50), 1)
            except Exception:
                pass
        profiles.append(info)
    return profiles


def main():
    print("=== Noetic_seed Server ===")
    print()

    # WSサーバー起動
    token = start_ws_server()
    print()

    profiles = get_profiles()
    if not profiles:
        print("[ERROR] No profiles found. Copy _template to create one.")
        sys.exit(1)

    print(f"  Profiles: {[p['name'] for p in profiles]}")
    print(f"  Waiting for app connection...")
    print(f"  (or press Enter for terminal selection)")
    print()

    # アプリ接続 or ターミナル入力を待つ
    selected = None

    import threading
    _terminal_input = [None]

    def _wait_terminal():
        try:
            inp = input("  Terminal select [number or name]: ").strip()
            _terminal_input[0] = inp
        except (EOFError, KeyboardInterrupt):
            pass

    t = threading.Thread(target=_wait_terminal, daemon=True)
    t.start()

    while selected is None:
        # アプリからの接続チェック
        if _ws_clients:
            print("  [ws] Profile list sent to app")

            # アプリからの選択を待つ
            # 0.5秒毎に profile_list を再送（途中で新 client が来ても対応）
            while selected is None:
                if _ws_clients:
                    broadcast({
                        "type": "profile_list",
                        "profiles": profiles,
                    })
                prof = get_pending_profile()
                if prof:
                    if any(p["name"] == prof for p in profiles):
                        selected = prof
                        break
                    else:
                        print(f"  [ws] Unknown profile: {prof}")
                # ターミナルも並行チェック
                if _terminal_input[0] is not None:
                    break
                time.sleep(0.5)

        # ターミナル入力チェック
        if _terminal_input[0] is not None and selected is None:
            inp = _terminal_input[0]
            try:
                idx = int(inp) - 1
                if 0 <= idx < len(profiles):
                    selected = profiles[idx]["name"]
            except ValueError:
                if any(p["name"] == inp for p in profiles):
                    selected = inp

            if selected is None:
                print(f"  [ERROR] Invalid: {inp}")
                print(f"  Available: {[p['name'] for p in profiles]}")
                _terminal_input[0] = None
                t = threading.Thread(target=_wait_terminal, daemon=True)
                t.start()

        time.sleep(0.3)

    profile_dir = PROFILES_DIR / selected
    main_py = profile_dir / "main.py"
    if not main_py.exists():
        print(f"[ERROR] main.py not found in {profile_dir}")
        sys.exit(1)

    print(f"\n  Starting profile: {selected}")
    print(f"  (WSシャットダウン → subprocess で main.py 起動。アプリは自動再接続します)\n")

    # WebSocket サーバーを綺麗に停止してポート 8765 を解放
    stop_ws_server()
    time.sleep(0.8)  # _send_loop が停止してソケットが解放されるまで待つ

    # subprocess.run で main.py を起動（親 server.py は main.py 終了まで待つ）
    # os.execv はプロファイル名にスペース含む場合に引数分割されるため使えない
    os.chdir(str(profile_dir))
    import subprocess
    try:
        result = subprocess.run([sys.executable, str(main_py)])
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Ctrl+C] 終了します。")
        sys.exit(0)
