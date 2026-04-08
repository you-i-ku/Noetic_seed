"""プロファイル選択スクリプト（run.batから呼ばれる）"""
import sys
from pathlib import Path

base = Path(__file__).parent
profiles_dir = base / "profiles"

# _template以外のプロファイルを取得
profiles = sorted([
    d.name for d in profiles_dir.iterdir()
    if d.is_dir() and d.name != "_template" and (d / "main.py").exists()
])

if not profiles:
    print("[ERROR] No profiles found. Copy _template to create one.")
    sys.exit(1)

print("=== iku - profile select ===")
print()
for i, name in enumerate(profiles, 1):
    state_file = profiles_dir / name / "state.json"
    status = ""
    if state_file.exists():
        try:
            import json
            data = json.loads(state_file.read_text(encoding="utf-8"))
            cycle = data.get("cycle_id", 0)
            entropy = data.get("entropy", "?")
            if isinstance(entropy, float):
                entropy = f"{entropy:.3f}"
            status = f"  (cycle:{cycle} entropy:{entropy})"
        except Exception:
            pass
    print(f"  {i}. {name}{status}")

print()
try:
    choice = input(f"Select [1]: ").strip()
except (EOFError, KeyboardInterrupt):
    sys.exit(1)

if not choice:
    choice = "1"

try:
    idx = int(choice) - 1
    if 0 <= idx < len(profiles):
        selected = profiles[idx]
    else:
        print(f"[ERROR] Invalid selection: {choice}")
        sys.exit(1)
except ValueError:
    # 番号じゃなくて名前を直接入力した場合
    if choice in profiles:
        selected = choice
    else:
        print(f"[ERROR] Profile '{choice}' not found.")
        sys.exit(1)

# 選択結果をtmpファイルに書き出し（run.batが読む）
tmp = base / "_last_profile.tmp"
tmp.write_text(selected, encoding="utf-8")
print(f"  -> {selected}")
