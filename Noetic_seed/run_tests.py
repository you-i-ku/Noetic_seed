"""Noetic_seed 単体スクリプト test runner。

profiles/_template/tests/test_*.py を 1 本ずつ独立プロセスで実行し、
exit code でパス/フェイル判定。Noetic_seed の test は sys.exit(0/1) で
結果を返す単体スクリプト運用なので、pytest の代わりに本 runner を使う。

Usage:
    .venv/Scripts/python.exe run_tests.py
    .venv/Scripts/python.exe run_tests.py test_world_model    # 部分一致
"""
import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent
VENV_PY = ROOT / ".venv/Scripts/python.exe"
PROFILE = ROOT / "profiles/_template"
TESTS_DIR = PROFILE / "tests"

if not VENV_PY.exists():
    sys.exit(f"venv python not found: {VENV_PY}")

filter_arg = sys.argv[1] if len(sys.argv) > 1 else ""
tests = sorted(TESTS_DIR.glob("test_*.py"))
if filter_arg:
    tests = [t for t in tests if filter_arg in t.stem]

passed, failed = [], []
for t in tests:
    rel = t.relative_to(PROFILE)
    r = subprocess.run(
        [str(VENV_PY), str(rel)],
        cwd=str(PROFILE),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode == 0:
        passed.append(t.name)
        print(f"OK   {t.name}")
    else:
        failed.append((t.name, r.stdout, r.stderr))
        print(f"FAIL {t.name}  (exit {r.returncode})")

print(f"\n{len(passed)} passed, {len(failed)} failed ({len(tests)} total)")
if failed:
    print("\n--- Failure details (tail) ---")
    for name, out, err in failed:
        print(f"\n=== {name} ===")
        tail = (out + err).strip().splitlines()[-10:]
        print("\n".join(tail))
sys.exit(1 if failed else 0)
