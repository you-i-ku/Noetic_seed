"""shell tests (bash / PowerShell / REPL).

PowerShell は Windows 環境で利用可能なら検証、無ければスキップ。
"""
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.registry import ToolRegistry
from core.runtime.tools import shell


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _skip(label):
    print(f"  [SKIP] {label}")
    return True


def _get_reg():
    reg = ToolRegistry()
    shell.register(reg)
    return reg


def test_bash_echo():
    print("== bash: echo ==")
    if not shutil.which("bash"):
        return _skip("bash not in PATH")
    reg = _get_reg()
    out = reg.execute("bash", {"command": "echo hello_bash"})
    return all([
        _assert("hello_bash" in out, "出力"),
        _assert("[stdout]" in out, "stdout section"),
    ])


def test_bash_exit_code():
    print("== bash: exit code ==")
    if not shutil.which("bash"):
        return _skip("bash not in PATH")
    reg = _get_reg()
    out = reg.execute("bash", {"command": "exit 42"})
    return _assert("exit code: 42" in out, "exit code 42")


def test_bash_stderr():
    print("== bash: stderr ==")
    if not shutil.which("bash"):
        return _skip("bash not in PATH")
    reg = _get_reg()
    out = reg.execute("bash", {"command": "echo err >&2"})
    return all([
        _assert("[stderr]" in out, "stderr section"),
        _assert("err" in out, "err content"),
    ])


def test_bash_timeout():
    print("== bash: timeout ==")
    if not shutil.which("bash"):
        return _skip("bash not in PATH")
    reg = _get_reg()
    out = reg.execute("bash", {"command": "sleep 5", "timeout": 1})
    return _assert("timeout" in out.lower(), "timeout エラー")


def test_bash_empty_command():
    print("== bash: empty command ==")
    reg = _get_reg()
    out = reg.execute("bash", {"command": ""})
    return _assert("required" in out.lower(), "empty rejected")


def test_bash_background():
    print("== bash: run_in_background ==")
    if not shutil.which("bash"):
        return _skip("bash not in PATH")
    reg = _get_reg()
    out = reg.execute("bash", {"command": "sleep 10",
                                "run_in_background": True})
    ok = "backgroundTaskId" in out
    # cleanup
    m = re.search(r"bg_[a-f0-9]+", out)
    if m:
        t = shell.get_background_task(m.group(0))
        if t and t["proc"].poll() is None:
            try:
                t["proc"].kill()
            except Exception:
                pass
    return _assert(ok, "task id 返却")


def test_powershell_optional():
    print("== PowerShell: optional ==")
    if not (shutil.which("pwsh") or shutil.which("powershell")):
        return _skip("powershell not in PATH")
    reg = _get_reg()
    out = reg.execute("PowerShell", {"command": "Write-Output ps_test"})
    return _assert("ps_test" in out, "出力")


def test_repl_python():
    print("== REPL: python ==")
    if not (shutil.which("python3") or shutil.which("python")):
        return _skip("python not in PATH")
    reg = _get_reg()
    out = reg.execute("REPL", {"code": "print(2+3)", "language": "python"})
    return _assert("5" in out, "計算結果")


def test_repl_unsupported_lang():
    print("== REPL: unsupported language ==")
    reg = _get_reg()
    out = reg.execute("REPL", {"code": "x", "language": "brainfuck"})
    return _assert("unsupported" in out.lower(), "拒否")


def test_repl_empty_code():
    print("== REPL: empty code ==")
    reg = _get_reg()
    out = reg.execute("REPL", {"code": "", "language": "python"})
    return _assert("required" in out.lower(), "拒否")


def test_register():
    print("== register: 3 tool ==")
    reg = ToolRegistry()
    shell.register(reg)
    return _assert(
        {"bash", "PowerShell", "REPL"}.issubset(set(reg.all_names())),
        "3 tool 全登録",
    )


def main():
    tests = [
        test_bash_echo, test_bash_exit_code, test_bash_stderr,
        test_bash_timeout, test_bash_empty_command, test_bash_background,
        test_powershell_optional,
        test_repl_python, test_repl_unsupported_lang, test_repl_empty_code,
        test_register,
    ]
    print(f"Running {len(tests)} test groups...\n")
    passed = 0
    for t in tests:
        if t():
            passed += 1
        print()
    print(f"=== Result: {passed}/{len(tests)} test groups passed ===")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
