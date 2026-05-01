"""make_bash_path_guard_hook の test (段階12 Step 7.5 ②, PLAN §3-5-2 ②)。

検証ケース:
  - bash + write 系コマンド + 絶対パス profile 外 → deny (rm / mv / cp / dd / tee)
  - bash + write 系 + 絶対パス profile 内 → allow
  - bash + write 系 + 相対パス → allow (絶対パスじゃないので)
  - bash + read 系 (cat / ls) + 絶対パス profile 外 → allow (read は対象外)
  - bash + redirect (>) + 絶対パス profile 外 → deny
  - 非 bash tool → allow (passthrough)
  - shlex 失敗 (構文エラー) → allow (validation hook に委譲)

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_bash_path_guard.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.hooks import make_bash_path_guard_hook


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup():
    return Path(tempfile.mkdtemp(prefix="noetic_bash_guard_test_"))


def test_rm_outside_profile_denied():
    print("== bash(rm /etc/passwd) は profile 外で deny ==")
    root = _setup()
    hook = make_bash_path_guard_hook(root)
    r = hook("bash", {"command": "rm /etc/passwd"})
    return all([
        _assert(r.denied, "denied=True"),
        _assert(any("profile 境界外" in m for m in r.messages),
                "境界外 message"),
    ])


def test_mv_outside_profile_denied():
    print("== bash(mv file /tmp/x) は profile 外で deny ==")
    root = _setup()
    hook = make_bash_path_guard_hook(root)
    r = hook("bash", {"command": "mv /tmp/source.txt /tmp/dest.txt"})
    return _assert(r.denied, "denied=True (mv も write 系)")


def test_cp_outside_profile_denied():
    print("== bash(cp file /etc/x) は profile 外で deny ==")
    root = _setup()
    hook = make_bash_path_guard_hook(root)
    r = hook("bash", {"command": "cp /etc/source /etc/dest"})
    return _assert(r.denied, "denied=True")


def test_redirect_outside_profile_denied():
    print("== bash(echo x > /etc/foo) は profile 外で deny ==")
    root = _setup()
    hook = make_bash_path_guard_hook(root)
    r = hook("bash", {"command": "echo x > /etc/foo"})
    return _assert(r.denied, "denied=True (> redirect は write 扱い)")


def test_write_inside_profile_allowed():
    print("== bash(rm <profile>/sandbox/x) は profile 内で allow ==")
    root = _setup()
    hook = make_bash_path_guard_hook(root)
    target = root / "sandbox" / "x.txt"
    r = hook("bash", {"command": f"rm {target}"})
    return _assert(not r.denied, "denied=False (profile 内なので OK)")


def test_relative_path_allowed():
    print("== bash(rm sandbox/x.txt) は相対パスで allow ==")
    root = _setup()
    hook = make_bash_path_guard_hook(root)
    r = hook("bash", {"command": "rm sandbox/x.txt"})
    return _assert(not r.denied, "denied=False (相対 path は対象外)")


def test_read_command_outside_profile_allowed():
    print("== bash(cat /etc/passwd) は read 系で allow ==")
    root = _setup()
    hook = make_bash_path_guard_hook(root)
    r = hook("bash", {"command": "cat /etc/passwd"})
    return _assert(not r.denied, "denied=False (read 系は対象外、Step 7.5 ② scope 外)")


def test_ls_outside_profile_allowed():
    print("== bash(ls /etc) は read 系で allow ==")
    root = _setup()
    hook = make_bash_path_guard_hook(root)
    r = hook("bash", {"command": "ls /etc"})
    return _assert(not r.denied, "denied=False (ls は read 系)")


def test_non_bash_tool_passthrough():
    print("== 非 bash tool (write_file 等) は passthrough ==")
    root = _setup()
    hook = make_bash_path_guard_hook(root)
    r = hook("write_file", {"path": "/etc/passwd", "content": "x"})
    return _assert(not r.denied, "denied=False (file_access_guard に委譲)")


def test_shlex_syntax_error_passthrough():
    print("== shlex 構文エラーは passthrough (validation に委譲) ==")
    root = _setup()
    hook = make_bash_path_guard_hook(root)
    r = hook("bash", {"command": 'echo "unclosed quote'})
    return _assert(not r.denied, "denied=False (shlex 失敗時は他 hook に委譲)")


if __name__ == "__main__":
    groups = [
        ("rm + 絶対 profile 外 deny", test_rm_outside_profile_denied),
        ("mv + 絶対 profile 外 deny", test_mv_outside_profile_denied),
        ("cp + 絶対 profile 外 deny", test_cp_outside_profile_denied),
        ("redirect > + profile 外 deny", test_redirect_outside_profile_denied),
        ("write + profile 内 allow", test_write_inside_profile_allowed),
        ("相対パスは allow", test_relative_path_allowed),
        ("cat (read) + profile 外 allow",
         test_read_command_outside_profile_allowed),
        ("ls (read) + profile 外 allow", test_ls_outside_profile_allowed),
        ("非 bash tool passthrough", test_non_bash_tool_passthrough),
        ("shlex 構文エラーは passthrough", test_shlex_syntax_error_passthrough),
    ]
    results = []
    for _label, fn in groups:
        print()
        ok = fn()
        results.append((_label, ok))
    print()
    print("=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for _label, ok in results:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {_label}")
    print(f"\n  {passed}/{total} groups passed")
    sys.exit(0 if passed == total else 1)
