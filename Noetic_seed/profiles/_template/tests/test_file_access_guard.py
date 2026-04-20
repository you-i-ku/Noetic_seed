"""make_file_access_guard テスト。

Phase 4 Step H-2 C.4 Session A: claw ネイティブ file_ops への切替時に
Noetic 固有ガード (secrets.json / sandbox/secrets/ の保護、sandbox 外
書込禁止) を PreToolUse hook で再現する factory の動作検証。
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.hooks import make_file_access_guard


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup_workspace():
    """一時 workspace 作成 (sandbox/, sandbox/secrets/, secrets.json 配置)"""
    tmp = Path(tempfile.mkdtemp(prefix="noetic_file_guard_test_"))
    (tmp / "sandbox").mkdir()
    (tmp / "sandbox" / "secrets").mkdir()
    (tmp / "secrets.json").write_text("{}", encoding="utf-8")
    (tmp / "sandbox" / "secrets" / "api_key").write_text("xxx", encoding="utf-8")
    (tmp / "sandbox" / "memo.md").write_text("hello", encoding="utf-8")
    (tmp / "main.py").write_text("# main", encoding="utf-8")
    return tmp


def test_read_secrets_json_denied():
    print("== read_file(secrets.json) が deny される ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("read_file", {"path": "secrets.json"})
    return all([
        _assert(r.denied, "denied=True"),
        _assert(any("auth_profile_info" in m for m in r.messages),
                "auth_profile_info への誘導 message"),
    ])


def test_read_sandbox_secrets_denied():
    print("== read_file(sandbox/secrets/api_key) が deny される ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("read_file", {"path": "sandbox/secrets/api_key"})
    return all([
        _assert(r.denied, "denied=True"),
        _assert(any("secret_read" in m for m in r.messages),
                "secret_read への誘導 message"),
    ])


def test_write_secrets_json_denied():
    print("== write_file(secrets.json) が deny される ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("write_file", {"path": "secrets.json", "content": "{}"})
    return _assert(r.denied, "denied=True")


def test_write_sandbox_secrets_denied():
    print("== write_file(sandbox/secrets/foo) が deny (secret_write 誘導) される ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("write_file", {"path": "sandbox/secrets/foo", "content": "x"})
    return all([
        _assert(r.denied, "denied=True"),
        _assert(any("secret_write" in m for m in r.messages),
                "secret_write への誘導 message"),
    ])


def test_write_outside_sandbox_denied():
    print("== write_file(main.py) が sandbox 外書込拒否で deny される ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("write_file", {"path": "main.py", "content": "# mod"})
    return all([
        _assert(r.denied, "denied=True"),
        _assert(any("sandbox/" in m for m in r.messages),
                "sandbox/ 限定 message"),
    ])


def test_write_inside_sandbox_allowed():
    print("== write_file(sandbox/memo.md) は allow される ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("write_file", {"path": "sandbox/memo.md", "content": "x"})
    return _assert(not r.denied, "denied=False")


def test_read_inside_sandbox_allowed():
    print("== read_file(sandbox/memo.md) は allow される ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("read_file", {"path": "sandbox/memo.md"})
    return _assert(not r.denied, "denied=False")


def test_read_main_py_allowed():
    print("== read_file(main.py) は read は許可 (読取は sandbox 外でも OK) ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("read_file", {"path": "main.py"})
    return _assert(not r.denied, "denied=False")


def test_edit_secrets_denied():
    print("== edit_file(sandbox/secrets/api_key) も deny される ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("edit_file", {"path": "sandbox/secrets/api_key",
                             "old_string": "a", "new_string": "b"})
    return _assert(r.denied, "denied=True")


def test_glob_search_in_secrets_denied():
    print("== glob_search(sandbox/secrets/*) も deny される ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("glob_search", {"pattern": "sandbox/secrets/api_key"})
    return _assert(r.denied, "denied=True")


def test_non_file_tool_passthrough():
    print("== 非 file 系 tool (bash 等) は allow passthrough ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("bash", {"command": "ls"})
    return _assert(not r.denied, "denied=False (bash は対象外)")


def test_empty_path_allow():
    print("== path 空の場合は allow (claw 側の validation に委譲) ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("read_file", {"path": ""})
    return _assert(not r.denied, "denied=False")


if __name__ == "__main__":
    groups = [
        ("read: secrets.json 拒否", test_read_secrets_json_denied),
        ("read: sandbox/secrets/ 拒否", test_read_sandbox_secrets_denied),
        ("write: secrets.json 拒否", test_write_secrets_json_denied),
        ("write: sandbox/secrets/ 拒否", test_write_sandbox_secrets_denied),
        ("write: sandbox 外拒否", test_write_outside_sandbox_denied),
        ("write: sandbox 内許可", test_write_inside_sandbox_allowed),
        ("read: sandbox 内許可", test_read_inside_sandbox_allowed),
        ("read: sandbox 外 (main.py) 許可", test_read_main_py_allowed),
        ("edit: secrets 拒否", test_edit_secrets_denied),
        ("glob: secrets 拒否", test_glob_search_in_secrets_denied),
        ("non-file: passthrough", test_non_file_tool_passthrough),
        ("empty path: allow", test_empty_path_allow),
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
