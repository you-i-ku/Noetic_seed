"""make_file_access_guard テスト。

Phase 4 Step H-2 C.4 Session A: claw ネイティブ file_ops への切替時に
Noetic 固有ガード (secrets.json / sandbox/secrets/ の保護、sandbox 外
書込禁止) を PreToolUse hook で再現する factory の動作検証。

段階12 Step 2 (PLAN §3-2): 「sandbox 外書込禁止」を「profile 外書込禁止」
に拡張。profile 配下すべてが身体改変対象になり、`.venv/` も身体拡張範囲
として allow される。symbolic link / .. による境界抜けは Path.resolve()
canonical 化で塞ぐ。
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


def test_write_outside_profile_denied():
    """段階12 Step 2: profile 境界外への絶対パス書込みは deny。
    旧 test_write_outside_sandbox_denied を rename + 内容更新 (sandbox 外
    deny ではなく profile 外 deny に意味変化、PLAN §3-2)。"""
    print("== write_file(<tmpdir>/outside.txt) が profile 境界外で deny される ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    outside = Path(tempfile.gettempdir()) / "noetic_outside_target.txt"
    r = guard("write_file", {"path": str(outside), "content": "# outside"})
    return all([
        _assert(r.denied, "denied=True"),
        _assert(any("プロファイル境界外" in m for m in r.messages),
                "profile 境界外 message"),
    ])


def test_write_inside_sandbox_allowed():
    print("== write_file(sandbox/memo.md) は allow される ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("write_file", {"path": "sandbox/memo.md", "content": "x"})
    return _assert(not r.denied, "denied=False")


def test_write_inside_profile_outside_sandbox_allowed():
    """段階12 Step 2: profile 内なら sandbox 外でも write 許可 (身体改変経路)。
    旧 sandbox 限定では deny されてた main.py / core/* が allow になる。"""
    print("== write_file(main.py) が profile 内なので allow される (段階12 緩和) ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("write_file", {"path": "main.py", "content": "# mod"})
    return _assert(not r.denied, "denied=False (profile 内 sandbox 外、身体改変経路)")


def test_write_in_venv_allowed():
    """段階12 Step 1.5 + Step 2: profile 配下の .venv も身体拡張範囲として allow。"""
    print("== write_file(.venv/lib/foo.py) が profile 内なので allow される (身体拡張) ==")
    root = _setup_workspace()
    (root / ".venv" / "lib").mkdir(parents=True, exist_ok=True)
    guard = make_file_access_guard(root)
    r = guard("write_file", {"path": ".venv/lib/foo.py", "content": "# pkg"})
    return _assert(not r.denied, "denied=False (.venv 配下、身体拡張経路)")


def test_write_via_dotdot_outside_profile_denied():
    """段階12 Step 2: `..` で profile 外を指すパスも resolve 後判定で deny。"""
    print("== write_file(../escape.txt) は resolve 後 profile 外で deny ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("write_file", {"path": "../escape.txt", "content": "x"})
    return all([
        _assert(r.denied, "denied=True"),
        _assert(any("プロファイル境界外" in m for m in r.messages),
                "境界外 message"),
    ])


def test_write_via_symbolic_link_outside_profile_denied():
    """段階12 Step 2: profile 内に作った symbolic link が profile 外を指すケースも
    resolve canonical 化後 deny。Windows では Developer Mode 必要、権限なしなら skip。"""
    print("== write_file(escape_link/foo.txt) は symbolic link 解決後 profile 外で deny ==")
    root = _setup_workspace()
    outside_dir = Path(tempfile.mkdtemp(prefix="noetic_outside_for_link_"))
    link_path = root / "escape_link"
    try:
        link_path.symlink_to(outside_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Windows で symlink 権限なし環境では skip 扱い (Developer Mode 必要)
        print("  [SKIP] symbolic link 作成権限がない環境")
        return True
    guard = make_file_access_guard(root)
    r = guard("write_file", {"path": "escape_link/foo.txt", "content": "x"})
    return all([
        _assert(r.denied, "denied=True"),
        _assert(any("プロファイル境界外" in m for m in r.messages),
                "境界外 message"),
    ])


def test_read_inside_sandbox_allowed():
    print("== read_file(sandbox/memo.md) は allow される ==")
    root = _setup_workspace()
    guard = make_file_access_guard(root)
    r = guard("read_file", {"path": "sandbox/memo.md"})
    return _assert(not r.denied, "denied=False")


def test_read_main_py_allowed():
    print("== read_file(main.py) は read は許可 (読取は元々 sandbox 外でも OK) ==")
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
    return _assert(not r.denied, "denied=False (bash は対象外、Step 7.5 で別途強化予定)")


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
        ("write: profile 境界外拒否 (段階12 拡張)", test_write_outside_profile_denied),
        ("write: sandbox 内許可", test_write_inside_sandbox_allowed),
        ("write: profile 内 sandbox 外許可 (段階12 緩和、身体改変)", test_write_inside_profile_outside_sandbox_allowed),
        ("write: .venv 配下許可 (段階12 身体拡張)", test_write_in_venv_allowed),
        ("write: .. 抜け拒否 (段階12 canonical 判定)", test_write_via_dotdot_outside_profile_denied),
        ("write: symbolic link 抜け拒否 (段階12 canonical 判定)", test_write_via_symbolic_link_outside_profile_denied),
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
