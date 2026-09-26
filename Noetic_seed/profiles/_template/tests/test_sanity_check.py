"""core/sanity_check.py の test (段階12 Step 6, PLAN §10 / §13-1)。

検査項目:
  - _check_state_json: 正常 / JSON 破損 / 必須キー欠落 / 不存在 (初回起動)
  - _check_memory_jsons: 正常 / 破損 JSON あり / memory/ 不存在 (初回起動)
  - _check_imports: 別プロセスの実 import、構文・実行時例外、起動失敗・timeout
  - _enumerate_runtime_modules: 実 walk + package broken 検出 (Issue 5 fix)
  - enforce_sanity_check: 成功経路 / 失敗 + revert 成功 / 失敗 + stash なし /
    auto_revert=False 失敗

一時 profile で検査し、実 git 操作はしない。
_try_auto_revert_from_stash を mock して復元後の再検査も検証する。

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_sanity_check.py
"""
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import sanity_check
from core.sanity_check import (
    SanityCheckError,
    _check_state_json,
    _check_memory_jsons,
    _check_imports,
    _enumerate_runtime_modules,
    enforce_sanity_check,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup_profile(state_content=None, memory_jsons=None):
    """tmp profile workspace を作る。state_content / memory_jsons を任意配置。"""
    tmp = Path(tempfile.mkdtemp(prefix="noetic_sanity_test_"))
    if state_content is not None:
        (tmp / "state.json").write_text(state_content, encoding="utf-8")
    if memory_jsons:
        (tmp / "memory").mkdir()
        for name, content in memory_jsons.items():
            (tmp / "memory" / name).write_text(content, encoding="utf-8")
    return tmp


def test_state_json_valid():
    print("== state.json 正常 ==")
    valid = json.dumps({"cycle_id": 1, "tool_level": 0, "subjective_entries": []})
    tmp = _setup_profile(state_content=valid)
    try:
        _check_state_json(tmp)
        return _assert(True, "正常 state.json で raise しない")
    except SanityCheckError as e:
        return _assert(False, f"raise 想定外: {e}")


def test_state_json_broken():
    print("== state.json 壊れた JSON ==")
    tmp = _setup_profile(state_content="{ invalid json }")
    try:
        _check_state_json(tmp)
        return _assert(False, "raise されるべき")
    except SanityCheckError as e:
        return _assert("JSON parse 失敗" in str(e),
                       f"reason に 'JSON parse 失敗' 含む (実測: {e})")


def test_state_json_missing_keys():
    print("== state.json 必須キー欠落 ==")
    incomplete = json.dumps({"cycle_id": 1})  # tool_level / log なし
    tmp = _setup_profile(state_content=incomplete)
    try:
        _check_state_json(tmp)
        return _assert(False, "raise されるべき")
    except SanityCheckError as e:
        return _assert("必須キー欠落" in str(e), f"reason に '必須キー欠落' 含む")


def test_state_json_absent_initial_boot():
    print("== state.json 不存在 (初回起動相当) → raise しない ==")
    tmp = _setup_profile()  # state.json 配置なし
    try:
        _check_state_json(tmp)
        return _assert(True, "初回起動として OK 扱い")
    except SanityCheckError as e:
        return _assert(False, f"raise 想定外: {e}")


def test_memory_jsons_all_valid():
    print("== memory/ 配下の JSON 全て正常 ==")
    tmp = _setup_profile(memory_jsons={
        "experience.json": "{}",
        "opinion.json": '{"x": 1}',
    })
    try:
        _check_memory_jsons(tmp)
        return _assert(True, "raise しない")
    except SanityCheckError as e:
        return _assert(False, f"raise 想定外: {e}")


def test_memory_jsons_broken():
    print("== memory/ 配下に壊れた JSON あり ==")
    tmp = _setup_profile(memory_jsons={
        "experience.json": "{}",
        "opinion.json": "{ broken",
    })
    try:
        _check_memory_jsons(tmp)
        return _assert(False, "raise されるべき")
    except SanityCheckError as e:
        return _assert("opinion.json" in str(e),
                       f"reason に 'opinion.json' 含む")


def test_memory_dir_absent():
    print("== memory/ 不存在 (初回起動相当) → raise しない ==")
    tmp = _setup_profile()
    try:
        _check_memory_jsons(tmp)
        return _assert(True, "初回起動として OK 扱い")
    except SanityCheckError as e:
        return _assert(False, f"raise 想定外: {e}")


def test_enumerate_runtime_modules_lists_known_files():
    """Issue 5 fix (2026-05-02): _enumerate_runtime_modules が
    core/runtime/ 配下の既知 module を含む list を返すか (実 walk)。
    """
    print("== _enumerate_runtime_modules: core.runtime 配下を実列挙 ==")
    modules = _enumerate_runtime_modules()
    expected_present = (
        "core.runtime.hooks",
        "core.runtime.permissions",
        "core.runtime.config",
    )
    return all([
        _assert(isinstance(modules, list), "list 型を返す"),
        _assert(
            all(m in modules for m in expected_present),
            f"既知の runtime module を含む (期待: {expected_present}, "
            f"実測 {len(modules)} 件)",
        ),
        _assert(
            all(m.startswith("core.runtime.") for m in modules),
            "全 module が 'core.runtime.' prefix",
        ),
    ])


def test_enumerate_runtime_pkg_import_failure_raises():
    """core.runtime package 自体の import 失敗 → SanityCheckError raise。"""
    print("== _enumerate_runtime_modules: package import 失敗で raise ==")

    def fake_import(name):
        if name == "core.runtime":
            raise ImportError("simulate package broken")
        return importlib.import_module(name)

    with patch.object(sanity_check.importlib, "import_module",
                      side_effect=fake_import):
        try:
            _enumerate_runtime_modules()
            return _assert(False, "raise されるべき")
        except SanityCheckError as e:
            return _assert(
                "core.runtime package" in str(e),
                f"reason に 'core.runtime package' 含む (実測: {e})",
            )


def test_check_imports_runtime_module_failure_detected():
    """Issue 5 fix の核心動作: core.runtime.* の import 失敗を検出。

    旧版は hardcoded "core.controller" / "tools" のみ検査で、
    core/runtime/util.py 等の SyntaxError を素通りさせていた (smoke12
    シナリオ C で発覚)。Fix 後は pkgutil.iter_modules で core/runtime/
    配下を walk して検出する。
    """
    print("== _check_imports: core.runtime.* の import 失敗を検出 ==")

    result = subprocess.CompletedProcess([], 1, "", "SyntaxError: core.runtime.fake_broken")
    with patch.object(sanity_check.subprocess, "run", return_value=result):
        try:
            _check_imports()
            return _assert(False, "raise されるべき")
        except SanityCheckError as e:
            return _assert(
                "core.runtime.fake_broken" in str(e),
                f"reason に 'core.runtime.fake_broken' 含む (実測: {e})",
            )


@pytest.mark.parametrize("source,diagnostic", [
    ("def invalid(:\n", "SyntaxError"),
    ("raise RuntimeError('import-time failure')\n", "import-time failure"),
])
def test_fresh_child_detects_disk_errors_and_revert_rechecks(tmp_path, monkeypatch, source, diagnostic):
    # A miniature profile runs the real subprocess path, without external APIs.
    for relative in ("core/__init__.py", "core/controller.py", "core/runtime/__init__.py", "tools/__init__.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    target = tmp_path / "core/runtime/broken.py"
    target.write_text(source, encoding="utf-8")
    checker = tmp_path / "core/sanity_check.py"
    checker.write_text(Path(sanity_check.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(sanity_check, "__file__", str(checker))
    with pytest.raises(SanityCheckError, match=diagnostic):
        sanity_check._check_imports()
    def revert(*args):
        target.write_text("answer = 42\n", encoding="utf-8")
        return True
    with patch.object(sanity_check, "_try_auto_revert_from_stash", side_effect=revert) as restore:
        sanity_check.enforce_sanity_check(tmp_path, "test")
        restore.assert_called_once()
    assert not list(tmp_path.rglob("*.pyc"))


@pytest.mark.parametrize("error", [OSError("cannot launch"), subprocess.TimeoutExpired("python", 60)])
def test_import_probe_launch_failure_and_timeout(error):
    with patch.object(sanity_check.subprocess, "run", side_effect=error):
        with pytest.raises(SanityCheckError, match=type(error).__name__):
            sanity_check._check_imports()


def test_import_probe_uses_profile_interpreter_and_no_parent_reload():
    result = subprocess.CompletedProcess([], 0, "", "")
    with patch.object(sanity_check.subprocess, "run", return_value=result) as run, \
         patch.object(sanity_check.importlib, "reload", side_effect=AssertionError("live reload")):
        sanity_check._check_imports()
    args, kwargs = run.call_args
    assert args[0][:3] == [sys.executable, "-B", "-c"]
    assert "_enumerate_runtime_modules" in args[0][3]
    assert kwargs["cwd"] == Path(sanity_check.__file__).resolve().parents[1]
    assert kwargs["timeout"] == 60


def test_enforce_success_returns_none():
    print("== enforce_sanity_check 全 OK で None 返り (起動続行) ==")
    tmp = Path(tempfile.mkdtemp(prefix="noetic_sanity_enforce_"))
    with patch.object(sanity_check, "_run_all_checks") as mock_check:
        mock_check.return_value = None  # success
        result = enforce_sanity_check(tmp, "testprof")
    return _assert(result is None, "None 返り (起動続行)")


def test_enforce_failure_with_revert_success():
    """初回 check 失敗 → revert 成功 → 再 check 成功 → 起動続行。"""
    print("== sanity 失敗 → revert 成功 → 再 check 成功 → 続行 ==")
    tmp = Path(tempfile.mkdtemp(prefix="noetic_sanity_revert_"))
    call_count = {"n": 0}

    def fake_run(_):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise SanityCheckError("初回失敗 (simulate)")
        return None  # 2 回目は成功

    with patch.object(sanity_check, "_run_all_checks", side_effect=fake_run), \
         patch.object(sanity_check, "_try_auto_revert_from_stash",
                      return_value=True):
        result = enforce_sanity_check(tmp, "testprof")
    return all([
        _assert(result is None, "起動続行"),
        _assert(call_count["n"] == 2, "_run_all_checks が 2 回呼ばれた"),
    ])


def test_enforce_failure_no_stash_exits():
    print("== sanity 失敗 → stash なし → sys.exit(1) ==")
    tmp = Path(tempfile.mkdtemp(prefix="noetic_sanity_nostash_"))
    with patch.object(sanity_check, "_run_all_checks",
                      side_effect=SanityCheckError("失敗 (simulate)")), \
         patch.object(sanity_check, "_try_auto_revert_from_stash",
                      return_value=False):
        try:
            enforce_sanity_check(tmp, "testprof")
            return _assert(False, "sys.exit 想定")
        except SystemExit as e:
            return _assert(e.code == 1, f"exit code = 1 (実測: {e.code})")


def test_enforce_failure_auto_revert_disabled_exits():
    print("== auto_revert=False で失敗時 sys.exit(1) (revert 試行なし) ==")
    tmp = Path(tempfile.mkdtemp(prefix="noetic_sanity_norevert_"))
    with patch.object(sanity_check, "_run_all_checks",
                      side_effect=SanityCheckError("失敗 (simulate)")), \
         patch.object(sanity_check,
                      "_try_auto_revert_from_stash") as mock_revert:
        try:
            enforce_sanity_check(tmp, "testprof", auto_revert=False)
            return _assert(False, "sys.exit 想定")
        except SystemExit as e:
            return all([
                _assert(e.code == 1, "exit code = 1"),
                _assert(not mock_revert.called,
                        "auto_revert=False なら _try_auto_revert 呼ばれない"),
            ])


if __name__ == "__main__":
    groups = [
        ("state.json 正常", test_state_json_valid),
        ("state.json 壊れた JSON", test_state_json_broken),
        ("state.json 必須キー欠落", test_state_json_missing_keys),
        ("state.json 不存在 (初回起動)", test_state_json_absent_initial_boot),
        ("memory/ 全 JSON 正常", test_memory_jsons_all_valid),
        ("memory/ 壊れた JSON あり", test_memory_jsons_broken),
        ("memory/ 不存在 (初回起動)", test_memory_dir_absent),
        ("Issue 5 fix: _enumerate_runtime_modules 実列挙",
         test_enumerate_runtime_modules_lists_known_files),
        ("Issue 5 fix: core.runtime package import 失敗で raise",
         test_enumerate_runtime_pkg_import_failure_raises),
        ("Issue 5 fix: core.runtime.* import 失敗を検出",
         test_check_imports_runtime_module_failure_detected),
        ("enforce 全 OK で続行", test_enforce_success_returns_none),
        ("enforce 失敗 + revert 成功 + 再 check 成功 で続行",
         test_enforce_failure_with_revert_success),
        ("enforce 失敗 + stash なし で exit",
         test_enforce_failure_no_stash_exits),
        ("auto_revert=False で失敗時即 exit",
         test_enforce_failure_auto_revert_disabled_exits),
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
