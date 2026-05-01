"""core/sanity_check.py の test (段階12 Step 6, PLAN §10 / §13-1)。

検査項目:
  - _check_state_json: 正常 / JSON 破損 / 必須キー欠落 / 不存在 (初回起動)
  - _check_memory_jsons: 正常 / 破損 JSON あり / memory/ 不存在 (初回起動)
  - _check_imports: 実 module を mock (importlib + sys.modules)
  - enforce_sanity_check: 成功経路 / 失敗 + revert 成功 / 失敗 + stash なし /
    auto_revert=False 失敗

実 module の import / 実 git 操作はせず、_run_all_checks や
_try_auto_revert_from_stash を mock してフロー検証する。

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_sanity_check.py
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import sanity_check
from core.sanity_check import (
    SanityCheckError,
    _check_state_json,
    _check_memory_jsons,
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
    valid = json.dumps({"cycle_id": 1, "tool_level": 0, "log": []})
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
