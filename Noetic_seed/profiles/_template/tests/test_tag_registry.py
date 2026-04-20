"""tag_registry.py テスト (段階7 Step 1)。

成功条件:
  - 標準 4 タグが register_standard_tags() で登録される
  - idempotent (同じタグを standard で再登録しても created_at 保持)
  - dynamic 既存への再登録は ValueError
  - 不正入力 (空 name / 非 dict rules) reject
  - 永続化 → 再 load で復元
  - list / get / is が正しく動作
  - 既定 display_format 適用

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_tag_registry.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.tag_registry as tr


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _fresh(tmpdir: Path) -> Path:
    """各テスト前に in-memory + registry file をリセット。"""
    reg_file = tmpdir / "registered_tags.json"
    if reg_file.exists():
        reg_file.unlink()
    tr._reset_for_testing(registry_file=reg_file)
    return reg_file


def test_register_standard_tags(tmpdir: Path):
    print("== register_standard_tags: 標準 4 タグ登録 ==")
    _fresh(tmpdir)
    tr.register_standard_tags()
    tags = tr.list_registered_tags()
    return all([
        _assert(set(tags) == {"wm", "experience", "opinion", "entity"}, "標準 4 タグ存在"),
        _assert(tr.is_tag_registered("wm"), "is_tag_registered('wm')"),
        _assert(tr.get_tag_rules("wm")["learning_rules"]["beta_plus"] is True, "wm beta_plus=True"),
        _assert(tr.get_tag_rules("wm")["learning_rules"]["bitemporal"] is True, "wm bitemporal=True"),
        _assert(tr.get_tag_rules("experience")["learning_rules"]["beta_plus"] is False, "experience beta_plus=False"),
        _assert(tr.get_tag_rules("opinion")["learning_rules"]["bitemporal"] is False, "opinion bitemporal=False"),
        _assert(tr.get_tag_rules("wm")["origin"] == "standard", "wm origin=standard"),
        _assert(tr.get_tag_rules("opinion")["display_format"] == "[opinion] {content} (確度:{confidence})", "opinion display_format 反映"),
    ])


def test_register_dynamic_tag(tmpdir: Path):
    print("== register_tag: 動的タグ新規登録 ==")
    _fresh(tmpdir)
    entry = tr.register_tag(
        "dream",
        learning_rules={"beta_plus": False, "bitemporal": False},
        display_format="[dream] {content}",
        origin="dynamic",
        intent="想像した情景を残したい",
    )
    return all([
        _assert(tr.is_tag_registered("dream"), "dream 登録済"),
        _assert(entry["origin"] == "dynamic", "origin=dynamic"),
        _assert(entry["intent"] == "想像した情景を残したい", "intent 保存"),
        _assert(entry["display_format"] == "[dream] {content}", "display_format 保存"),
        _assert(entry["learning_rules"]["beta_plus"] is False, "beta_plus=False"),
    ])


def test_dynamic_reregister_rejects(tmpdir: Path):
    print("== register_tag: dynamic 既存への再登録は ValueError ==")
    _fresh(tmpdir)
    tr.register_tag("dream", learning_rules={"beta_plus": False, "bitemporal": False})
    try:
        tr.register_tag("dream", learning_rules={"beta_plus": True, "bitemporal": True})
        return _assert(False, "ValueError 期待")
    except ValueError:
        return _assert(True, "ValueError 発生")


def test_standard_idempotent(tmpdir: Path):
    print("== register_standard_tags: idempotent (created_at 保持) ==")
    _fresh(tmpdir)
    tr.register_standard_tags()
    first_created = tr.get_tag_rules("wm")["created_at"]
    tr.register_standard_tags()
    second_created = tr.get_tag_rules("wm")["created_at"]
    return _assert(first_created == second_created, "created_at 保持 (上書きなし)")


def test_invalid_inputs(tmpdir: Path):
    print("== register_tag: 不正入力 reject ==")
    _fresh(tmpdir)
    results = []
    try:
        tr.register_tag("", learning_rules={"beta_plus": False, "bitemporal": False})
        results.append(_assert(False, "空 name で ValueError 期待"))
    except ValueError:
        results.append(_assert(True, "空 name reject"))
    try:
        tr.register_tag("   ", learning_rules={"beta_plus": False, "bitemporal": False})
        results.append(_assert(False, "空白のみ name で ValueError 期待"))
    except ValueError:
        results.append(_assert(True, "空白のみ name reject"))
    try:
        tr.register_tag("x", learning_rules="invalid")  # type: ignore
        results.append(_assert(False, "非 dict rules で ValueError 期待"))
    except ValueError:
        results.append(_assert(True, "非 dict rules reject"))
    return all(results)


def test_persistence_and_reload(tmpdir: Path):
    print("== 永続化 + 再 load ==")
    reg_file = _fresh(tmpdir)
    tr.register_standard_tags()
    tr.register_tag(
        "dream",
        learning_rules={"beta_plus": False, "bitemporal": False},
        intent="夢",
    )
    results = [_assert(reg_file.exists(), "registry file 存在")]
    tr._reset_for_testing(registry_file=reg_file)
    results.append(_assert(tr.is_tag_registered("wm"), "再 load で wm 復元"))
    results.append(_assert(tr.is_tag_registered("dream"), "再 load で dream 復元"))
    results.append(_assert(tr.get_tag_rules("dream")["intent"] == "夢", "dream intent 復元"))
    results.append(_assert(len(tr.list_registered_tags()) == 5, "total 5 タグ (標準4+dream)"))
    return all(results)


def test_missing_file_graceful(tmpdir: Path):
    print("== 永続化ファイル不在 → graceful empty ==")
    reg_file = tmpdir / "no_such_file.json"
    tr._reset_for_testing(registry_file=reg_file)
    return all([
        _assert(tr.list_registered_tags() == [], "空 list 返却"),
        _assert(tr.get_tag_rules("wm") is None, "未登録は None"),
        _assert(tr.is_tag_registered("wm") is False, "is_tag_registered False"),
    ])


def test_corrupt_file_graceful(tmpdir: Path):
    print("== 永続化ファイル壊れ → graceful skip ==")
    reg_file = tmpdir / "corrupt.json"
    reg_file.write_text("not a valid json", encoding="utf-8")
    tr._reset_for_testing(registry_file=reg_file)
    return _assert(tr.list_registered_tags() == [], "壊れたファイル skip → 空")


def test_default_display_format(tmpdir: Path):
    print("== display_format 省略時: [{name}] {content} 既定 ==")
    _fresh(tmpdir)
    entry = tr.register_tag("goal", learning_rules={"beta_plus": False, "bitemporal": False})
    return _assert(entry["display_format"] == "[goal] {content}", "既定 display_format")


def run_all():
    print("=" * 60)
    print("test_tag_registry.py (段階7 Step 1)")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        results = [
            test_register_standard_tags(tmpdir),
            test_register_dynamic_tag(tmpdir),
            test_dynamic_reregister_rejects(tmpdir),
            test_standard_idempotent(tmpdir),
            test_invalid_inputs(tmpdir),
            test_persistence_and_reload(tmpdir),
            test_missing_file_graceful(tmpdir),
            test_corrupt_file_graceful(tmpdir),
            test_default_display_format(tmpdir),
        ]
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"結果: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
