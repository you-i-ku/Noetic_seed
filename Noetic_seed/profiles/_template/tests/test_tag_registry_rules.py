"""test_tag_registry_rules.py — 段階11-B Phase 1 Step 1.6。

検証対象:
  - get_tags_with_rule("c_gradual_source") が entity のみ返す (初期状態)
  - get_tags_with_rule("beta_plus") が wm / opinion / entity を返す (既存 rule でも動く)
  - get_tags_with_rule("nonexistent") が [] を返す
  - 空 registry (reset 後) で [] を返す (Phase 5 白紙 onboarding 想定)
  - dynamic tag が c_gradual_source=True で register されたら list に入る

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_tag_registry_rules.py
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
    reg_file = tmpdir / "registered_tags.json"
    if reg_file.exists():
        reg_file.unlink()
    tr._reset_for_testing(registry_file=reg_file)
    return reg_file


def test_c_gradual_source_on_entity_only(tmpdir: Path):
    print("== c_gradual_source を持つのは entity のみ (初期状態) ==")
    _fresh(tmpdir)
    tr.register_standard_tags()
    tags = tr.get_tags_with_rule("c_gradual_source")
    return all([
        _assert(tags == ["entity"], f"entity のみ返る (got {tags})"),
        _assert("wm" not in tags, "wm は含まれない"),
        _assert("opinion" not in tags, "opinion は含まれない"),
        _assert("experience" not in tags, "experience は含まれない"),
    ])


def test_beta_plus_rule_returns_three_tags(tmpdir: Path):
    print("== beta_plus を持つのは wm / opinion / entity ==")
    _fresh(tmpdir)
    tr.register_standard_tags()
    tags = set(tr.get_tags_with_rule("beta_plus"))
    return all([
        _assert(tags == {"wm", "opinion", "entity"}, f"{tags} == wm/opinion/entity"),
        _assert("experience" not in tags, "experience は含まれない (beta_plus=False)"),
    ])


def test_nonexistent_rule_returns_empty(tmpdir: Path):
    print("== 未知 rule_name → [] ==")
    _fresh(tmpdir)
    tr.register_standard_tags()
    return _assert(tr.get_tags_with_rule("nonexistent") == [], "空 list")


def test_empty_registry_returns_empty(tmpdir: Path):
    print("== 空 registry (Phase 5 白紙想定) → [] ==")
    _fresh(tmpdir)
    # register_standard_tags() は呼ばない
    return all([
        _assert(tr.get_tags_with_rule("c_gradual_source") == [], "c_gradual_source で []"),
        _assert(tr.get_tags_with_rule("beta_plus") == [], "beta_plus でも []"),
    ])


def test_dynamic_tag_with_c_gradual_source(tmpdir: Path):
    print("== dynamic tag が c_gradual_source=True で register → list に入る ==")
    _fresh(tmpdir)
    tr.register_standard_tags()
    tr.register_tag(
        "concept",
        learning_rules={"beta_plus": True, "bitemporal": True, "c_gradual_source": True},
        origin="dynamic",
    )
    tags = set(tr.get_tags_with_rule("c_gradual_source"))
    return all([
        _assert(tags == {"entity", "concept"}, f"entity + concept (got {tags})"),
        _assert(tr.get_tag_rules("concept")["learning_rules"]["c_gradual_source"] is True,
                "concept rules に c_gradual_source=True"),
    ])


def test_dynamic_tag_without_c_gradual_source(tmpdir: Path):
    print("== dynamic tag が c_gradual_source 省略 → 既定 False で list 外 ==")
    _fresh(tmpdir)
    tr.register_standard_tags()
    tr.register_tag(
        "memo",
        learning_rules={"beta_plus": False, "bitemporal": False},  # c_gradual_source 省略
        origin="dynamic",
    )
    tags = tr.get_tags_with_rule("c_gradual_source")
    return all([
        _assert("memo" not in tags, "memo は list に含まれない (default False)"),
        _assert(tr.get_tag_rules("memo")["learning_rules"]["c_gradual_source"] is False,
                "memo rules に c_gradual_source=False が明示格納"),
    ])


def run_all():
    print("=" * 60)
    print("test_tag_registry_rules.py (段階11-B Phase 1 Step 1.6)")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        results = [
            test_c_gradual_source_on_entity_only(tmpdir),
            test_beta_plus_rule_returns_three_tags(tmpdir),
            test_nonexistent_rule_returns_empty(tmpdir),
            test_empty_registry_returns_empty(tmpdir),
            test_dynamic_tag_with_c_gradual_source(tmpdir),
            test_dynamic_tag_without_c_gradual_source(tmpdir),
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
