"""test_tag_emergence_monitor.py — 段階11-B Phase 2 Step 2.6。

検証対象:
  - collect_emergence_stats が期待 field (total / standard / dynamic /
    write_protected) を返す
  - register_standard_tags() 後の初期値 (標準 5 タグ = wm/experience/opinion/
    entity/tag_consideration)
  - 新 dynamic tag register で dynamic_count が増える
  - write_protected=True の dynamic tag で write_protected_count が増える
  - 空 registry (reset 後) で 全 count=0 (Phase 5 白紙 onboarding 想定)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_tag_emergence_monitor.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.tag_registry as tr
from core.tag_emergence_monitor import collect_emergence_stats


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


def test_empty_registry(tmpdir: Path):
    print("== 空 registry → 全 count=0 (Phase 5 白紙想定) ==")
    _fresh(tmpdir)
    stats = collect_emergence_stats()
    return all([
        _assert(stats["total_registered"] == 0, "total_registered=0"),
        _assert(stats["standard_count"] == 0, "standard_count=0"),
        _assert(stats["dynamic_count"] == 0, "dynamic_count=0"),
        _assert(stats["write_protected_count"] == 0, "write_protected_count=0"),
    ])


def test_standard_tags_registered(tmpdir: Path):
    print("== 標準タグ 4 種登録後 (wm/experience/opinion/entity) ==")
    _fresh(tmpdir)
    tr.register_standard_tags()
    stats = collect_emergence_stats()
    return all([
        _assert(stats["total_registered"] == 4, f"total_registered=4 (got {stats['total_registered']})"),
        _assert(stats["standard_count"] == 4, f"standard_count=4 (got {stats['standard_count']})"),
        _assert(stats["dynamic_count"] == 0, f"dynamic_count=0 (got {stats['dynamic_count']})"),
        _assert(
            stats["write_protected_count"] == 0,
            f"write_protected_count=0 (標準 4 タグは全て write_protected=False、got {stats['write_protected_count']})",
        ),
    ])


def test_dynamic_tag_increments_count(tmpdir: Path):
    print("== 新 dynamic tag register → dynamic_count +1 ==")
    _fresh(tmpdir)
    tr.register_standard_tags()
    before = collect_emergence_stats()
    tr.register_tag(
        "concept",
        learning_rules={"beta_plus": True, "bitemporal": True},
        origin="dynamic",
        intent="概念カテゴリ",
    )
    after = collect_emergence_stats()
    return all([
        _assert(
            after["dynamic_count"] == before["dynamic_count"] + 1,
            f"dynamic_count +1 ({before['dynamic_count']} → {after['dynamic_count']})",
        ),
        _assert(
            after["total_registered"] == before["total_registered"] + 1,
            "total_registered +1",
        ),
        _assert(
            after["standard_count"] == before["standard_count"],
            "standard_count 不変",
        ),
        _assert(
            after["write_protected_count"] == before["write_protected_count"],
            "write_protected_count 不変 (concept は write_protected=False)",
        ),
    ])


def test_write_protected_dynamic_tag(tmpdir: Path):
    print("== dynamic write_protected tag → write_protected_count +1 ==")
    _fresh(tmpdir)
    tr.register_standard_tags()
    before = collect_emergence_stats()
    tr.register_tag(
        "my_meta",
        learning_rules={"write_protected": True},
        origin="dynamic",
        intent="meta-section",
    )
    after = collect_emergence_stats()
    return all([
        _assert(
            after["write_protected_count"] == before["write_protected_count"] + 1,
            f"write_protected_count +1 ({before['write_protected_count']} → {after['write_protected_count']})",
        ),
        _assert(
            after["dynamic_count"] == before["dynamic_count"] + 1,
            "dynamic_count も +1 (両 count は併存)",
        ),
    ])


def test_expected_fields(tmpdir: Path):
    print("== 戻り値 dict が期待 field を全て持つ ==")
    _fresh(tmpdir)
    tr.register_standard_tags()
    stats = collect_emergence_stats()
    expected = {"total_registered", "standard_count", "dynamic_count", "write_protected_count"}
    return _assert(
        set(stats.keys()) == expected,
        f"field 集合 == {expected} (got {set(stats.keys())})",
    )


def run_all():
    print("=" * 60)
    print("test_tag_emergence_monitor.py (段階11-B Phase 2 Step 2.6)")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        results = [
            test_empty_registry(tmpdir),
            test_standard_tags_registered(tmpdir),
            test_dynamic_tag_increments_count(tmpdir),
            test_write_protected_dynamic_tag(tmpdir),
            test_expected_fields(tmpdir),
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
