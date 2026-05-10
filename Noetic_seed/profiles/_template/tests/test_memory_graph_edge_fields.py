"""memory_graph edge field expansion テスト (v0.5 Phase 5 Slice 4 F-003).

Slice 4 F-003: memory_graph view 層で Physarum 4 field を additive expose
(raw strength + lazy-decayed strength + usage_count + last_used_cycle)。
graph material 不変、view layer のみ拡張。

成功条件:
  - edge dict に strength / decayed_strength / usage_count / last_used_cycle
    の 4 field 全部含まれる + 既存 confidence は保持
  - current_cycle 進むと decayed_strength < raw_strength (lazy decay 効く)
  - current_cycle=None で decayed_strength == raw_strength (decay skip)
  - Phase 2 以前 link (strength 欠落) は confidence fallback で raw 取得
  - link_type='none' は edge から除外 (既存挙動保持)

識別力 (CLAUDE.md §5):
  - decay 計算配線漏れの誤実装で test_a2 が fail (decayed == raw のまま)
  - field 追加漏れの誤実装で test_a1 が fail
  - confidence 保持漏れで test_a5 が fail (backward compat 違反)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_memory_graph_edge_fields.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.memory as memory_mod
import core.memory_links as ml_mod
import core.tag_registry as tr
import tools.memory_graph_tool as mg


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup(tmp_path: Path):
    from core import config as cfg
    cfg.MEMORY_DIR = tmp_path
    memory_mod.MEMORY_DIR = tmp_path
    ml_mod.MEMORY_DIR = tmp_path
    reg_file = tmp_path / "registered_tags.json"
    tr._reset_for_testing(registry_file=reg_file)
    tr.register_standard_tags()


def _write_link(tmp_path: Path, link: dict):
    fpath = tmp_path / "memory_links.jsonl"
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(link, ensure_ascii=False) + "\n")


# ============================================================
# Section A: edge field expansion
# ============================================================

def test_a1_edge_has_all_4_fields(tmp_path: Path):
    print("== edge dict に Physarum 4 field 全部 + confidence 保持 ==")
    _setup(tmp_path)
    _write_link(tmp_path, {
        "id": "link_x",
        "from_id": "a", "to_id": "b",
        "link_type": "similar",
        "confidence": 0.7,
        "strength": 0.8,
        "last_used_cycle": 100,
        "usage_count": 3,
    })
    edges = mg._compute_memory_edges(current_cycle=100)
    if not edges:
        return _assert(False, "edge 1 件期待")
    e = edges[0]
    return all([
        _assert("strength" in e, "strength field 存在"),
        _assert("decayed_strength" in e, "decayed_strength field 存在"),
        _assert("usage_count" in e, "usage_count field 存在"),
        _assert("last_used_cycle" in e, "last_used_cycle field 存在"),
        _assert("confidence" in e, "既存 confidence 保持 (backward compat)"),
    ])


def test_a2_decayed_strength_with_cycle_advance(tmp_path: Path):
    print("== current_cycle=200 (100 cycle 経過) → decayed_strength < raw ==")
    _setup(tmp_path)
    _write_link(tmp_path, {
        "id": "link_x",
        "from_id": "a", "to_id": "b",
        "link_type": "similar",
        "confidence": 0.7,
        "strength": 1.0,
        "last_used_cycle": 100,
        "usage_count": 3,
    })
    edges = mg._compute_memory_edges(current_cycle=200)
    e = edges[0]
    return all([
        _assert(e["strength"] == 1.0,
                f"raw strength=1.0 不変 (got: {e['strength']})"),
        _assert(e["decayed_strength"] < 1.0,
                f"decayed_strength < raw (got: {e['decayed_strength']})"),
        _assert(e["decayed_strength"] > 0.0,
                f"decayed_strength > 0 (got: {e['decayed_strength']})"),
    ])


def test_a3_no_cycle_decayed_equals_raw(tmp_path: Path):
    print("== current_cycle=None → decayed_strength == raw (decay skip) ==")
    _setup(tmp_path)
    _write_link(tmp_path, {
        "id": "link_x",
        "from_id": "a", "to_id": "b",
        "link_type": "similar",
        "confidence": 0.7,
        "strength": 0.8,
        "last_used_cycle": 100,
        "usage_count": 3,
    })
    edges = mg._compute_memory_edges(current_cycle=None)
    e = edges[0]
    return _assert(
        abs(e["decayed_strength"] - e["strength"]) < 1e-9,
        f"decayed == raw (decayed={e['decayed_strength']}, raw={e['strength']})"
    )


def test_a4_phase2_legacy_link_fallback(tmp_path: Path):
    print("== Phase 2 以前 link (strength 欠落) → strength = confidence fallback ==")
    _setup(tmp_path)
    _write_link(tmp_path, {
        "id": "link_legacy",
        "from_id": "a", "to_id": "b",
        "link_type": "similar",
        "confidence": 0.65,
        # strength / last_used_cycle / usage_count 全部欠落 (Phase 2 以前)
    })
    edges = mg._compute_memory_edges(current_cycle=100)
    e = edges[0]
    return all([
        _assert(abs(e["strength"] - 0.65) < 1e-9,
                f"strength = confidence fallback 0.65 (got: {e['strength']})"),
        _assert(abs(e["decayed_strength"] - 0.65) < 1e-9,
                f"decayed_strength も confidence fallback (got: {e['decayed_strength']})"),
        _assert(e["usage_count"] == 0,
                f"usage_count default 0 (got: {e['usage_count']})"),
        _assert(e["last_used_cycle"] is None,
                f"last_used_cycle None (got: {e['last_used_cycle']!r})"),
    ])


def test_a5_existing_confidence_preserved(tmp_path: Path):
    print("== 既存 confidence は保持 (backward compat) ==")
    _setup(tmp_path)
    _write_link(tmp_path, {
        "id": "link_x",
        "from_id": "a", "to_id": "b",
        "link_type": "elaborate",
        "confidence": 0.92,
        "strength": 1.0,
        "last_used_cycle": 100,
        "usage_count": 5,
    })
    edges = mg._compute_memory_edges(current_cycle=100)
    e = edges[0]
    return _assert(abs(e["confidence"] - 0.92) < 1e-9,
                   f"confidence=0.92 保持 (got: {e['confidence']})")


def test_a6_relation_none_excluded(tmp_path: Path):
    print("== link_type='none' は edge から除外 (既存挙動保持) ==")
    _setup(tmp_path)
    _write_link(tmp_path, {
        "id": "l1", "from_id": "a", "to_id": "b",
        "link_type": "none", "confidence": 0.5,
    })
    _write_link(tmp_path, {
        "id": "l2", "from_id": "c", "to_id": "d",
        "link_type": "similar", "confidence": 0.8,
    })
    edges = mg._compute_memory_edges(current_cycle=100)
    relations = [e["relation"] for e in edges]
    return all([
        _assert(len(edges) == 1, f"none 除外 (got: {len(edges)} edges)"),
        _assert(relations == ["similar"],
                f"残り relation=['similar'] (got: {relations})"),
    ])


def test_a7_get_link_current_strength_public_api(tmp_path: Path):
    print("== get_link_current_strength (公開 API) 直接 call ==")
    _setup(tmp_path)
    link = {
        "strength": 1.0,
        "confidence": 0.7,
        "last_used_cycle": 100,
    }
    s_raw = ml_mod.get_link_current_strength(link, current_cycle=None)
    s_decayed = ml_mod.get_link_current_strength(link, current_cycle=200)
    return all([
        _assert(s_raw == 1.0, f"current_cycle=None で raw 1.0 (got: {s_raw})"),
        _assert(s_decayed < 1.0,
                f"current_cycle=200 で decayed < 1.0 (got: {s_decayed})"),
    ])


def test_a8_compute_memory_edges_uses_public_wrapper(tmp_path: Path):
    """P2-3 boundary identification: _compute_memory_edges は公開 wrapper
    (get_link_current_strength) 経由で decay 計算する。
    private 関数 (_link_strength / _apply_lazy_decay) を直接呼ぶ bypass 実装で
    識別力ある fail を出すため、spy/sentinel monkeypatch で wrapper 呼出を assert
    (Codex Slice 4 review P2-3)。"""
    print("== _compute_memory_edges は get_link_current_strength wrapper を呼ぶ ==")
    _setup(tmp_path)
    _write_link(tmp_path, {
        "id": "link_x",
        "from_id": "a", "to_id": "b",
        "link_type": "similar",
        "confidence": 0.7,
        "strength": 0.8,
        "last_used_cycle": 100,
        "usage_count": 3,
    })
    sentinel = 0.4242
    call_count = {"n": 0}
    orig = mg.get_link_current_strength

    def spy_wrapper(link, current_cycle=None):
        call_count["n"] += 1
        return sentinel
    mg.get_link_current_strength = spy_wrapper
    try:
        edges = mg._compute_memory_edges(current_cycle=200)
    finally:
        mg.get_link_current_strength = orig
    if not edges:
        return _assert(False, "edge 1 件期待")
    e = edges[0]
    return all([
        _assert(call_count["n"] >= 1,
                f"wrapper 呼ばれる (got: {call_count['n']} 回)"),
        _assert(abs(e["decayed_strength"] - sentinel) < 1e-9,
                f"decayed_strength = sentinel {sentinel} (bypass なら raw 0.8、got: {e['decayed_strength']})"),
    ])


def run_all():
    print("=" * 60)
    print("test_memory_graph_edge_fields.py (v0.5 Phase 5 Slice 4 F-003)")
    print("=" * 60)
    results = []
    with tempfile.TemporaryDirectory() as td:
        for fn in [
            test_a1_edge_has_all_4_fields,
            test_a2_decayed_strength_with_cycle_advance,
            test_a3_no_cycle_decayed_equals_raw,
            test_a4_phase2_legacy_link_fallback,
            test_a5_existing_confidence_preserved,
            test_a6_relation_none_excluded,
            test_a7_get_link_current_strength_public_api,
            test_a8_compute_memory_edges_uses_public_wrapper,
        ]:
            sub = Path(td) / fn.__name__
            sub.mkdir(exist_ok=True)
            results.append(fn(sub))
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"結果: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
