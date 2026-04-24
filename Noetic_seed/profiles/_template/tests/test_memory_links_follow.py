"""test_memory_links_follow.py — 段階11-C G-lite Phase 1.

検証対象: core.memory_links.follow_links

  [A] 境界条件 (空 node_id / depth<=0 / 存在しない id / link 無し)
  [B] depth=1 で直接 link 先を返す
  [C] depth=2 で 2 hop 先を返す (再帰)
  [D] visited で循環回避 (A→B→A 無限ループ防止)
  [E] link_types filter で type 限定
  [F] min_confidence threshold で低品質 link 除外
  [G] top_n_per_depth で各 depth 上位 N 件のみ展開
  [H] 返り値 schema (memory_entry / via_link / depth / strength_hint)
  [I] API 契約: strength_hint = via_link.confidence (G-lite)、G-full で
      strength field 差し替え可能な設計

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_memory_links_follow.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_follow_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

import core.memory_links as _ml
_ml.MEMORY_DIR = _tmp_memory

from core.memory import memory_store
from core.memory_links import (
    LINK_TRAVERSAL_MAX_DEPTH_DEFAULT,
    LINK_TRAVERSAL_TOP_N_DEFAULT,
    _append_link,
    _build_link_entry,
    follow_links,
)
from core.tag_registry import register_standard_tags


register_standard_tags()


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _make_entry(content: str, tag: str = "opinion") -> dict:
    """memory_store 経由で entry 作成 (storage only、LLM 呼出なし)."""
    return memory_store(tag, content, {}, origin="test")


def _make_link(src: dict, dst: dict, link_type: str = "similar",
               confidence: float = 0.8) -> dict:
    """手動 link 作成 (_build_link_entry + _append_link、LLM judge skip)."""
    verdict = {"link_type": link_type, "confidence": confidence, "reason": "test"}
    link = _build_link_entry(src, dst, verdict)
    _append_link(link)
    return link


# ============================================================
# [A] 境界条件
# ============================================================

def test_empty_and_invalid_inputs():
    print("== 境界: 空 node_id / depth<=0 / 存在しない id / link 無し ==")
    e1 = follow_links("", depth=1)
    e2 = follow_links("nonexistent_id_xyz", depth=1)
    entry_a = _make_entry("A_only")
    e3 = follow_links(entry_a["id"], depth=0)
    e4 = follow_links(entry_a["id"], depth=-1)
    e5 = follow_links(entry_a["id"], depth=1)   # link 無し
    return all([
        _assert(e1 == [], "空 node_id で []"),
        _assert(e2 == [], "存在しない id で []"),
        _assert(e3 == [], "depth=0 で []"),
        _assert(e4 == [], "depth<0 で []"),
        _assert(e5 == [], "link 無し entry で []"),
    ])


# ============================================================
# [B] depth=1 直接 link 先
# ============================================================

def test_depth_1_returns_direct_neighbors():
    print("== depth=1: 直接 link 先を返す ==")
    a = _make_entry("A_d1")
    b = _make_entry("B_d1")
    _make_link(a, b, link_type="similar", confidence=0.9)

    reached = follow_links(a["id"], depth=1)
    return all([
        _assert(len(reached) == 1, "1 件の near"),
        _assert(reached[0]["memory_entry"]["id"] == b["id"], "to_id が B"),
        _assert(reached[0]["depth"] == 1, "depth=1 記録"),
        _assert(reached[0]["strength_hint"] == 0.9, "strength_hint = confidence"),
        _assert(reached[0]["via_link"]["from_id"] == a["id"], "via_link.from_id = A"),
    ])


# ============================================================
# [C] depth=2 再帰
# ============================================================

def test_depth_2_traverses_two_hops():
    print("== depth=2: 2 hop 先まで traverse ==")
    a = _make_entry("A_d2")
    b = _make_entry("B_d2")
    c = _make_entry("C_d2")
    _make_link(a, b, confidence=0.9)
    _make_link(b, c, confidence=0.85)

    reached = follow_links(a["id"], depth=2)
    reached_ids = [r["memory_entry"]["id"] for r in reached]
    depths_by_id = {r["memory_entry"]["id"]: r["depth"] for r in reached}
    return all([
        _assert(len(reached) == 2, "2 件 (B + C)"),
        _assert(b["id"] in reached_ids, "B 到達"),
        _assert(c["id"] in reached_ids, "C 到達 (2 hop)"),
        _assert(depths_by_id.get(b["id"]) == 1, "B の depth=1"),
        _assert(depths_by_id.get(c["id"]) == 2, "C の depth=2"),
    ])


# ============================================================
# [D] 循環回避
# ============================================================

def test_cycle_prevention():
    print("== 循環 A→B→A が visited で停止 ==")
    a = _make_entry("A_cyc")
    b = _make_entry("B_cyc")
    _make_link(a, b, confidence=0.9)
    _make_link(b, a, confidence=0.9)   # 逆向き循環

    reached = follow_links(a["id"], depth=5)
    reached_ids = [r["memory_entry"]["id"] for r in reached]
    return all([
        _assert(reached_ids.count(b["id"]) == 1, "B は 1 回だけ"),
        _assert(reached_ids.count(a["id"]) == 0, "A は return されない (起点 visited)"),
    ])


# ============================================================
# [E] link_types filter
# ============================================================

def test_link_types_filter():
    print("== link_types filter ==")
    a = _make_entry("A_f")
    b = _make_entry("B_f")
    c = _make_entry("C_f")
    _make_link(a, b, link_type="similar", confidence=0.9)
    _make_link(a, c, link_type="causal", confidence=0.9)

    all_reached = follow_links(a["id"], depth=1)
    similar_only = follow_links(a["id"], depth=1, link_types=("similar",))
    similar_ids = [r["memory_entry"]["id"] for r in similar_only]
    return all([
        _assert(len(all_reached) == 2, "filter なしで 2 件"),
        _assert(len(similar_only) == 1, "filter similar で 1 件"),
        _assert(similar_ids == [b["id"]], "B のみ返る"),
    ])


# ============================================================
# [F] min_confidence threshold
# ============================================================

def test_min_confidence_threshold():
    print("== min_confidence で低品質 link 除外 ==")
    a = _make_entry("A_c")
    b = _make_entry("B_c")
    c = _make_entry("C_c")
    _make_link(a, b, confidence=0.9)
    _make_link(a, c, confidence=0.6)   # 閾値下

    high_only = follow_links(a["id"], depth=1, min_confidence=0.7)
    all_loose = follow_links(a["id"], depth=1, min_confidence=0.0)
    return all([
        _assert(len(high_only) == 1, "0.7 閾値で 1 件"),
        _assert(high_only[0]["memory_entry"]["id"] == b["id"], "B のみ"),
        _assert(len(all_loose) == 2, "0.0 閾値で 2 件"),
    ])


# ============================================================
# [G] top_n_per_depth
# ============================================================

def test_top_n_per_depth():
    print("== top_n_per_depth で各 depth 上位 N 件のみ展開 ==")
    a = _make_entry("A_n")
    b1 = _make_entry("B1_n")
    b2 = _make_entry("B2_n")
    b3 = _make_entry("B3_n")
    _make_link(a, b1, confidence=0.95)
    _make_link(a, b2, confidence=0.85)
    _make_link(a, b3, confidence=0.75)

    top2 = follow_links(a["id"], depth=1, top_n_per_depth=2)
    top1 = follow_links(a["id"], depth=1, top_n_per_depth=1)
    top2_ids = [r["memory_entry"]["id"] for r in top2]
    return all([
        _assert(len(top2) == 2, "top_n=2 で 2 件"),
        _assert(len(top1) == 1, "top_n=1 で 1 件"),
        _assert(top1[0]["memory_entry"]["id"] == b1["id"],
                "top=1 で最高 conf (B1)"),
        _assert(top2_ids == [b1["id"], b2["id"]],
                "top=2 で confidence 降順 (B1, B2)"),
    ])


# ============================================================
# [H][I] 返り値 schema + API 契約
# ============================================================

def test_return_schema_and_api_contract():
    print("== 返り値 schema + API 契約 (strength_hint) ==")
    a = _make_entry("A_s")
    b = _make_entry("B_s")
    _make_link(a, b, confidence=0.8)

    reached = follow_links(a["id"], depth=1)
    r = reached[0]
    return all([
        _assert("memory_entry" in r, "memory_entry field"),
        _assert("via_link" in r, "via_link field"),
        _assert("depth" in r, "depth field"),
        _assert("strength_hint" in r, "strength_hint field"),
        _assert(isinstance(r["strength_hint"], float), "strength_hint is float"),
        _assert(r["strength_hint"] == float(r["via_link"]["confidence"]),
                "G-lite: strength_hint = via_link.confidence"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    try:
        groups = [
            ("境界条件", test_empty_and_invalid_inputs),
            ("depth=1 直接", test_depth_1_returns_direct_neighbors),
            ("depth=2 再帰", test_depth_2_traverses_two_hops),
            ("循環回避", test_cycle_prevention),
            ("link_types filter", test_link_types_filter),
            ("min_confidence", test_min_confidence_threshold),
            ("top_n_per_depth", test_top_n_per_depth),
            ("schema + API 契約", test_return_schema_and_api_contract),
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
    finally:
        shutil.rmtree(_tmp_root, ignore_errors=True)
