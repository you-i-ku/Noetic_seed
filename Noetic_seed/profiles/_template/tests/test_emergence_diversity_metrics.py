"""test_emergence_diversity_metrics.py — 段階11-C G-lite Phase 2 Step 2.5。

検証対象 (§5 Phase 2 Step 2.1-2.3):
  - _compute_tag_distribution_metrics:
    Shannon H / Pielou E / Gini-Simpson sanity (単一 / 均等 / 偏り)
  - _compute_link_graph_metrics:
    空 graph / 三角形 / 単一 link / degree_distribution 形式
  - _compute_tag_dependency_metrics:
    opinion/entity baseline (全 op/ent で ratio=1.0) / 非 op/ent 混在

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" \\
      tests/test_emergence_diversity_metrics.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tag_emergence_monitor import (
    _compute_tag_distribution_metrics,
    _compute_link_graph_metrics,
    _compute_tag_dependency_metrics,
)


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# =========================================================================
# Section A: Shannon / Pielou / Gini-Simpson sanity
# =========================================================================
print("=== Section A: tag 分布 metric ===")

# A-1: 空 dict → 全 0
m_a1 = _compute_tag_distribution_metrics({})
_assert(m_a1["tag_count"] == 0, "A-1 空 dict で tag_count=0")
_assert(m_a1["total_usage"] == 0, "A-1 空 dict で total_usage=0")
_assert(m_a1["shannon_h"] == 0.0, "A-1 空 dict で shannon_h=0")
_assert(m_a1["pielou_evenness"] == 0.0, "A-1 空 dict で pielou=0")
_assert(m_a1["gini_simpson"] == 0.0, "A-1 空 dict で gini=0")

# A-2: 全件 usage=0 → 空相当
m_a2 = _compute_tag_distribution_metrics({"a": 0, "b": 0, "c": 0})
_assert(m_a2["tag_count"] == 0, "A-2 usage=0 のみ → tag_count=0 (support 集合空)")
_assert(m_a2["total_usage"] == 0, "A-2 usage=0 のみ → total_usage=0")

# A-3: 単一 tag (p=1) → shannon=0, pielou=0 (定義上), gini=0
m_a3 = _compute_tag_distribution_metrics({"a": 10})
_assert(m_a3["tag_count"] == 1, "A-3 単一 tag で tag_count=1")
_assert(m_a3["total_usage"] == 10, "A-3 total_usage=10")
_assert(_close(m_a3["shannon_h"], 0.0), f"A-3 単一 tag shannon=0 (got {m_a3['shannon_h']})")
_assert(_close(m_a3["pielou_evenness"], 0.0), "A-3 単一 tag pielou=0 (均等化空間なし)")
_assert(_close(m_a3["gini_simpson"], 0.0), "A-3 単一 tag gini=0 (2 件 pick で必ず同 tag)")

# A-4: 均等分布 2 tag → shannon=ln(2), pielou=1.0, gini=0.5
m_a4 = _compute_tag_distribution_metrics({"a": 5, "b": 5})
_assert(m_a4["tag_count"] == 2, "A-4 tag_count=2")
_assert(_close(m_a4["shannon_h"], math.log(2)), f"A-4 均等 2 tag shannon=ln2 (got {m_a4['shannon_h']})")
_assert(_close(m_a4["pielou_evenness"], 1.0), "A-4 均等 2 tag pielou=1.0")
_assert(_close(m_a4["gini_simpson"], 0.5), f"A-4 均等 2 tag gini=0.5 (got {m_a4['gini_simpson']})")

# A-5: 均等分布 4 tag → pielou=1.0, shannon=ln(4), gini=0.75
m_a5 = _compute_tag_distribution_metrics({"a": 3, "b": 3, "c": 3, "d": 3})
_assert(_close(m_a5["pielou_evenness"], 1.0), "A-5 均等 4 tag pielou=1.0")
_assert(_close(m_a5["shannon_h"], math.log(4)), "A-5 均等 4 tag shannon=ln4")
_assert(_close(m_a5["gini_simpson"], 0.75), "A-5 均等 4 tag gini=0.75")

# A-6: 偏り分布 → pielou < 1, shannon < ln(n)
m_a6 = _compute_tag_distribution_metrics({"a": 90, "b": 10})  # 9:1
_assert(m_a6["tag_count"] == 2, "A-6 偏り tag_count=2")
_assert(
    0.0 < m_a6["pielou_evenness"] < 1.0,
    f"A-6 偏り pielou は (0,1) (got {m_a6['pielou_evenness']})",
)
_assert(
    m_a6["shannon_h"] < math.log(2),
    f"A-6 偏り shannon < ln(2) (got {m_a6['shannon_h']})",
)
_assert(
    m_a6["gini_simpson"] < 0.5,
    f"A-6 偏り gini < 0.5 (got {m_a6['gini_simpson']})",
)

# A-7: usage=0 tag は分布から除外 (混在ケース)
m_a7 = _compute_tag_distribution_metrics({"a": 5, "b": 5, "unused": 0})
_assert(m_a7["tag_count"] == 2, "A-7 usage=0 除外後 tag_count=2")
_assert(_close(m_a7["pielou_evenness"], 1.0), "A-7 usage>0 の 2 tag が均等で pielou=1.0")

# A-8: tag 数増加で shannon 上昇 (均等条件下)
m_a8_small = _compute_tag_distribution_metrics({"a": 1, "b": 1})
m_a8_large = _compute_tag_distribution_metrics({"a": 1, "b": 1, "c": 1, "d": 1})
_assert(
    m_a8_large["shannon_h"] > m_a8_small["shannon_h"],
    f"A-8 tag 数増加で shannon 上昇 (2tag={m_a8_small['shannon_h']:.3f}, 4tag={m_a8_large['shannon_h']:.3f})",
)
_assert(
    _close(m_a8_small["pielou_evenness"], m_a8_large["pielou_evenness"]),
    "A-8 両者とも均等なので pielou=1.0 で同値 (tag 数効果の除去確認)",
)


# =========================================================================
# Section B: link graph topology
# =========================================================================
print("=== Section B: link graph topology ===")

# B-1: 空 graph
m_b1 = _compute_link_graph_metrics([], memory_count=10)
_assert(m_b1["link_density"] == 0.0, "B-1 空 link で density=0")
_assert(m_b1["avg_degree"] == 0.0, "B-1 空 link で avg_degree=0")
_assert(m_b1["clustering_coefficient"] == 0.0, "B-1 空 link で clustering=0")
_assert(m_b1["degree_distribution"] == {}, "B-1 空 link で degree_dist={}")

# B-2: memory_count=0 では graph 関係なく全 0
m_b2 = _compute_link_graph_metrics([
    {"from_id": "a", "to_id": "b", "link_type": "similar", "confidence": 0.9},
], memory_count=0)
_assert(m_b2["link_density"] == 0.0, "B-2 memory=0 で density=0")
_assert(m_b2["degree_distribution"] == {}, "B-2 memory=0 で degree_dist={}")

# B-3: 単一 link (無向換算で 2 node degree=1)
links_b3 = [{"from_id": "a", "to_id": "b", "link_type": "similar", "confidence": 0.9}]
m_b3 = _compute_link_graph_metrics(links_b3, memory_count=2)
_assert(_close(m_b3["link_density"], 0.5), f"B-3 link=1/mem=2 で density=0.5 (got {m_b3['link_density']})")
_assert(_close(m_b3["avg_degree"], 1.0), f"B-3 avg_degree=1.0 (got {m_b3['avg_degree']})")
_assert(
    m_b3["degree_distribution"] == {1: 2},
    f"B-3 2 node degree=1 → {{1: 2}} (got {m_b3['degree_distribution']})",
)
_assert(
    m_b3["clustering_coefficient"] == 0.0,
    "B-3 degree<2 nodes のみ → clustering=0",
)

# B-4: 三角形 (完全 graph K3) → clustering=1.0
links_b4 = [
    {"from_id": "a", "to_id": "b", "link_type": "similar", "confidence": 0.9},
    {"from_id": "b", "to_id": "c", "link_type": "similar", "confidence": 0.9},
    {"from_id": "a", "to_id": "c", "link_type": "similar", "confidence": 0.9},
]
m_b4 = _compute_link_graph_metrics(links_b4, memory_count=3)
_assert(_close(m_b4["clustering_coefficient"], 1.0), f"B-4 K3 clustering=1.0 (got {m_b4['clustering_coefficient']})")
_assert(
    m_b4["degree_distribution"] == {2: 3},
    f"B-4 全 node degree=2 → {{2: 3}} (got {m_b4['degree_distribution']})",
)

# B-5: link_type=none / self-loop 排除
links_b5 = [
    {"from_id": "a", "to_id": "b", "link_type": "similar", "confidence": 0.9},
    {"from_id": "x", "to_id": "y", "link_type": "none", "confidence": 0.9},   # none は除外
    {"from_id": "z", "to_id": "z", "link_type": "similar", "confidence": 0.9}, # self-loop 除外
]
m_b5 = _compute_link_graph_metrics(links_b5, memory_count=5)
# 有効 link は 1 本のみ
_assert(
    _close(m_b5["link_density"], 1 / 5),
    f"B-5 有効 link 1/mem 5 で density=0.2 (got {m_b5['link_density']})",
)
_assert(
    m_b5["degree_distribution"] == {1: 2},
    f"B-5 self-loop/none 除外後 2 node degree=1 (got {m_b5['degree_distribution']})",
)


# =========================================================================
# Section C: tag 依存率
# =========================================================================
print("=== Section C: tag 依存率 ===")

# C-1: 全件 opinion/entity → opinion_entity_ratio=1.0, new_tag_usage_ratio=0
m_c1 = _compute_tag_dependency_metrics({"opinion": 10, "entity": 6})
_assert(m_c1["opinion_entity_count"] == 16, "C-1 op+ent=16")
_assert(m_c1["other_tag_count"] == 0, "C-1 other=0")
_assert(_close(m_c1["opinion_entity_ratio"], 1.0), "C-1 ratio=1.0 (11-B smoke baseline)")
_assert(_close(m_c1["new_tag_usage_ratio"], 0.0), "C-1 new_tag_usage=0")

# C-2: 非 op/ent 混在
m_c2 = _compute_tag_dependency_metrics({"opinion": 4, "entity": 2, "wm": 2, "experience": 2})
_assert(m_c2["opinion_entity_count"] == 6, "C-2 op+ent=6")
_assert(m_c2["other_tag_count"] == 4, "C-2 other (wm+experience)=4")
_assert(
    _close(m_c2["opinion_entity_ratio"], 6 / 10),
    f"C-2 ratio=0.6 (got {m_c2['opinion_entity_ratio']})",
)
_assert(
    _close(m_c2["new_tag_usage_ratio"], 4 / 10),
    f"C-2 new_tag_usage=0.4 (got {m_c2['new_tag_usage_ratio']})",
)

# C-3: 空 dict → 全 0
m_c3 = _compute_tag_dependency_metrics({})
_assert(m_c3["opinion_entity_count"] == 0, "C-3 空で op+ent=0")
_assert(_close(m_c3["opinion_entity_ratio"], 0.0), "C-3 空で ratio=0")
_assert(_close(m_c3["dynamic_origin_ratio"], 0.0), "C-3 空で dynamic_origin=0")


# =========================================================================
print("=" * 60)
_pass = sum(1 for r, _ in results if r)
_total = len(results)
print(f"結果: {_pass}/{_total} passed")
for ok, msg in results:
    if not ok:
        print(f"  FAIL: {msg}")
print("=" * 60)

sys.exit(0 if _pass == _total else 1)
