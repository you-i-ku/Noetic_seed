"""段階11-D Phase 6 (Step 6.1 + 6.2): small-world index + cluster MI tests。

PLAN §5 Phase 6:
  - Step 6.1: Watts-Strogatz small-world index sigma = (C/C_rand) / (L/L_rand)
  - Step 6.2: cluster mutual information (Phase 5 estimate_clusters 出力 + memory_links)
  - Step 6.3: power-law fit は v1.2 で drop (Broido & Clauset 2019 整合)
  - Step 6.4: 本 test (small-world + MI sanity)

実行:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_emergence_full_metrics.py
"""
import os
import sys
from pathlib import Path

PROFILE_PATH = Path(__file__).parent.parent
os.environ["NOETIC_PROFILE"] = str(PROFILE_PATH)
sys.path.insert(0, str(PROFILE_PATH))

from core.tag_emergence_monitor import (
    _compute_small_world_metrics,
    compute_cluster_mutual_information,
    _generate_random_graph_metrics,
    _compute_average_shortest_path,
)


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


# =========================================================
# Step 6.1: small-world index
# =========================================================


def test_small_world_empty_graph():
    print("== small_world: 空 graph -> sigma=0, L=0 ==")
    metrics = _compute_small_world_metrics([], 0)
    _assert(metrics["small_world_sigma"] == 0.0, "空 graph sigma=0")
    _assert(metrics["avg_path_length"] == 0.0, "空 graph L=0")
    metrics = _compute_small_world_metrics([], 5)
    _assert(metrics["small_world_sigma"] == 0.0, "link 空 sigma=0")
    return True


def test_small_world_complete_graph():
    print("== small_world: 5-node 完全グラフ で sigma 計算成立 ==")
    nodes = ["a", "b", "c", "d", "e"]
    links = []
    for i, x in enumerate(nodes):
        for y in nodes[i + 1:]:
            links.append({"from_id": x, "to_id": y, "link_type": "similar"})
    metrics = _compute_small_world_metrics(links, len(nodes))
    # 完全グラフ: C=1.0, L=1.0
    _assert(abs(metrics["avg_path_length"] - 1.0) < 0.01,
            f"完全グラフ L=1.0, got {metrics['avg_path_length']}")
    # ER 解析近似で C_random / L_random > 0、sigma > 0
    _assert(metrics["small_world_C_random"] > 0,
            f"C_random>0, got {metrics['small_world_C_random']}")
    _assert(metrics["small_world_sigma"] > 0,
            f"完全グラフ sigma>0 (numerical sanity), got {metrics['small_world_sigma']}")
    return True


def test_average_shortest_path_chain():
    print("== avg_shortest_path: 5-node chain で L=2.0 ==")
    # chain a-b-c-d-e、無向
    adj = {
        "a": {"b"},
        "b": {"a", "c"},
        "c": {"b", "d"},
        "d": {"c", "e"},
        "e": {"d"},
    }
    L = _compute_average_shortest_path(adj)
    # ordered pairs: 5*4=20、distance |i-j| in 0..4
    # 各 distance d (1..4) は (5-d)*2 ordered pair → sum = 2*(4+6+6+4) = 40
    # avg = 40/20 = 2.0
    _assert(abs(L - 2.0) < 0.01, f"chain L=2.0, got {L}")
    return True


def test_average_shortest_path_disconnected():
    print("== avg_shortest_path: disconnected 部分は除外 ==")
    # a-b と c-d の 2 component
    adj = {
        "a": {"b"},
        "b": {"a"},
        "c": {"d"},
        "d": {"c"},
    }
    L = _compute_average_shortest_path(adj)
    # 連結 pair のみ: a-b, b-a, c-d, d-c の 4 ordered pair、全部 distance 1
    # avg = 4/4 = 1.0
    _assert(abs(L - 1.0) < 0.01, f"disconnected L=1.0, got {L}")
    return True


def test_random_graph_metrics_analytic():
    print("== random_graph_metrics: ER 解析式 sanity ==")
    # n=100, m=200 (avg_degree=4)
    metrics = _generate_random_graph_metrics(100, 200)
    # L_random = ln(100)/ln(4) = 4.605/1.386 ~ 3.32
    _assert(2.5 < metrics["L_random"] < 4.5,
            f"L_random ~ 3.32, got {metrics['L_random']}")
    # C_random = 4/(100-1) ~ 0.0404
    _assert(0.02 < metrics["C_random"] < 0.06,
            f"C_random ~ 0.04, got {metrics['C_random']}")
    return True


def test_random_graph_metrics_boundary():
    print("== random_graph_metrics: 境界条件 (n<2, m<1, avg_degree<1) ==")
    _assert(_generate_random_graph_metrics(0, 0)["L_random"] == 0.0, "n=0 -> L=0")
    _assert(_generate_random_graph_metrics(1, 0)["C_random"] == 0.0, "n=1 -> C=0")
    # avg_degree < 1 (n=10, m=2, <k>=0.4)
    metrics = _generate_random_graph_metrics(10, 2)
    _assert(metrics["L_random"] == 0.0, f"avg_degree<1 -> L_random=0, got {metrics['L_random']}")
    return True


# =========================================================
# Step 6.2: cluster mutual information
# =========================================================


def test_mi_empty_inputs():
    print("== MI: 空入力 -> 0 ==")
    m = compute_cluster_mutual_information([], [])
    _assert(m["cluster_mi"] == 0.0, "空空 MI=0")
    m = compute_cluster_mutual_information([], [{"from_id": "a", "to_id": "b", "link_type": "similar"}])
    _assert(m["cluster_mi"] == 0.0, "cluster 空 -> MI=0")
    m = compute_cluster_mutual_information([{"cluster_id": "c1", "memory_ids": ["a"]}], [])
    _assert(m["cluster_mi"] == 0.0, "link 空 -> MI=0")
    return True


def test_mi_strongly_clustered():
    print("== MI: 全 link が cluster 内 -> 高 MI (~ log2(2) = 1.0) ==")
    clusters = [
        {"cluster_id": "c1", "memory_ids": ["m1", "m2", "m3"]},
        {"cluster_id": "c2", "memory_ids": ["m4", "m5", "m6"]},
    ]
    links = [
        {"from_id": "m1", "to_id": "m2", "link_type": "similar"},
        {"from_id": "m1", "to_id": "m3", "link_type": "similar"},
        {"from_id": "m2", "to_id": "m3", "link_type": "similar"},
        {"from_id": "m4", "to_id": "m5", "link_type": "similar"},
        {"from_id": "m4", "to_id": "m6", "link_type": "similar"},
        {"from_id": "m5", "to_id": "m6", "link_type": "similar"},
    ]
    metrics = compute_cluster_mutual_information(clusters, links)
    # 全 cluster 内 link、2 cluster 均等 -> MI ~ log2(2) = 1.0
    _assert(metrics["cluster_mi"] > 0.5,
            f"cluster 内のみ link -> MI 高い (>0.5), got {metrics['cluster_mi']}")
    _assert(metrics["cluster_inter_ratio"] == 0.0,
            f"全 cluster 内 -> inter_ratio=0, got {metrics['cluster_inter_ratio']}")
    _assert(metrics["cluster_link_pairs"] == 6,
            f"link 6 件, got {metrics['cluster_link_pairs']}")
    return True


def test_mi_balanced_inter_intra():
    print("== MI: cluster 内外 link 均等 -> 低 MI ==")
    clusters = [
        {"cluster_id": "c1", "memory_ids": ["m1", "m2", "m3", "m4"]},
        {"cluster_id": "c2", "memory_ids": ["m5", "m6", "m7", "m8"]},
    ]
    # 内 link 4 + 外 link 4 = 完全均等
    links = [
        {"from_id": "m1", "to_id": "m2", "link_type": "similar"},
        {"from_id": "m3", "to_id": "m4", "link_type": "similar"},
        {"from_id": "m5", "to_id": "m6", "link_type": "similar"},
        {"from_id": "m7", "to_id": "m8", "link_type": "similar"},
        {"from_id": "m1", "to_id": "m5", "link_type": "similar"},
        {"from_id": "m2", "to_id": "m6", "link_type": "similar"},
        {"from_id": "m3", "to_id": "m7", "link_type": "similar"},
        {"from_id": "m4", "to_id": "m8", "link_type": "similar"},
    ]
    metrics = compute_cluster_mutual_information(clusters, links)
    # 完全均等分布 -> MI ~ 0 (cluster 構造と link 構造が独立)
    _assert(metrics["cluster_mi"] < 0.1,
            f"均等分布 MI ~ 0 (<0.1), got {metrics['cluster_mi']}")
    _assert(0.4 < metrics["cluster_inter_ratio"] < 0.6,
            f"inter_ratio ~ 0.5, got {metrics['cluster_inter_ratio']}")
    return True


def test_mi_unknown_membership_skip():
    print("== MI: cluster に属さない memory の link は skip ==")
    clusters = [
        {"cluster_id": "c1", "memory_ids": ["m1", "m2"]},
    ]
    links = [
        {"from_id": "m1", "to_id": "m2", "link_type": "similar"},  # OK
        {"from_id": "m1", "to_id": "unknown", "link_type": "similar"},  # skip
        {"from_id": "ghost1", "to_id": "ghost2", "link_type": "similar"},  # skip
    ]
    metrics = compute_cluster_mutual_information(clusters, links)
    _assert(metrics["cluster_link_pairs"] == 1,
            f"valid link 1 のみ, got {metrics['cluster_link_pairs']}")
    return True


def test_mi_link_type_none_filtered():
    print("== MI: link_type=none は filter ==")
    clusters = [
        {"cluster_id": "c1", "memory_ids": ["m1", "m2"]},
    ]
    links = [
        {"from_id": "m1", "to_id": "m2", "link_type": "none"},
    ]
    metrics = compute_cluster_mutual_information(clusters, links)
    _assert(metrics["cluster_mi"] == 0.0, "link_type=none で MI=0")
    _assert(metrics["cluster_link_pairs"] == 0,
            f"link_type=none で valid_pairs=0, got {metrics['cluster_link_pairs']}")
    return True


# =========================================================
# main
# =========================================================


if __name__ == "__main__":
    print("test_emergence_full_metrics.py (段階11-D Phase 6 Step 6.1 + 6.2)")
    print("=" * 60)
    tests = [
        # Step 6.1: small-world
        test_small_world_empty_graph,
        test_small_world_complete_graph,
        test_average_shortest_path_chain,
        test_average_shortest_path_disconnected,
        test_random_graph_metrics_analytic,
        test_random_graph_metrics_boundary,
        # Step 6.2: cluster MI
        test_mi_empty_inputs,
        test_mi_strongly_clustered,
        test_mi_balanced_inter_intra,
        test_mi_unknown_membership_skip,
        test_mi_link_type_none_filtered,
    ]
    pass_count = 0
    for fn in tests:
        try:
            ok = fn()
            if ok:
                pass_count += 1
                print(f"  OK  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print("=" * 60)
    print(f"{pass_count}/{len(tests)} passed")
    sys.exit(0 if pass_count == len(tests) else 1)
