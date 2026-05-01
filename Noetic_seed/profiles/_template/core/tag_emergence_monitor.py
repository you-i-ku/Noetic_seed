"""tag_emergence_monitor — 段階11-B Phase 2 Step 2.4 + Phase 5 Step 5.4
                            + 段階11-C G-lite Phase 2 (diversity / topology 拡張)。

AI の tag 発明パターン / link graph / reconciliation 発火を集計する観察 util。
通常 loop では呼ばれない、smoke 後の手動分析や cycle 末尾の log_cycle_metrics
(emergence jsonl 記録) として呼び出す。

Phase 2 実装: collect_emergence_stats (origin / write_protected 分布)
Phase 5 実装: log_cycle_metrics (cycle 単位の複合 metric + jsonl 永続化)
段階11-C G-lite Phase 2 (§5 Step 2.1-2.4):
    - _compute_tag_distribution_metrics: Shannon H / Pielou E / Gini-Simpson
    - _compute_link_graph_metrics: link topology (density/degree/clustering)
    - _compute_tag_dependency_metrics: opinion/entity 依存率 + 新 tag 使用率
    observation 専用、scale-free 強制なし (feedback_drift_is_not_developer_error)
"""
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.config import MEMORY_DIR
from core.tag_registry import get_tag_rules, list_registered_tags


def collect_emergence_stats(state: Optional[dict] = None) -> dict:
    """起動後の tag 発明パターンを集計 (観察専用、挙動影響なし)。

    Args:
        state: 将来拡張で memory log 等と連動する時の受け皿。Phase 2 では未使用。

    Returns:
        {
            "total_registered": int,         # 登録済 tag 数
            "standard_count": int,           # origin=standard
            "dynamic_count": int,            # origin=dynamic (AI 発明)
            "write_protected_count": int,    # write_protected=True (pseudo-tag)
        }
    """
    tags = list_registered_tags()
    total = len(tags)
    standard = 0
    dynamic = 0
    write_protected = 0
    for name in tags:
        entry = get_tag_rules(name) or {}
        origin = entry.get("origin", "")
        if origin == "standard":
            standard += 1
        elif origin == "dynamic":
            dynamic += 1
        if entry.get("learning_rules", {}).get("write_protected", False):
            write_protected += 1
    return {
        "total_registered": total,
        "standard_count": standard,
        "dynamic_count": dynamic,
        "write_protected_count": write_protected,
    }


# ============================================================
# 段階11-C G-lite Phase 2: diversity / topology 観察 metric
# (§5 Phase 2 Step 2.1-2.3, observation only, 挙動影響ゼロ)
# ============================================================


def _collect_tag_usage_from_memory() -> dict:
    """全 registered tag を走査、tag 別 memory entry 件数を集計。

    Returns:
        {tag_name: int} 使用件数 (record=0 tag も key として含む、
        分布計算側で 0 件 tag は除外する方針)。
    """
    from core.memory import list_records
    counts: dict = {}
    for tag_name in list_registered_tags():
        try:
            counts[tag_name] = len(list_records(tag_name, limit=10000))
        except Exception:
            counts[tag_name] = 0
    return counts


def _compute_tag_distribution_metrics(tag_usage_counts: dict) -> dict:
    """tag 使用頻度分布から Shannon / Pielou / Gini-Simpson を計算 (Step 2.1)。

    Args:
        tag_usage_counts: {tag_name: int} 使用回数 (_collect_tag_usage_from_memory の返り値)

    Returns:
        {
            "tag_count": int,           # usage > 0 の tag 数 (分布の支持集合 size)
            "total_usage": int,         # 全 record 数
            "shannon_h": float,         # -Σ p_i ln(p_i)、tag 数増加で自然上昇
            "pielou_evenness": float,   # H / ln(N)、0-1 正規化、均等度 (主 metric)
            "gini_simpson": float,      # 1 - Σ p_i²、2 件 pick で違う tag の確率
        }

    設計意図:
        - Pielou E が主 metric (tag 数の影響を除去した均等性)
        - Shannon H は絶対値比較用、Gini-Simpson は tie-breaker / 直感解釈
        - usage=0 tag は分布から除外 (支持のない道は「多様性」の議論外)
    """
    positive = {t: c for t, c in tag_usage_counts.items() if c > 0}
    total = sum(positive.values())
    n = len(positive)
    if total == 0 or n == 0:
        return {
            "tag_count": n,
            "total_usage": total,
            "shannon_h": 0.0,
            "pielou_evenness": 0.0,
            "gini_simpson": 0.0,
        }
    probs = [c / total for c in positive.values()]
    shannon = -sum(p * math.log(p) for p in probs)
    if n > 1:
        evenness = shannon / math.log(n)
    else:
        # 単一 tag は p=1 なので shannon=0、evenness の定義上 0 とする
        # (均等化すべき空間が存在しない状態)
        evenness = 0.0
    gini_simpson = 1.0 - sum(p * p for p in probs)
    return {
        "tag_count": n,
        "total_usage": total,
        "shannon_h": shannon,
        "pielou_evenness": evenness,
        "gini_simpson": gini_simpson,
    }


def _compute_link_graph_metrics(links: list, memory_count: int) -> dict:
    """link graph topology 集計 (Step 2.2, 観察のみ、scale-free 強制なし)。

    Args:
        links: list_links() の返り値 (memory_links.jsonl entry list)
        memory_count: 全 memory entry 件数 (density 算出の分母)

    Returns:
        {
            "link_density": float,              # link_count / max(1, memory_count)
            "avg_degree": float,                # 2 * link_count / memory_count (無向視)
            "clustering_coefficient": float,    # 局所 clustering 平均 (三角形率)
            "degree_distribution": dict,        # {degree_int: node_count} (§11-e full dict 確定)
        }

    設計意図 (reference_information_existence_graph_foundations):
        Broido & Clauset 2019 より実世界 network の 2/3 以上は power-law ではない。
        本 metric は観察記録のみ、power-law fit / scale-free 判定は行わない。
    """
    if memory_count <= 0 or not links:
        return {
            "link_density": 0.0,
            "avg_degree": 0.0,
            "clustering_coefficient": 0.0,
            "degree_distribution": {},
        }
    # 無向隣接 (link は有向だが G-lite は両向き走査可能 follow_links と整合)
    adj: dict = {}
    valid_link_count = 0
    for link in links:
        if link.get("link_type", "none") == "none":
            continue
        from_id = link.get("from_id")
        to_id = link.get("to_id")
        if not from_id or not to_id or from_id == to_id:
            continue
        adj.setdefault(from_id, set()).add(to_id)
        adj.setdefault(to_id, set()).add(from_id)
        valid_link_count += 1
    link_density = valid_link_count / max(1, memory_count)
    avg_degree = (2.0 * valid_link_count) / memory_count if memory_count > 0 else 0.0
    # degree distribution (full dict, §11-e 確定)
    degree_dist: dict = {}
    for neighbors in adj.values():
        d = len(neighbors)
        degree_dist[d] = degree_dist.get(d, 0) + 1
    # clustering coefficient (平均局所、k<2 の node は分母から除外)
    total_cc = 0.0
    node_count_for_cc = 0
    for node, neighbors in adj.items():
        k = len(neighbors)
        if k < 2:
            continue
        possible_pairs = k * (k - 1) / 2
        nbrs = list(neighbors)
        actual_pairs = 0
        for i in range(len(nbrs)):
            ni_adj = adj.get(nbrs[i], set())
            for j in range(i + 1, len(nbrs)):
                if nbrs[j] in ni_adj:
                    actual_pairs += 1
        total_cc += actual_pairs / possible_pairs
        node_count_for_cc += 1
    clustering = total_cc / node_count_for_cc if node_count_for_cc > 0 else 0.0
    return {
        "link_density": link_density,
        "avg_degree": avg_degree,
        "clustering_coefficient": clustering,
        "degree_distribution": degree_dist,
    }


def _compute_tag_dependency_metrics(tag_usage_counts: dict) -> dict:
    """opinion/entity 依存率 + 新 tag 使用率 (Step 2.3)。

    Args:
        tag_usage_counts: {tag_name: int} 使用件数

    Returns:
        {
            "opinion_entity_count": int,        # op+ent の record 合計
            "other_tag_count": int,             # 非 op/ent の record 合計
            "opinion_entity_ratio": float,      # (op+ent) / total
            "dynamic_origin_ratio": float,      # dynamic origin tag 数 / total_registered
            "new_tag_usage_ratio": float,       # 非 opinion/entity tag 使用率
        }

    baseline (11-B smoke 2nd, cycle 41): opinion_entity_ratio=1.0 / new_tag_usage_ratio=0.0。
    smoke 3 段目で follow_links + cold_start の緩和効果を経時追跡。
    """
    opinion_entity_count = 0
    other_tag_count = 0
    dynamic_registered = 0
    total_registered = 0
    for tag_name, cnt in tag_usage_counts.items():
        total_registered += 1
        entry = get_tag_rules(tag_name) or {}
        if entry.get("origin", "") == "dynamic":
            dynamic_registered += 1
        if tag_name in ("opinion", "entity"):
            opinion_entity_count += cnt
        else:
            other_tag_count += cnt
    total_usage = opinion_entity_count + other_tag_count
    opinion_entity_ratio = (opinion_entity_count / total_usage) if total_usage > 0 else 0.0
    new_tag_usage_ratio = (other_tag_count / total_usage) if total_usage > 0 else 0.0
    dynamic_origin_ratio = (
        dynamic_registered / total_registered if total_registered > 0 else 0.0
    )
    return {
        "opinion_entity_count": opinion_entity_count,
        "other_tag_count": other_tag_count,
        "opinion_entity_ratio": opinion_entity_ratio,
        "dynamic_origin_ratio": dynamic_origin_ratio,
        "new_tag_usage_ratio": new_tag_usage_ratio,
    }


def _default_emergence_log_file() -> Path:
    """Phase 5 emergence jsonl の既定保存先: {profile}/logs/phase5_emergence.jsonl"""
    return MEMORY_DIR.parent / "logs" / "phase5_emergence.jsonl"


def log_cycle_metrics(cycle_idx: int,
                      state: Optional[dict] = None, *,
                      log_file: Optional[Path] = None) -> dict:
    """cycle 終端で呼出、phase5_emergence.jsonl に append (Phase 5 Step 5.4)。

    Phase 5 時点の最小集計:
      - collect_emergence_stats (Phase 2 実装) の全 field
      - memory_count (全 network の entry 合計)
      - link_count (memory_links.jsonl の行数)
      - link_grad_density (link_count / memory_count、Phase 4 retrieval 拡張判定
        の主要 metric、> 0.2 なら follow_links 実装検討)
      - reconciliation_ec_count (Phase 3 reconciliation 由来 EC 誤差件数)

    段階11-C G-lite Phase 2 拡張 (§5 Step 2.4):
      - tag 分布: tag_count / total_usage / shannon_h / pielou_evenness / gini_simpson
      - link topology: link_density / avg_degree / clustering_coefficient / degree_distribution
      - tag 依存率: opinion_entity_count / other_tag_count / opinion_entity_ratio /
                    dynamic_origin_ratio / new_tag_usage_ratio
      既存 field (memory_count / link_count / link_grad_density 等) は全保持。

    Args:
        cycle_idx: cycle 番号 (main.py cycle loop の idx)
        state: 集計元 state (None で空 dict 扱い)
        log_file: 出力先 Path (None で _default_emergence_log_file)

    Returns:
        append した metrics dict (副作用: jsonl 追記)
    """
    from core.memory_links import list_links

    stats = collect_emergence_stats(state)

    # memory 全件カウント + tag 別使用度 (G-lite Phase 2 で tag_usage 併用)
    tag_usage = _collect_tag_usage_from_memory()
    mem_count = sum(tag_usage.values())

    # link count + density (既存、link_grad_density は backward compat で保持)
    links = list_links(limit=10000)
    link_count = len(links)
    link_grad_density = link_count / max(1, mem_count)

    # reconciliation EC 件数 (cumulative、cycle 間 diff は smoke 後分析側で計算)
    recon_hist: list = []
    if state is not None:
        recon_hist = state.get("prediction_error_history_by_source", {}).get("reconciliation", [])
    recon_ec_count = len(recon_hist)

    # G-lite Phase 2 新規 metric 3 群
    tag_dist = _compute_tag_distribution_metrics(tag_usage)
    link_topo = _compute_link_graph_metrics(links, mem_count)
    tag_dep = _compute_tag_dependency_metrics(tag_usage)

    # 段階11-D Phase 6 Step 6.1: small-world index (毎 cycle 計算、観察のみ)
    small_world = _compute_small_world_metrics(links, mem_count)

    # 段階11-D Phase 6 Step 6.2: 直近 reflect の cluster MI を state から拾う
    # (reflect 時のみ更新、cycle 間は前値継続。reflect なし cycle は前回値が log)
    phase6_state = (state or {}).get("phase6_metrics", {}) if state is not None else {}

    metrics = {
        "cycle": int(cycle_idx),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **stats,
        "memory_count": mem_count,
        "link_count": link_count,
        "link_grad_density": link_grad_density,
        "reconciliation_ec_count": recon_ec_count,
        # G-lite Phase 2 新規 (既存 field とは key 名重複なし)
        **tag_dist,    # tag_count / total_usage / shannon_h / pielou_evenness / gini_simpson
        **link_topo,   # link_density / avg_degree / clustering_coefficient / degree_distribution
        **tag_dep,     # opinion_entity_count / other_tag_count / opinion_entity_ratio /
                       # dynamic_origin_ratio / new_tag_usage_ratio
        # 段階11-D Phase 6 新規 (small-world は毎 cycle、cluster_* は reflect 時に state 経由 update)
        **small_world, # avg_path_length / small_world_sigma / small_world_C_random / small_world_L_random
        "cluster_mi": phase6_state.get("cluster_mi", 0.0),
        "cluster_inter_ratio": phase6_state.get("cluster_inter_ratio", 0.0),
        "cluster_link_pairs": phase6_state.get("last_cluster_link_pairs", 0),
        "phase6_last_reflect_cycle": phase6_state.get("last_reflect_cycle"),
    }

    target_file = log_file if log_file is not None else _default_emergence_log_file()
    target_file.parent.mkdir(exist_ok=True, parents=True)
    with open(target_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    return metrics


# ============================================================
# 段階11-D Phase 6: small-world index + cluster MI (観察のみ)
# (§5 Phase 6 Step 6.1 + 6.2、Step 6.3 power-law fit は v1.2 で drop)
# 設計参照:
#   - Watts-Strogatz 1998 (small-world index σ)
#   - Broido & Clauset 2019 (line 149-151 で power-law fit を排除済、v1.2 で PLAN 整合)
# ============================================================


def _compute_average_shortest_path(adj: dict) -> float:
    """BFS で全 pair shortest path 平均 (disconnected pair は除外)。

    Args:
        adj: {node_id: set(neighbor_ids)} 無向隣接 dict

    Returns:
        average shortest path length。node 数 < 2 → 0.0、全 disconnected → 0.0。
    """
    nodes = list(adj.keys())
    n = len(nodes)
    if n < 2:
        return 0.0
    total_dist = 0
    pair_count = 0
    for src in nodes:
        # BFS
        dist = {src: 0}
        queue = [src]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            for nxt in adj.get(cur, set()):
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        for tgt, d in dist.items():
            if tgt != src:
                total_dist += d
                pair_count += 1
    return total_dist / pair_count if pair_count > 0 else 0.0


def _generate_random_graph_metrics(node_count: int, edge_count: int) -> dict:
    """Erdős-Renyi G(n, m) ランダムグラフの C_random / L_random 解析近似。

    Watts-Strogatz 1998 の正統手法: ランダムグラフの解析式を使い、実 graph と
    同じ node 数 / edge 数で比較 (新マジックナンバー追加なし、Watts-Strogatz
    1998 reference)。

    Args:
        node_count: ノード数 (= memory_count)
        edge_count: 有向 link を無向視した edge 数

    Returns:
        {"C_random": float, "L_random": float}
        avg_degree < 1 -> disconnected expected、L_random=0.0
    """
    if node_count < 2 or edge_count < 1:
        return {"C_random": 0.0, "L_random": 0.0}
    avg_degree = 2.0 * edge_count / node_count
    # ER graph: C_random = p ~ <k>/(n-1)
    C_random = avg_degree / max(1, node_count - 1)
    # ER graph: L_random ~ ln(n) / ln(<k>)、avg_degree > 1 のみ妥当
    if avg_degree > 1.0:
        L_random = math.log(node_count) / math.log(avg_degree)
    else:
        L_random = 0.0
    return {"C_random": C_random, "L_random": L_random}


def _compute_small_world_metrics(links: list, memory_count: int) -> dict:
    """Watts-Strogatz small-world index sigma = (C/C_random) / (L/L_random)。

    Phase 6 Step 6.1 (PLAN §5 Phase 6)。観察のみ、強制なし。
    sigma > 1 で small-world、ただし判定 threshold は持たない
    (`feedback_drift_is_not_developer_error`)。

    Args:
        links: list_links() の戻り値
        memory_count: 全 memory 件数

    Returns:
        {
            "avg_path_length": float,         # L (実 graph)
            "small_world_sigma": float,       # sigma (>1 で small-world)
            "small_world_C_random": float,    # C_random (ER 解析近似)
            "small_world_L_random": float,    # L_random (ER 解析近似)
        }
    """
    if memory_count < 2 or not links:
        return {
            "avg_path_length": 0.0,
            "small_world_sigma": 0.0,
            "small_world_C_random": 0.0,
            "small_world_L_random": 0.0,
        }
    # 無向隣接構築 (link_type=none 除外、_compute_link_graph_metrics と同方式)
    adj: dict = {}
    valid_link_count = 0
    for link in links:
        if link.get("link_type", "none") == "none":
            continue
        from_id = link.get("from_id")
        to_id = link.get("to_id")
        if not from_id or not to_id or from_id == to_id:
            continue
        adj.setdefault(from_id, set()).add(to_id)
        adj.setdefault(to_id, set()).add(from_id)
        valid_link_count += 1

    L = _compute_average_shortest_path(adj)

    # C 計算 (_compute_link_graph_metrics と同じロジック、独立計算)
    total_cc = 0.0
    cc_node_count = 0
    for neighbors in adj.values():
        k = len(neighbors)
        if k < 2:
            continue
        possible_pairs = k * (k - 1) / 2
        nbrs = list(neighbors)
        actual_pairs = 0
        for i in range(len(nbrs)):
            ni_adj = adj.get(nbrs[i], set())
            for j in range(i + 1, len(nbrs)):
                if nbrs[j] in ni_adj:
                    actual_pairs += 1
        total_cc += actual_pairs / possible_pairs
        cc_node_count += 1
    C = total_cc / cc_node_count if cc_node_count > 0 else 0.0

    rand = _generate_random_graph_metrics(memory_count, valid_link_count)
    C_rand = rand["C_random"]
    L_rand = rand["L_random"]

    if C_rand > 0 and L_rand > 0 and L > 0:
        sigma = (C / C_rand) / (L / L_rand)
    else:
        sigma = 0.0

    return {
        "avg_path_length": L,
        "small_world_sigma": sigma,
        "small_world_C_random": C_rand,
        "small_world_L_random": L_rand,
    }


def compute_cluster_mutual_information(clusters: list, links: list) -> dict:
    """cluster membership と link 構造の相互情報量 (Phase 6 Step 6.2)。

    無向 link 集合に対し X = link 片端の cluster / Y = 他端の cluster と定義、
    両方向カウントで I(X;Y) = sum p(x,y) log2(p(x,y)/(p(x)p(y))) 計算。

    Phase 5 estimate_clusters の出力を入力にする (reflect 時のみ呼出)。
    高 MI = cluster 構造と link 構造が強相関、低 MI = cluster と link が独立。
    観察のみ、強制なし。

    Args:
        clusters: estimate_clusters の戻り値
                  [{"cluster_id": str, "memory_ids": [...], ...}, ...]
        links: list_links() の戻り値

    Returns:
        {
            "cluster_mi": float,            # MI (bit、>=0、数値誤差で負は 0 clamp)
            "cluster_link_pairs": int,      # 集計対象 link 数 (両端 cluster 既知)
            "cluster_inter_ratio": float,   # cluster 跨ぎ link 比率
        }
    """
    if not clusters or not links:
        return {"cluster_mi": 0.0, "cluster_link_pairs": 0, "cluster_inter_ratio": 0.0}

    # memory_id -> cluster_id mapping
    mem_to_cluster: dict = {}
    for cluster in clusters:
        cid = cluster.get("cluster_id", "")
        for mid in cluster.get("memory_ids", []):
            mem_to_cluster[mid] = cid

    if not mem_to_cluster:
        return {"cluster_mi": 0.0, "cluster_link_pairs": 0, "cluster_inter_ratio": 0.0}

    joint: dict = {}            # {(c_a, c_b): count} ordered pair (両方向カウント)
    cluster_marginal: dict = {} # {cluster_id: total_endpoint_count}
    inter_count = 0
    valid_pairs = 0
    for link in links:
        if link.get("link_type", "none") == "none":
            continue
        from_id = link.get("from_id")
        to_id = link.get("to_id")
        if not from_id or not to_id or from_id == to_id:
            continue
        ca = mem_to_cluster.get(from_id)
        cb = mem_to_cluster.get(to_id)
        if ca is None or cb is None:
            continue
        if ca != cb:
            inter_count += 1
        valid_pairs += 1
        # 無向化: 両方向カウント
        for x, y in ((ca, cb), (cb, ca)):
            joint[(x, y)] = joint.get((x, y), 0) + 1
            cluster_marginal[x] = cluster_marginal.get(x, 0) + 1

    if valid_pairs == 0:
        return {"cluster_mi": 0.0, "cluster_link_pairs": 0, "cluster_inter_ratio": 0.0}

    total = sum(joint.values())  # = 2 * valid_pairs
    mi = 0.0
    for (x, y), n_xy in joint.items():
        p_xy = n_xy / total
        p_x = cluster_marginal[x] / total
        p_y = cluster_marginal[y] / total
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))

    inter_ratio = inter_count / valid_pairs if valid_pairs > 0 else 0.0

    return {
        "cluster_mi": max(0.0, mi),  # 数値誤差 clamp
        "cluster_link_pairs": valid_pairs,
        "cluster_inter_ratio": inter_ratio,
    }
