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
    }

    target_file = log_file if log_file is not None else _default_emergence_log_file()
    target_file.parent.mkdir(exist_ok=True, parents=True)
    with open(target_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    return metrics
