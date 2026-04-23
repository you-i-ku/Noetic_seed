"""tag_emergence_monitor — 段階11-B Phase 2 Step 2.4 + Phase 5 Step 5.4。

AI の tag 発明パターン / link graph / reconciliation 発火を集計する観察 util。
通常 loop では呼ばれない、smoke 後の手動分析や cycle 末尾の log_cycle_metrics
(emergence jsonl 記録) として呼び出す。

Phase 2 実装: collect_emergence_stats (origin / write_protected 分布)
Phase 5 実装: log_cycle_metrics (cycle 単位の複合 metric + jsonl 永続化)

Phase 5+ 拡張余地 (smoke 2 段目観察後に必要性判断):
- usage_freq_zipf_ratio (tag 使用頻度の Zipf 適合度)
- hallucination_rate (承認 reject 比率)
- consideration_engagement (Phase 2' 撤去後は不要化、affordance 再評価時に)
"""
import json
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

    Phase 5+ 拡張予定 (smoke 観察後):
      - usage_freq_zipf_ratio / hallucination_rate / その他 emergence 指標

    Args:
        cycle_idx: cycle 番号 (main.py cycle loop の idx)
        state: 集計元 state (None で空 dict 扱い)
        log_file: 出力先 Path (None で _default_emergence_log_file)

    Returns:
        append した metrics dict (副作用: jsonl 追記)
    """
    from core.memory import list_records
    from core.memory_links import list_links

    stats = collect_emergence_stats(state)

    # memory 全件カウント (全 registered network)
    mem_count = 0
    for tag_name in list_registered_tags():
        # pseudo-tag (write_protected) は jsonl 持たない可能性 → 失敗 graceful
        try:
            mem_count += len(list_records(tag_name, limit=10000))
        except Exception:
            continue

    # link count + density
    link_count = len(list_links(limit=10000))
    link_density = link_count / max(1, mem_count)

    # reconciliation EC 件数 (cumulative、cycle 間 diff は smoke 後分析側で計算)
    recon_hist: list = []
    if state is not None:
        recon_hist = state.get("prediction_error_history_by_source", {}).get("reconciliation", [])
    recon_ec_count = len(recon_hist)

    metrics = {
        "cycle": int(cycle_idx),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **stats,
        "memory_count": mem_count,
        "link_count": link_count,
        "link_grad_density": link_density,
        "reconciliation_ec_count": recon_ec_count,
    }

    target_file = log_file if log_file is not None else _default_emergence_log_file()
    target_file.parent.mkdir(exist_ok=True, parents=True)
    with open(target_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    return metrics
