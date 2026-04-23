"""tag_emergence_monitor — 段階11-B Phase 2 Step 2.4。

AI の tag 発明パターンを集計する **観察専用** util。
通常 loop では呼ばれない、smoke 後の手動分析や Phase 5 Step 5.4 の
`log_cycle_metrics` (emergence jsonl 記録) の一部として呼び出す。

Phase 2 での責務: 登録済 tag の origin / write_protected 分布の最小集計。
Phase 5 で Zipf 適合度 / consideration_engagement / hallucination_rate 等を
本モジュールに段階拡張する予定 (正典 PLAN §5 Step 5.4)。
"""
from typing import Optional

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
