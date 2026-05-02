"""world_fact_view — 段階13 Phase 0.3: raw_events.jsonl を view する独立 tool。

PLAN §18-A4 (vii) ゆう発案 literal 継承の name。memory_graph_tool (subjective
layer = LLM 解釈・自己モデルの graph view) と並走する raw layer (immutable
monitor footage = 物理世界の fact = tool 実行 + 結果) の view 経路。

哲学整合:
- raw_events.jsonl = source of truth、append-only、永続不変
- subjective_entries.jsonl と id 共有、merge_log_view で合本可能 (内部利用)
- world_fact_view は LLM が「過去の事実を確認したい」時に呼ぶ affordance
  (subjective layer の memory_graph_tool と並走、layer 越境せず)

API: world_fact_view(args)
  args:
    mode      : "recent" (default) / "by_tool" / "by_channel"
    filter_tool   : tool 名 (mode=by_tool 時必須、他 mode では追加 filter)
    filter_channel: channel 名 (mode=by_channel 時必須、他 mode では追加 filter)
    limit         : 件数 (default 20、最大 100)

戻り値: 中立 JSON 構造化 text (memory_graph_tool 同精神、自然言語ゼロ、
feedback_llm_as_brain 整合)。
"""
import json
from typing import Optional

from core.state import load_state


DEFAULT_LIMIT = 20
MAX_LIMIT = 100


def _coerce_limit(raw_limit) -> int:
    """args.limit を int 化、範囲 clamp。"""
    try:
        n = int(raw_limit)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return max(1, min(MAX_LIMIT, n))


def _filter_events(events: list, mode: str,
                   filter_tool: Optional[str],
                   filter_channel: Optional[str]) -> list:
    """raw_events を filter で絞り込む。

    mode は呼出側 validation 用 (by_tool は filter_tool 必須等)、本関数では
    mode に依存せず filter_tool / filter_channel を **常に conjunctive 適用**
    する (Codex review 0.3 BLOCKER 反映: 全 mode で combined filter が直感整合)。
    """
    filtered = list(events)
    if filter_tool:
        filtered = [e for e in filtered if e.get("tool", "") == filter_tool]
    if filter_channel:
        filtered = [e for e in filtered if e.get("channel", "") == filter_channel]
    return filtered


def _summarize_event(entry: dict, result_cap: int = 200) -> dict:
    """raw event entry を view 用に整形 (long result は cap)。"""
    result = entry.get("result", "")
    result_str = str(result)
    if len(result_str) > result_cap:
        result_str = result_str[:result_cap] + f"...[{len(str(result))}字]"
    return {
        "id": entry.get("id", ""),
        "time": entry.get("time", ""),
        "channel": entry.get("channel", ""),
        "tool": entry.get("tool", ""),
        "args": entry.get("args"),
        "result": result_str,
    }


def _world_fact_view(args: dict) -> str:
    """world_fact_view tool 本体。raw_events.jsonl の view を返す。

    LLM が呼ぶ entry point。state は load_state で fresh 取得。
    """
    mode = str(args.get("mode", "recent")).strip() or "recent"
    if mode not in ("recent", "by_tool", "by_channel"):
        return json.dumps({
            "error": f"unknown mode: {mode}",
            "valid_modes": ["recent", "by_tool", "by_channel"],
        }, ensure_ascii=False)

    filter_tool = args.get("filter_tool")
    if filter_tool is not None:
        filter_tool = str(filter_tool).strip() or None
    filter_channel = args.get("filter_channel")
    if filter_channel is not None:
        filter_channel = str(filter_channel).strip() or None

    if mode == "by_tool" and not filter_tool:
        return json.dumps({
            "error": "mode=by_tool requires filter_tool",
        }, ensure_ascii=False)
    if mode == "by_channel" and not filter_channel:
        return json.dumps({
            "error": "mode=by_channel requires filter_channel",
        }, ensure_ascii=False)

    limit = _coerce_limit(args.get("limit"))

    state = load_state()
    raw_events = state.get("raw_events", [])
    total = len(raw_events)

    filtered = _filter_events(raw_events, mode, filter_tool, filter_channel)
    matched = len(filtered)
    # 新しい順 (末尾) から limit 件
    selected = filtered[-limit:] if limit < matched else filtered
    # 表示は時系列降順 (新しい→古い、LLM が直近を最初に見やすい)
    selected_view = [_summarize_event(e) for e in reversed(selected)]

    return json.dumps({
        "mode": mode,
        "filter_tool": filter_tool,
        "filter_channel": filter_channel,
        "limit": limit,
        "total_raw_events": total,
        "matched": matched,
        "returned": len(selected_view),
        "events": selected_view,
    }, ensure_ascii=False, indent=2)
