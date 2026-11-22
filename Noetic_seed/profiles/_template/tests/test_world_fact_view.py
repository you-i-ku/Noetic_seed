"""段階13 Phase 0.3 — world_fact_view tool テスト。

PLAN §18-A4 (vii) ゆう発案 literal 継承の name。raw_events.jsonl を view する
独立 tool (memory_graph_tool subjective layer と並走の raw layer 経路)。

API:
  mode      : "recent" (default) / "by_tool" / "by_channel"
  filter_tool / filter_channel
  limit     : 1〜100 (default 20)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_world_fact_view.py
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.world_fact_view_tool import _world_fact_view
from core.runtime.registry import ToolError


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _fake_state(events: list) -> dict:
    """raw_events をセットしただけの最小 state。"""
    return {"raw_events": events, "subjective_entries": []}


def _events_basic() -> list:
    return [
        {"id": "e1", "time": "09:00", "channel": "device",
         "tool": "output_display", "args": {"x": 1}, "result": "OK"},
        {"id": "e2", "time": "09:05", "channel": "internal",
         "tool": "read_file", "args": {"path": "a"}, "result": "AAA"},
        {"id": "e3", "time": "09:10", "channel": "device",
         "tool": "output_display", "args": {"x": 2}, "result": "OK2"},
        {"id": "e4", "time": "09:15", "channel": "x",
         "tool": "x_post", "args": {"text": "hi"}, "result": "posted"},
    ]


def _run_tool(args: dict, events: list, *, expect_error=False) -> dict:
    """JSONL reader を patch して tool を実行、JSON parse して返す。"""
    with patch("tools.world_fact_view_tool.read_memory_jsonl",
               return_value=events):
        if expect_error:
            try:
                _world_fact_view(args)
            except ToolError as exc:
                out = str(exc)
            else:
                raise AssertionError("expected ToolError")
        else:
            out = _world_fact_view(args)
    return json.loads(out)


def test_recent_default():
    print("== mode=recent (default): 全件、新しい順、limit=20 ==")
    res = _run_tool({}, _events_basic())
    return all([
        _assert(res["mode"] == "recent", "mode=recent"),
        _assert(res["total_raw_events"] == 4, "total=4"),
        _assert(res["matched"] == 4, "matched=4"),
        _assert(res["returned"] == 4, "returned=4"),
        _assert(res["events"][0]["id"] == "e4", "新しい順 先頭=e4"),
        _assert(res["events"][-1]["id"] == "e1", "最古=e1"),
    ])


def test_by_tool():
    print("== mode=by_tool, filter_tool=output_display ==")
    res = _run_tool(
        {"mode": "by_tool", "filter_tool": "output_display"},
        _events_basic(),
    )
    return all([
        _assert(res["matched"] == 2, "matched=2 (e1, e3)"),
        _assert(res["events"][0]["id"] == "e3", "新しい順 先頭=e3"),
        _assert(res["events"][1]["id"] == "e1", "次=e1"),
    ])


def test_by_channel():
    print("== mode=by_channel, filter_channel=device ==")
    res = _run_tool(
        {"mode": "by_channel", "filter_channel": "device"},
        _events_basic(),
    )
    return all([
        _assert(res["matched"] == 2, "matched=2 (e1, e3)"),
        _assert(all(e["channel"] == "device" for e in res["events"]),
                "全 entry channel=device"),
    ])


def test_recent_with_filter_tool():
    print("== mode=recent + filter_tool=read_file (combined filter) ==")
    res = _run_tool(
        {"filter_tool": "read_file"},
        _events_basic(),
    )
    return all([
        _assert(res["matched"] == 1, "matched=1 (e2 のみ)"),
        _assert(res["events"][0]["id"] == "e2", "id=e2"),
    ])


def _events_combined() -> list:
    """combined filter test 専用 fixture (CLAUDE.md §5 識別力)。

    各 condition が異なる subset を選ぶよう 4 種混在:
      a: 両方一致 (tool=output_display, channel=device)
      b: tool のみ一致 (tool=output_display, channel=x)
      c: channel のみ一致 (tool=read_file, channel=device)
      d: 両方不一致 (tool=read_file, channel=x)

    filter_tool=output_display 単独 → a, b (2 件)
    filter_channel=device 単独    → a, c (2 件)
    AND (combined)                → a のみ (1 件)
    → 単独と combined で結果集合が異なる = 誤実装 (片方 ignore) 検出可能
    """
    return [
        {"id": "a", "time": "1", "channel": "device",
         "tool": "output_display", "result": ""},
        {"id": "b", "time": "2", "channel": "x",
         "tool": "output_display", "result": ""},
        {"id": "c", "time": "3", "channel": "device",
         "tool": "read_file", "result": ""},
        {"id": "d", "time": "4", "channel": "x",
         "tool": "read_file", "result": ""},
    ]


def test_by_tool_with_extra_filter_channel():
    """段階13 Phase 0.3 修正 (Codex review BLOCKER 反映): by_tool でも filter_channel
    は conjunctive 追加 filter として動作する。

    識別力 fixture (CLAUDE.md §5): 単独 filter で 2 件 / combined で 1 件、
    片方 ignore 誤実装なら matched=2 で fail。
    """
    print("== mode=by_tool + filter_channel (conjunctive 追加) ==")
    res = _run_tool(
        {"mode": "by_tool", "filter_tool": "output_display",
         "filter_channel": "device"},
        _events_combined(),
    )
    return all([
        _assert(res["matched"] == 1,
                f"matched=1 (a のみ、両方一致 entry、got {res['matched']})"),
        _assert(len(res["events"]) == 1, "events 1 件のみ"),
        _assert(res["events"][0]["id"] == "a",
                f"id=a (got {res['events'][0]['id']})"),
        _assert(res["events"][0]["tool"] == "output_display",
                "tool=output_display"),
        _assert(res["events"][0]["channel"] == "device",
                "channel=device"),
    ])


def test_by_channel_with_extra_filter_tool():
    """段階13 Phase 0.3 修正: by_channel でも filter_tool は conjunctive 追加 filter。

    識別力 fixture (CLAUDE.md §5): 単独 filter で 2 件 / combined で 1 件、
    片方 ignore 誤実装なら matched=2 で fail。
    """
    print("== mode=by_channel + filter_tool (conjunctive 追加) ==")
    res = _run_tool(
        {"mode": "by_channel", "filter_channel": "device",
         "filter_tool": "output_display"},
        _events_combined(),
    )
    return all([
        _assert(res["matched"] == 1,
                f"matched=1 (a のみ、両方一致 entry、got {res['matched']})"),
        _assert(len(res["events"]) == 1, "events 1 件のみ"),
        _assert(res["events"][0]["id"] == "a",
                f"id=a (got {res['events'][0]['id']})"),
        _assert(res["events"][0]["channel"] == "device",
                "channel=device"),
        _assert(res["events"][0]["tool"] == "output_display",
                "tool=output_display"),
    ])


def test_combined_baseline_single_filters():
    """combined fixture が単独 filter で 2 件出ることを確認 (識別力 baseline)。

    CLAUDE.md §5 fixture 設計検証: 単独 filter と combined で結果集合が
    本当に異なるか、baseline で確認しておく。
    """
    print("== combined fixture: 単独 filter で 2 件 (識別力 baseline) ==")
    events = _events_combined()
    res_tool = _run_tool({"filter_tool": "output_display"}, events)
    res_channel = _run_tool({"filter_channel": "device"}, events)
    return all([
        _assert(res_tool["matched"] == 2,
                f"filter_tool 単独 → 2 件 (got {res_tool['matched']})"),
        _assert(res_channel["matched"] == 2,
                f"filter_channel 単独 → 2 件 (got {res_channel['matched']})"),
        _assert({e["id"] for e in res_tool["events"]} == {"a", "b"},
                "filter_tool 結果 = {a, b}"),
        _assert({e["id"] for e in res_channel["events"]} == {"a", "c"},
                "filter_channel 結果 = {a, c}"),
    ])


def test_limit_clamp():
    print("== limit clamp: max=100 / 不正値 → default 20 ==")
    res_max = _run_tool({"limit": 9999}, _events_basic())
    res_default = _run_tool({"limit": "abc"}, _events_basic())
    return all([
        _assert(res_max["limit"] == 100, f"max cap (got {res_max['limit']})"),
        _assert(res_default["limit"] == 20,
                f"不正値 → default (got {res_default['limit']})"),
    ])


def test_limit_truncation():
    print("== limit=2 で末尾 2 件、新しい順 ==")
    res = _run_tool({"limit": 2}, _events_basic())
    return all([
        _assert(res["matched"] == 4, "matched=4 (filter なし)"),
        _assert(res["returned"] == 2, "returned=2"),
        _assert(res["events"][0]["id"] == "e4", "新しい順 先頭=e4"),
        _assert(res["events"][1]["id"] == "e3", "次=e3"),
    ])


def test_unknown_mode_error():
    print("== unknown mode → error JSON ==")
    res = _run_tool({"mode": "weird"}, _events_basic(), expect_error=True)
    return all([
        _assert("error" in res, "error key 存在"),
        _assert("weird" in res["error"], "error msg に mode 名"),
    ])


def test_by_tool_missing_filter_error():
    print("== mode=by_tool で filter_tool 未指定 → error ==")
    res = _run_tool({"mode": "by_tool"}, _events_basic(), expect_error=True)
    return _assert("error" in res, "error key 存在")


def test_by_channel_missing_filter_error():
    print("== mode=by_channel で filter_channel 未指定 → error ==")
    res = _run_tool({"mode": "by_channel"}, _events_basic(), expect_error=True)
    return _assert("error" in res, "error key 存在")


def test_long_result_capped():
    print("== long result は 200 字 cap + 全字数 footer ==")
    long_result = "x" * 500
    events = [{"id": "e1", "time": "09:00", "channel": "internal",
               "tool": "bash", "result": long_result}]
    res = _run_tool({}, events)
    rendered = res["events"][0]["result"]
    return all([
        _assert(len(rendered) < 500, f"短縮された (len={len(rendered)})"),
        _assert("500字" in rendered, "全字数 footer 表示"),
    ])


def test_empty_raw_events():
    print("== raw_events 空 → 空 events ==")
    res = _run_tool({}, [])
    return all([
        _assert(res["total_raw_events"] == 0, "total=0"),
        _assert(res["matched"] == 0, "matched=0"),
        _assert(res["events"] == [], "events 空"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    print("test_world_fact_view.py (段階13 Phase 0.3)")
    print("=" * 60)

    groups = [
        ("mode=recent default", test_recent_default),
        ("mode=by_tool", test_by_tool),
        ("mode=by_channel", test_by_channel),
        ("mode=recent + filter_tool combined", test_recent_with_filter_tool),
        ("mode=by_tool + filter_channel (conjunctive)", test_by_tool_with_extra_filter_channel),
        ("mode=by_channel + filter_tool (conjunctive)", test_by_channel_with_extra_filter_tool),
        ("combined fixture 識別力 baseline", test_combined_baseline_single_filters),
        ("limit clamp (max / invalid)", test_limit_clamp),
        ("limit truncation (新しい順)", test_limit_truncation),
        ("unknown mode error", test_unknown_mode_error),
        ("by_tool 必須 filter_tool 欠落", test_by_tool_missing_filter_error),
        ("by_channel 必須 filter_channel 欠落", test_by_channel_missing_filter_error),
        ("long result cap", test_long_result_capped),
        ("空 raw_events", test_empty_raw_events),
    ]

    results = []
    for label, fn in groups:
        print()
        ok = fn()
        results.append((label, ok))

    print()
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"  {passed}/{total} groups passed")

    sys.exit(0 if passed == total else 1)
