"""UPS v2 (Unified Pending Schema v2) テスト。

project_pending_unification.md §2-§4 の仕様を網羅:
  - pending_add: スキーマ生成、priority 自動計算
  - pending_observe: gap 更新、priority 降順消化、match_source_actions
  - pending_prune: protected / time / dynamic_n 別の淘汰
  - calc_priority: gap × lag_weight × channel_multiplier
  - pending_recalc_priorities: 一括再計算

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_pending_unified.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pending_unified import (
    CHANNEL_MULTIPLIERS,
    LAG_WEIGHTS,
    calc_priority,
    pending_add,
    pending_observe,
    pending_prune,
    pending_recalc_priorities,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _fresh_state():
    return {
        "cycle_id": 10,
        "log": [],
        "pending": [],
    }


# ============================================================
# calc_priority
# ============================================================

def test_priority_basic():
    print("== calc_priority: gap × lag × channel の基本 ==")
    # gap=0.5, lag="minutes" (3.0), channel="device" (2.0) → 0.5*3*2 = 3.0
    p = calc_priority({
        "gap": 0.5,
        "observation_lag_kind": "minutes",
        "observed_channel": "device",
    })
    return _assert(abs(p - 3.0) < 1e-6, f"計算値 {p} == 3.0")


def test_priority_all_lag_kinds():
    print("== calc_priority: 全 lag_kind の weight 適用 ==")
    expected = {
        "seconds": 1.0,
        "minutes": 3.0,
        "hours":   2.0,
        "cycles":  1.5,
        "unknown": 3.0,
    }
    results = []
    for lag, w in expected.items():
        p = calc_priority({
            "gap": 1.0,
            "observation_lag_kind": lag,
            "observed_channel": None,
        })
        results.append(_assert(abs(p - w) < 1e-6, f"{lag}={w}"))
    return all(results)


def test_priority_unknown_defaults():
    print("== calc_priority: 未知 lag_kind / None channel の fallback ==")
    # 未知 lag_kind → unknown(3.0), channel 未知 → 1.0
    p1 = calc_priority({
        "gap": 1.0,
        "observation_lag_kind": "weeks",  # 未知
        "observed_channel": None,
    })
    p2 = calc_priority({
        "gap": 1.0,
        "observation_lag_kind": "seconds",
        "observed_channel": "mars",  # 未知
    })
    return all([
        _assert(abs(p1 - 3.0) < 1e-6, f"未知 lag → unknown (3.0): {p1}"),
        _assert(abs(p2 - 1.0) < 1e-6, f"未知 channel → 1.0: {p2}"),
    ])


def test_priority_missing_fields():
    print("== calc_priority: entry にフィールドが無い場合は default 値 ==")
    p = calc_priority({})
    # gap=1.0 default, lag="unknown"(3.0), channel=None(1.0) → 3.0
    return _assert(abs(p - 3.0) < 1e-6, f"空 entry → 3.0: {p}")


# ============================================================
# pending_add
# ============================================================

def test_add_basic():
    print("== pending_add: 基本スキーマ生成 ==")
    state = _fresh_state()
    entry = pending_add(
        state, source_action="output_display",
        expected_observation="ゆうからの返答",
        lag_kind="minutes", content="ゆうへの応答",
        cycle_id=10, channel="device",
    )
    return all([
        _assert(entry["type"] == "pending", "type=pending"),
        _assert(entry["source_action"] == "output_display", "source_action"),
        _assert(entry["observation_lag_kind"] == "minutes", "lag_kind"),
        _assert(entry["gap"] == 1.0, "初期 gap=1.0"),
        _assert(entry["observed_content"] is None, "未観測=None"),
        _assert(entry["origin_cycle"] == 10, "origin_cycle"),
        _assert(entry["expiry_policy"] == "dynamic_n", "default policy"),
        _assert(state["pending"][0] is entry, "state['pending'] に追加"),
    ])


def test_add_priority_auto_calc():
    print("== pending_add: priority 自動計算 ==")
    state = _fresh_state()
    # gap=1.0, lag="minutes"(3.0), channel="device"(2.0) → 6.0
    entry = pending_add(
        state, source_action="output_display",
        expected_observation="返答",
        lag_kind="minutes", content="x",
        cycle_id=10, channel="device",
    )
    return _assert(abs(entry["priority"] - 6.0) < 1e-6,
                   f"priority=6.0 自動計算 (実={entry['priority']})")


def test_add_living_presence():
    print("== pending_add: living_presence (spontaneous 受容枠) ==")
    state = _fresh_state()
    entry = pending_add(
        state, source_action="living_presence",
        expected_observation="(spontaneous 到着)",
        lag_kind="unknown", content="外部到着待ち",
        cycle_id=10, channel=None,  # まだ channel 未定
    )
    return all([
        _assert(entry["source_action"] == "living_presence",
                "living_presence 受容"),
        _assert(entry["observation_lag_kind"] == "unknown", "unknown lag"),
        # unknown(3.0) × None(1.0) × gap(1.0) = 3.0
        _assert(abs(entry["priority"] - 3.0) < 1e-6,
                f"priority=3.0 (unknown は高め): {entry['priority']}"),
    ])


def test_add_empty_state():
    print("== pending_add: state['pending'] 未初期化でも setdefault ==")
    state = {"cycle_id": 0, "log": []}  # pending key なし
    pending_add(
        state, source_action="x_post",
        expected_observation="反応",
        lag_kind="hours", content="x 投稿",
        cycle_id=0, channel="x",
    )
    return _assert("pending" in state and len(state["pending"]) == 1,
                   "setdefault で作成され追加")


# ============================================================
# pending_observe
# ============================================================

def test_observe_basic():
    print("== pending_observe: 1 件マッチで gap=0 + observed_* 埋まる ==")
    state = _fresh_state()
    pending_add(
        state, source_action="output_display",
        expected_observation="返答", lag_kind="minutes",
        content="x", cycle_id=10, channel="device",
    )
    updated = pending_observe(
        state, observed_content="はーい",
        channel="device", cycle_id=11,
    )
    entry = state["pending"][0]
    return all([
        _assert(len(updated) == 1, "1 件更新"),
        _assert(entry["observed_content"] == "はーい", "observed_content"),
        _assert(entry["observed_channel"] == "device", "observed_channel"),
        _assert(entry["gap"] == 0.0, "gap=0.0"),
        _assert(entry["last_cycle"] == 11, "last_cycle"),
        _assert(entry["priority"] == 0.0, "priority=0 (gap 0 なので)"),
    ])


def test_observe_priority_descending():
    print("== pending_observe: priority 降順で上位から消化 ==")
    state = _fresh_state()
    # 低 priority: lag="seconds"(1.0), channel=None(1.0), gap=1.0 → 1.0
    low = pending_add(
        state, source_action="E_eval",
        expected_observation="低", lag_kind="seconds",
        content="低", cycle_id=0, channel=None,
    )
    # 高 priority: lag="minutes"(3.0), channel="device"(2.0), gap=1.0 → 6.0
    high = pending_add(
        state, source_action="output_display",
        expected_observation="高", lag_kind="minutes",
        content="高", cycle_id=0, channel="device",
    )
    updated = pending_observe(
        state, observed_content="obs", channel="device",
        cycle_id=1, limit=1,
    )
    return all([
        _assert(len(updated) == 1, "limit=1"),
        _assert(updated[0]["id"] == high["id"], "高 priority が先に消化"),
        _assert(low["observed_content"] is None, "低は未消化"),
    ])


def test_observe_match_source_actions():
    print("== pending_observe: match_source_actions で絞り込み ==")
    state = _fresh_state()
    p1 = pending_add(
        state, source_action="elyth_post",
        expected_observation="反応", lag_kind="hours",
        content="e", cycle_id=0, channel="elyth",
    )
    p2 = pending_add(
        state, source_action="output_display",
        expected_observation="返答", lag_kind="minutes",
        content="o", cycle_id=0, channel="device",
    )
    updated = pending_observe(
        state, observed_content="reply", channel="elyth",
        cycle_id=1, match_source_actions=["elyth_post"],
    )
    return all([
        _assert(len(updated) == 1, "1 件のみ"),
        _assert(updated[0]["id"] == p1["id"], "elyth_post が消化"),
        _assert(p2["observed_content"] is None, "output_display は温存"),
    ])


def test_observe_skips_already_observed():
    print("== pending_observe: 既に observed 済みは skip ==")
    state = _fresh_state()
    pending_add(
        state, source_action="output_display",
        expected_observation="返答", lag_kind="minutes",
        content="x", cycle_id=0, channel="device",
    )
    pending_observe(
        state, observed_content="first", channel="device", cycle_id=1,
    )
    # 2 回目の observation は消化先なし
    updated2 = pending_observe(
        state, observed_content="second", channel="device", cycle_id=2,
    )
    return all([
        _assert(len(updated2) == 0, "該当なし"),
        _assert(state["pending"][0]["observed_content"] == "first",
                "元の observation は上書きされない"),
    ])


def test_observe_no_match_returns_empty():
    print("== pending_observe: pending なしで空 list ==")
    state = _fresh_state()
    updated = pending_observe(
        state, observed_content="x", channel="device", cycle_id=0,
    )
    return _assert(updated == [], "空 list")


def test_observe_limit_multiple():
    print("== pending_observe: limit > 1 で複数消化 ==")
    state = _fresh_state()
    for i in range(3):
        pending_add(
            state, source_action="output_display",
            expected_observation=f"r{i}", lag_kind="minutes",
            content=f"c{i}", cycle_id=0, channel="device",
        )
    updated = pending_observe(
        state, observed_content="bulk", channel="device",
        cycle_id=1, limit=2,
    )
    observed = sum(1 for p in state["pending"] if p["observed_content"])
    return all([
        _assert(len(updated) == 2, "2 件更新"),
        _assert(observed == 2, "state 上も 2 件 observed"),
    ])


# ============================================================
# pending_prune
# ============================================================

def test_prune_protected_kept():
    print("== pending_prune: protected は常に残る ==")
    state = _fresh_state()
    pending_add(
        state, source_action="living_presence",
        expected_observation="永続", lag_kind="unknown",
        content="p", cycle_id=0, channel=None,
        expiry_policy="protected",
    )
    dropped = pending_prune(state, current_cycle=1000)
    return all([
        _assert(dropped == 0, "削除なし"),
        _assert(len(state["pending"]) == 1, "1 件残る"),
    ])


def test_prune_time_expired():
    print("== pending_prune: time 期限切れは消える ==")
    state = _fresh_state()
    pending_add(
        state, source_action="E_eval",
        expected_observation="即", lag_kind="seconds",
        content="e", cycle_id=0, channel=None,
        expiry_policy="time", ttl_cycles=5,
    )
    # cycle 0 追加、cycle 4 で prune (4 < 5 → 生きる)
    dropped1 = pending_prune(state, current_cycle=4)
    # cycle 10 で prune (10 - 0 = 10 >= 5 → 死ぬ)
    dropped2 = pending_prune(state, current_cycle=10)
    return all([
        _assert(dropped1 == 0, "期限内は残る"),
        _assert(dropped2 == 1, "期限超過で削除"),
        _assert(len(state["pending"]) == 0, "空になる"),
    ])


def test_prune_dynamic_n_top():
    print("== pending_prune: dynamic_n で gap 上位 N のみ残す ==")
    state = _fresh_state()
    # gap 0.9, 0.5, 0.3, 0.1 の 4 件追加
    for gap in [0.9, 0.5, 0.3, 0.1]:
        pending_add(
            state, source_action="reflection",
            expected_observation=f"g{gap}", lag_kind="cycles",
            content=f"gap{gap}", cycle_id=0, channel="self",
            initial_gap=gap,
        )
    dropped = pending_prune(state, current_cycle=1, dynamic_n=2)
    remaining_gaps = sorted([p["gap"] for p in state["pending"]], reverse=True)
    return all([
        _assert(dropped == 2, "2 件削除"),
        _assert(len(state["pending"]) == 2, "2 件残る"),
        _assert(remaining_gaps == [0.9, 0.5], f"上位 2 件: {remaining_gaps}"),
    ])


def test_prune_dynamic_n_from_log():
    print("== pending_prune: dynamic_n=None → log 長から自動 (max3, min20) ==")
    state = _fresh_state()
    # log 10 件 → max(3, min(20, 10//5)) = max(3, 2) = 3
    state["log"] = [{"cycle": i} for i in range(10)]
    for gap in [0.9, 0.7, 0.5, 0.3, 0.1]:
        pending_add(
            state, source_action="reflection",
            expected_observation=f"g{gap}", lag_kind="cycles",
            content=f"gap{gap}", cycle_id=0, channel="self",
            initial_gap=gap,
        )
    pending_prune(state, current_cycle=1, dynamic_n=None)
    return _assert(len(state["pending"]) == 3,
                   f"log 10 件 → 上位 3 残る (実={len(state['pending'])})")


def test_prune_ignores_non_ups():
    print("== pending_prune: UPS v2 以外 (旧形式) は touch しない ==")
    state = _fresh_state()
    # 旧形式 (Phase 3 以前): type="external_message" / "unresolved_intent"
    state["pending"].append({
        "type": "external_message", "content": "legacy", "priority": 9.0,
    })
    state["pending"].append({
        "type": "unresolved_intent", "content": "old", "gap": 0.1,
    })
    # UPS v2 を 3 件追加
    for gap in [0.9, 0.3, 0.1]:
        pending_add(
            state, source_action="reflection",
            expected_observation="x", lag_kind="cycles",
            content="x", cycle_id=0, channel="self",
            initial_gap=gap,
        )
    pending_prune(state, current_cycle=1, dynamic_n=1)
    types = [p.get("type") for p in state["pending"]]
    return all([
        _assert(types.count("external_message") == 1, "legacy external_message 保持"),
        _assert(types.count("unresolved_intent") == 1, "legacy unresolved 保持"),
        _assert(types.count("pending") == 1, "UPS v2 は dynamic_n=1 で 1 件"),
    ])


# ============================================================
# pending_recalc_priorities
# ============================================================

def test_recalc_priorities():
    print("== pending_recalc_priorities: 全 UPS v2 を再計算 ==")
    state = _fresh_state()
    entry = pending_add(
        state, source_action="output_display",
        expected_observation="x", lag_kind="minutes",
        content="x", cycle_id=0, channel=None,  # channel 未確定
    )
    # channel 未確定時 priority = 1.0 * 3.0 * 1.0 = 3.0
    assert abs(entry["priority"] - 3.0) < 1e-6

    # 手動で channel を上書き (spontaneous 到着で channel 後から判明)
    entry["observed_channel"] = "device"
    # 再計算しないと priority は古いまま
    n = pending_recalc_priorities(state)
    # 再計算後: 1.0 * 3.0 * 2.0 = 6.0 (device multiplier)
    return all([
        _assert(n == 1, "1 件再計算"),
        _assert(abs(entry["priority"] - 6.0) < 1e-6,
                f"priority 更新: {entry['priority']}"),
    ])


def test_recalc_skips_non_ups():
    print("== pending_recalc_priorities: 旧形式 pending は skip ==")
    state = _fresh_state()
    state["pending"].append({"type": "external_message", "priority": 9.0})
    pending_add(
        state, source_action="x_post",
        expected_observation="r", lag_kind="hours",
        content="x", cycle_id=0, channel="x",
    )
    n = pending_recalc_priorities(state)
    return all([
        _assert(n == 1, "UPS v2 の 1 件のみ再計算"),
        _assert(state["pending"][0]["priority"] == 9.0,
                "legacy priority 保持"),
    ])


# ============================================================
# 統合シナリオ
# ============================================================

def test_integration_add_observe_prune():
    print("== 統合: add 5 → observe 2 → prune で dynamic_n 動作 ==")
    state = _fresh_state()
    state["log"] = [{"cycle": i} for i in range(20)]  # dynamic_n = 4

    # 5 件追加 (gap 0.9, 0.7, 0.5, 0.3, 0.1)
    entries = []
    for gap in [0.9, 0.7, 0.5, 0.3, 0.1]:
        entries.append(pending_add(
            state, source_action="output_display",
            expected_observation=f"g{gap}", lag_kind="minutes",
            content=f"gap{gap}", cycle_id=0, channel="device",
            initial_gap=gap,
        ))

    # 上位 2 件 observe → gap 0 に
    observed = pending_observe(
        state, observed_content="obs", channel="device",
        cycle_id=1, limit=2,
    )

    # prune: dynamic_n=None → log 20 件 → max(3, 20//5=4) = 4
    # observed した 2 件 (gap=0) は残すべきじゃない (gap 低い → 捨てられる候補)
    # でも observe 後の gap=0 は「消化済み」の意味。実装は gap 降順で上位 N
    # なので observe 済み 2 件 (gap=0) は下位 → 淘汰、未 observe 3 件 (gap
    # 0.5, 0.3, 0.1) が残るべき。dynamic_n=4 だとギリ全部残る。
    pending_prune(state, current_cycle=2, dynamic_n=None)
    remaining = state["pending"]
    return all([
        _assert(len(observed) == 2, "observe 2 件"),
        _assert(observed[0]["gap"] == 0.0, "observe 後 gap=0"),
        _assert(len(remaining) == 4, f"prune 後 dynamic_n=4 で 4 件残る (実={len(remaining)})"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("priority: 基本", test_priority_basic),
        ("priority: 全 lag_kind", test_priority_all_lag_kinds),
        ("priority: 未知 → default", test_priority_unknown_defaults),
        ("priority: 空 entry → default", test_priority_missing_fields),
        ("add: 基本スキーマ", test_add_basic),
        ("add: priority 自動", test_add_priority_auto_calc),
        ("add: living_presence", test_add_living_presence),
        ("add: state 未初期化", test_add_empty_state),
        ("observe: 基本", test_observe_basic),
        ("observe: priority 降順", test_observe_priority_descending),
        ("observe: match_source_actions", test_observe_match_source_actions),
        ("observe: observed 済み skip", test_observe_skips_already_observed),
        ("observe: 該当なし → 空 list", test_observe_no_match_returns_empty),
        ("observe: limit 複数", test_observe_limit_multiple),
        ("prune: protected 残る", test_prune_protected_kept),
        ("prune: time 期限切れ", test_prune_time_expired),
        ("prune: dynamic_n 上位 N", test_prune_dynamic_n_top),
        ("prune: dynamic_n None → log 長", test_prune_dynamic_n_from_log),
        ("prune: 旧形式 skip", test_prune_ignores_non_ups),
        ("recalc: 全 UPS v2 再計算", test_recalc_priorities),
        ("recalc: 旧形式 skip", test_recalc_skips_non_ups),
        ("統合: add→observe→prune", test_integration_add_observe_prune),
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
