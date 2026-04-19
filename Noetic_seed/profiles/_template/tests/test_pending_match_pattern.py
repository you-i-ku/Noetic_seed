"""段階8 v4 — pending 側 match_pattern 対称消化判定テスト。

WORLD_MODEL_DESIGN/STAGE8_REPETITION_AND_PREDICTOR_PLAN.md §4-2 の仕様を網羅:
  - _matches: tool_name_any / channel_match / content_similarity_threshold
  - try_observe_all: match_pattern 駆動の自動消化、priority 降順、1 消化/1 実行

設計哲学:
  - tool 側 rules ゼロ、pending 側 match_pattern が自己消化条件を持つ
  - Active Inference 対称性: tool = 行動→observation / pending = 期待→match
  - 全 tool が同じ hook で処理される (特別扱いゼロ)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_pending_match_pattern.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import pending_unified
from core.pending_unified import (
    _matches,
    pending_add,
    try_observe_all,
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
        "session_id": "test",
    }


def _install_sim_mock(fn):
    """_sim_check を差し替えて embedding 依存を排除 (test 専用)。"""
    original = pending_unified._sim_check
    pending_unified._sim_check = fn
    return original


# ============================================================
# _matches: 個別フィールドの判定
# ============================================================

def test_matches_tool_name_any_hit():
    print("== _matches: tool_name_any リスト内で True ==")
    mp = {"tool_name_any": ["output_display", "elyth_post"]}
    pending = {"expected_channel": "device"}
    ok = _matches(mp, "output_display", {}, "result", "device", pending)
    return _assert(ok is True, "output_display が候補にあるので True")


def test_matches_tool_name_any_miss():
    print("== _matches: tool_name_any 外で False ==")
    mp = {"tool_name_any": ["output_display"]}
    pending = {"expected_channel": "device"}
    ok = _matches(mp, "bash", {}, "result", "device", pending)
    return _assert(ok is False, "bash は候補にないので False")


def test_matches_tool_name_any_none_means_any():
    print("== _matches: tool_name_any=None はどの tool でも OK ==")
    mp = {"tool_name_any": None}
    pending = {"expected_channel": "device"}
    ok = _matches(mp, "random_tool", {}, "r", "device", pending)
    return _assert(ok is True, "None は skip 扱い (全 tool 許可)")


def test_matches_channel_match_hit():
    print("== _matches: channel_match で channel 一致なら True ==")
    mp = {"channel_match": True}
    pending = {"expected_channel": "device"}
    ok = _matches(mp, "output_display", {}, "r", "device", pending)
    return _assert(ok is True, "channel 一致")


def test_matches_channel_match_mismatch():
    print("== _matches: channel_match ミスマッチで False ==")
    mp = {"channel_match": True}
    pending = {"expected_channel": "device"}
    ok = _matches(mp, "output_display", {}, "r", "claude", pending)
    return _assert(ok is False, "device pending だが claude 実行 → False")


def test_matches_similarity_hit():
    print("== _matches: content_similarity_threshold >= 閾値で True ==")
    original = _install_sim_mock(lambda a, b, t: True)  # 常に類似度 OK
    try:
        mp = {"content_similarity_threshold": 0.7}
        pending = {"content": "foo bar"}
        ok = _matches(mp, "search_memory", {}, "result bar foo", None, pending)
        return _assert(ok is True, "類似度 >= 閾値で True")
    finally:
        pending_unified._sim_check = original


def test_matches_similarity_miss():
    print("== _matches: content_similarity_threshold < 閾値で False ==")
    original = _install_sim_mock(lambda a, b, t: False)
    try:
        mp = {"content_similarity_threshold": 0.9}
        pending = {"content": "foo"}
        ok = _matches(mp, "search_memory", {}, "totally unrelated", None, pending)
        return _assert(ok is False, "類似度 < 閾値で False")
    finally:
        pending_unified._sim_check = original


def test_matches_all_fields_and():
    print("== _matches: 複数フィールドは AND 判定 ==")
    original = _install_sim_mock(lambda a, b, t: True)
    try:
        mp = {
            "tool_name_any": ["output_display"],
            "channel_match": True,
            "content_similarity_threshold": 0.5,
        }
        pending = {"expected_channel": "device", "content": "reply"}
        # 全条件 OK
        ok_all = _matches(mp, "output_display", {}, "reply", "device", pending)
        # tool_name だけ外れる
        pending_unified._sim_check = lambda a, b, t: True
        fail_tool = _matches(mp, "bash", {}, "reply", "device", pending)
        # channel だけ外れる
        fail_ch = _matches(mp, "output_display", {}, "reply", "claude", pending)
        return all([
            _assert(ok_all is True, "全条件 OK で True"),
            _assert(fail_tool is False, "tool_name 外れ で False"),
            _assert(fail_ch is False, "channel 外れ で False"),
        ])
    finally:
        pending_unified._sim_check = original


# ============================================================
# try_observe_all: 実際の消化フロー
# ============================================================

def test_try_observe_tool_name_match():
    print("== try_observe_all: tool_name 一致の pending を消化 ==")
    state = _fresh_state()
    p = pending_add(
        state, source_action="response_to_external",
        expected_observation="返信", lag_kind="cycles",
        content="おねーたんへの返事", cycle_id=0, channel="claude",
        match_pattern={"tool_name_any": ["output_display"], "channel_match": True},
    )
    updated = try_observe_all(
        state=state, tool_name="output_display",
        tool_args={"channel": "claude", "content": "hi"},
        tool_result="送信完了 (claude): hi",
        channel="claude", cycle_id=1,
    )
    return all([
        _assert(len(updated) == 1, "1 件消化"),
        _assert(updated[0]["id"] == p["id"], "対象 pending が消化"),
        _assert(p["observed_content"] is not None, "observed_content 埋まった"),
        _assert(p["gap"] == 0.0, "gap=0"),
    ])


def test_try_observe_tool_name_miss_skips():
    print("== try_observe_all: tool_name 外れなら消化しない ==")
    state = _fresh_state()
    p = pending_add(
        state, source_action="response_to_external",
        expected_observation="返信", lag_kind="cycles",
        content="返事", cycle_id=0, channel="device",
        match_pattern={"tool_name_any": ["output_display"]},
    )
    updated = try_observe_all(
        state=state, tool_name="bash",
        tool_args={}, tool_result="shell output",
        channel="self", cycle_id=1,
    )
    return all([
        _assert(len(updated) == 0, "消化ゼロ"),
        _assert(p["observed_content"] is None, "pending は未消化のまま"),
    ])


def test_try_observe_no_match_pattern_skips():
    print("== try_observe_all: match_pattern なし pending は消化されない ==")
    state = _fresh_state()
    # match_pattern を付与せず pending_add
    p = pending_add(
        state, source_action="reflection",
        expected_observation="reflection", lag_kind="cycles",
        content="何かの intent", cycle_id=0, channel="self",
        # match_pattern は default None
    )
    updated = try_observe_all(
        state=state, tool_name="output_display",
        tool_args={"channel": "self"}, tool_result="dummy",
        channel="self", cycle_id=1,
    )
    return all([
        _assert(len(updated) == 0, "match_pattern なしは消化対象外"),
        _assert(p["observed_content"] is None, "pending 残る"),
    ])


def test_try_observe_priority_wins():
    print("== try_observe_all: 複数 match あるとき priority 最高のみ消化 ==")
    state = _fresh_state()
    # 低 priority (channel=None で multiplier 1.0)
    low = pending_add(
        state, source_action="response_to_external",
        expected_observation="低", lag_kind="seconds",
        content="低 priority", cycle_id=0, channel=None,
        match_pattern={"tool_name_any": ["output_display"]},
    )
    # 高 priority (channel="device" で multiplier 2.0, lag="minutes" で 3.0)
    high = pending_add(
        state, source_action="response_to_external",
        expected_observation="高", lag_kind="minutes",
        content="高 priority", cycle_id=0, channel="device",
        match_pattern={"tool_name_any": ["output_display"]},
    )
    updated = try_observe_all(
        state=state, tool_name="output_display",
        tool_args={"channel": "device"}, tool_result="sent",
        channel="device", cycle_id=1,
    )
    return all([
        _assert(len(updated) == 1, "1 件だけ消化 (limit=1)"),
        _assert(updated[0]["id"] == high["id"], "高 priority が優先"),
        _assert(low["observed_content"] is None, "低はまだ消化されない"),
    ])


def test_try_observe_already_observed_skipped():
    print("== try_observe_all: 既に消化済 pending は再消化しない ==")
    state = _fresh_state()
    p = pending_add(
        state, source_action="response_to_external",
        expected_observation="返信", lag_kind="cycles",
        content="response", cycle_id=0, channel="device",
        match_pattern={"tool_name_any": ["output_display"]},
    )
    # 最初の消化
    first = try_observe_all(
        state=state, tool_name="output_display",
        tool_args={"channel": "device"}, tool_result="first",
        channel="device", cycle_id=1,
    )
    # 同じ条件でもう一度
    second = try_observe_all(
        state=state, tool_name="output_display",
        tool_args={"channel": "device"}, tool_result="second",
        channel="device", cycle_id=2,
    )
    return all([
        _assert(len(first) == 1, "初回は消化"),
        _assert(len(second) == 0, "2 回目は何もしない"),
    ])


def test_try_observe_channel_mismatch_skips():
    print("== try_observe_all: channel_match で channel ミスマッチなら skip ==")
    state = _fresh_state()
    p = pending_add(
        state, source_action="response_to_external",
        expected_observation="返信", lag_kind="cycles",
        content="reply", cycle_id=0, channel="device",
        match_pattern={"tool_name_any": ["output_display"], "channel_match": True},
    )
    # tool は claude channel で実行
    updated = try_observe_all(
        state=state, tool_name="output_display",
        tool_args={"channel": "claude"}, tool_result="sent to claude",
        channel="claude", cycle_id=1,
    )
    return all([
        _assert(len(updated) == 0, "channel ミスマッチで消化されず"),
        _assert(p["observed_content"] is None, "device pending は残る"),
    ])


def test_try_observe_no_cross_contamination_same_source_action():
    """段階8 hotfix 回帰防止: 同 source_action 複数 pending で target が正しく選ばれる。

    バグ再現: smoke 2026-04-20 で観測した現象。
    - p_..._0023 (response_to_external, expected_channel=device, priority 3.0)
    - p_..._0011 (response_to_external, expected_channel=claude, priority 1.5)
    output_display(channel=claude) 実行時、match_pattern 判定では 0011 (claude) のみ
    候補のはずが、pending_observe 側が match_source_actions だけで絞って priority 順に
    0023 (device) を誤消化していた。target_id を渡すことで pinpoint 指定可能に。
    """
    print("== hotfix: 同 source_action 複数 pending で target ズレなし ==")
    state = _fresh_state()
    # 高 priority pending (device channel)
    device_pending = pending_add(
        state, source_action="response_to_external",
        expected_observation="device 応答", lag_kind="minutes",
        content="device 宛て", cycle_id=0, channel="device",
        match_pattern={"tool_name_any": ["output_display"], "channel_match": True},
    )
    # 低 priority pending (claude channel)
    claude_pending = pending_add(
        state, source_action="response_to_external",
        expected_observation="claude 応答", lag_kind="cycles",
        content="claude 宛て", cycle_id=0, channel="claude",
        match_pattern={"tool_name_any": ["output_display"], "channel_match": True},
    )
    # tool は claude channel で実行 → claude pending のみ消化されるべき
    updated = try_observe_all(
        state=state, tool_name="output_display",
        tool_args={"channel": "claude"},
        tool_result="送信完了 (claude): hi onee-tan",
        channel="claude", cycle_id=1,
    )
    return all([
        _assert(len(updated) == 1, "1 件消化"),
        _assert(updated[0]["id"] == claude_pending["id"],
                "claude pending が target (priority 低でも正しく指定)"),
        _assert(claude_pending["observed_content"] is not None,
                "claude pending が埋まる"),
        _assert(device_pending["observed_content"] is None,
                "device pending は未消化のまま (誤消化されない)"),
    ])


def test_pending_observe_target_id_direct():
    """pending_observe(target_id=...) 直接指定で特定 pending のみ消化される。"""
    print("== pending_observe: target_id 指定で特定 pending のみ消化 ==")
    from core.pending_unified import pending_observe
    state = _fresh_state()
    p1 = pending_add(
        state, source_action="reflect",
        expected_observation="intent_a", lag_kind="cycles",
        content="intent A", cycle_id=0, channel="self",
    )
    p2 = pending_add(
        state, source_action="reflect",
        expected_observation="intent_b", lag_kind="cycles",
        content="intent B", cycle_id=0, channel="self",
    )
    updated = pending_observe(
        state=state, observed_content="obs",
        channel="self", cycle_id=1,
        target_id=p2["id"],
    )
    return all([
        _assert(len(updated) == 1, "1 件消化"),
        _assert(updated[0]["id"] == p2["id"], "target_id の pending のみ"),
        _assert(p1["observed_content"] is None, "他 pending (p1) は未消化"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("_matches: tool_name_any hit", test_matches_tool_name_any_hit),
        ("_matches: tool_name_any miss", test_matches_tool_name_any_miss),
        ("_matches: tool_name_any None = any", test_matches_tool_name_any_none_means_any),
        ("_matches: channel_match hit", test_matches_channel_match_hit),
        ("_matches: channel_match mismatch", test_matches_channel_match_mismatch),
        ("_matches: similarity hit", test_matches_similarity_hit),
        ("_matches: similarity miss", test_matches_similarity_miss),
        ("_matches: 複数 AND", test_matches_all_fields_and),
        ("try_observe: tool_name 一致で消化", test_try_observe_tool_name_match),
        ("try_observe: tool_name 外れで skip", test_try_observe_tool_name_miss_skips),
        ("try_observe: match_pattern なしで skip", test_try_observe_no_match_pattern_skips),
        ("try_observe: priority 最高のみ消化", test_try_observe_priority_wins),
        ("try_observe: 消化済 pending を再消化しない", test_try_observe_already_observed_skipped),
        ("try_observe: channel_match ミスマッチで skip", test_try_observe_channel_mismatch_skips),
        ("hotfix: 同 source_action 複数 pending で target ズレなし", test_try_observe_no_cross_contamination_same_source_action),
        ("pending_observe: target_id 直接指定", test_pending_observe_target_id_direct),
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
