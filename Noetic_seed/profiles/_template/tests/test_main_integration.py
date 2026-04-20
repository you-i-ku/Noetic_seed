"""Phase 4 Step E-4: Step E 全コンポーネントの E2E 統合テスト。

main.py の _run_one_fire 相当のセットアップ (ConversationRuntime +
noetic_stub_tools + hooks + approval_callback + prompt_assembly) を
手動で組み合わせて、Noetic 固有の統合動作を検証する。

main() 本体は while True + ws_server + pressure tick 等を含むため直接
実行せず、fire cycle の中核だけを並行に再現して検証する。

検証スコープ:
  - Step A 承認 3 層 × Step B E 値評価 hook 連動
  - Step C UPS v2 pending 基盤 (add / observe / prune + retro E2)
  - Step D Session.push_observation の LLM コンテキスト投入
  - Step E-2a run_turn_with_forced_tool
  - Step E-2b noetic_stub_tools 登録
  - Step E-2c approval_callback の pause 連動 (pause_on_await=False)
  - Step E-3a retro_log_entry_id → 遡及 E2 自動発火
  - Step E-3d 相当の hook 自動発火で state["e_values"] / ledger / pending
    が一度に更新される

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_main_integration.py
"""
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.approval_callback import make_approval_callback
from core.pending_unified import pending_add, pending_observe, pending_prune
from core.providers.base import ApiRequest, AssistantMessage, BaseProvider, ToolUseBlock
from core.runtime.conversation import ConversationRuntime
from core.runtime.hooks import (
    HookRunner,
    make_post_tool_use_evaluation,
    make_post_tool_use_failure_logger,
    make_pre_tool_use_approval_check,
)
from core.runtime.tools.noetic_ext import register_noetic_tools
from core.runtime.permissions import PermissionEnforcer, PermissionMode
from core.runtime.registry import ToolRegistry


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


class _FakeProvider(BaseProvider):
    """テスト用 provider: 事前定義した AssistantMessage を返す。"""
    name = "openai_compat"

    def __init__(self, messages_to_return: list):
        super().__init__(model="fake", api_key="", base_url="")
        self._queue = list(messages_to_return)

    def stream(self, request: ApiRequest) -> AssistantMessage:
        if self._queue:
            return self._queue.pop(0)
        return AssistantMessage(text="(empty)", tool_uses=[], stop_reason="end_turn")


def _fake_tools(fail_tool_name: str = "") -> dict:
    """noetic_stub が期待する TOOLS dict を mock で構築。

    fail_tool_name を指定するとその tool が例外を投げる (failure hook 検証用)。
    """
    def _make_handler(name):
        def _h(args):
            if name == fail_tool_name:
                raise RuntimeError(f"simulated failure: {name}")
            content = args.get("content") or args.get("query") or "(no content)"
            return f"[{name} executed] {str(content)[:40]}"
        return _h
    # Noetic 固有 17 tool (noetic_ext 登録対象) を mock
    names = [
        "output_display", "wait", "reflect", "update_self",
        "search_memory", "memory_store", "memory_update", "memory_forget",
        "view_image", "listen_audio", "mic_record",
        "camera_stream", "camera_stream_stop", "screen_peek",
        "auth_profile_info", "secret_read", "secret_write",
    ]
    return {n: {"desc": n, "func": _make_handler(n)} for n in names}


def _fresh_state():
    return {
        "cycle_id": 5,
        "log": [],
        "pending": [],
        "action_ledger": [],
        "self": {},
        "files_read": [],
        "files_written": [],
        "energy": 50,
        "entropy": 0.65,
        "e_values": {},
        "tool_errors": [],
    }


def _mock_eval_llm(prompt, max_tokens=None, temperature=None):
    """PostToolUse hook が内部で呼ぶ eval_with_llm の mock response。
    E1=70, E2=60, E3=80, E4=90 を返す形式。"""
    return "E1=70\nE2=60\nE3=80\nE4=90"


def _build_runtime(state: dict, provider: _FakeProvider,
                   fail_tool_name: str = "") -> ConversationRuntime:
    """fire cycle 用の runtime + hooks + registry セットアップを構築。

    main() の Step E-2d 相当を手動で並行に組む (state は in-place mutate)。
    """
    registry = ToolRegistry()
    register_noetic_tools(registry, _fake_tools(fail_tool_name))

    hook_runner = HookRunner()
    hook_runner.register_pre(make_pre_tool_use_approval_check(
        missing_field_policy="deny"
    ))

    _hook_ctx = {"state_before": deepcopy(state)}

    base_post = make_post_tool_use_evaluation(
        state=state,
        get_state_before=lambda: _hook_ctx["state_before"],
        call_llm_fn=_mock_eval_llm,
        get_cycle_id=lambda: state.get("cycle_id", 0),
        get_recent_intents=lambda: [
            e.get("intent", "") for e in state.get("log", [])[-3:]
            if e.get("intent")
        ],
    )
    hook_runner.register_post(base_post)
    hook_runner.register_failure(make_post_tool_use_failure_logger(
        state=state,
        get_cycle_id=lambda: state.get("cycle_id", 0),
    ))

    approval_cb = make_approval_callback(
        pause_on_await=False,
        # request_approval_fn を常に True で stub (pause も触らない)
        request_approval_fn=lambda tn, prev, to: True,
        set_paused_fn=lambda v: None,
    )

    rt = ConversationRuntime(
        provider=provider,
        tool_registry=registry,
        hook_runner=hook_runner,
        permission_enforcer=PermissionEnforcer(mode=PermissionMode.PROMPT),
        max_iterations=1,
        approval_callback=approval_cb,
        max_tokens=4096,
        temperature=0.4,
    )
    return rt


def _forced_tool_msg(tool_name: str, tool_input: dict) -> AssistantMessage:
    return AssistantMessage(
        text="",
        tool_uses=[ToolUseBlock(id=f"call_{tool_name}",
                                name=tool_name,
                                input=tool_input)],
        stop_reason="tool_use",
    )


# ============================================================
# Test 1: 基本統合 — tool 実行 + hook で state["e_values"] 埋まる
# ============================================================

def test_basic_integration_fire_tool_eval():
    print("== 基本統合: tool 実行 → post hook → state['e_values'] 更新 ==")
    state = _fresh_state()
    provider = _FakeProvider([
        _forced_tool_msg("output_display", {
            "content": "おはよう",
            "tool_intent": "ゆうへの朝の挨拶",
            "tool_expected_outcome": "応答が届く",
            "message": "おはようと伝えます",
        }),
    ])
    rt = _build_runtime(state, provider)

    summary = rt.run_turn_with_forced_tool(
        forced_tool_name="output_display",
        user_input="朝の挨拶 intent",
    )

    return all([
        # max_iterations=1 + tool 実行 → for-else で finish_reason=max_iterations が仕様
        _assert(summary.finish_reason == "max_iterations",
                f"finish_reason=max_iterations (max_iter=1 で tool 実行時の正常値): "
                f"{summary.finish_reason}"),
        _assert(len(summary.tool_invocations) == 1, "1 tool 実行"),
        _assert(summary.tool_invocations[0].tool_name == "output_display",
                "output_display 実行"),
        _assert("e_values" in state and state["e_values"],
                "state['e_values'] が hook で埋まった"),
        _assert("%" in state["e_values"].get("e1", ""), "e1 に %"),
        _assert(len(state.get("action_ledger", [])) == 1,
                "action_ledger に記録"),
    ])


# ============================================================
# Test 2: 承認 3 層欠損 → pre hook deny → tool 未実行
# ============================================================

def test_approval_missing_denies_tool():
    print("== 3 層欠損: pre hook deny で tool が実行されない ==")
    state = _fresh_state()
    # tool_input が 3 層欠損 (content のみ)
    provider = _FakeProvider([
        _forced_tool_msg("output_display", {
            "content": "こんにちは",
            # tool_intent / tool_expected_outcome / message 全欠損
        }),
    ])
    rt = _build_runtime(state, provider)

    summary = rt.run_turn_with_forced_tool(
        forced_tool_name="output_display",
        user_input="deny テスト",
    )
    rec = summary.tool_invocations[-1] if summary.tool_invocations else None

    return all([
        _assert(rec is not None, "invocation 記録はされる (denied 状態で)"),
        _assert(rec.is_error is True, "is_error=True (pre hook deny)"),
        _assert("denied" in rec.output or "pre hook" in rec.output,
                f"output に deny 記録: {rec.output[:80]}"),
        _assert(state.get("e_values", {}) == {},
                "post hook 未発火 (tool 実行されていない)"),
        _assert(state.get("action_ledger", []) == [],
                "ledger 空 (実行されていない)"),
    ])


# ============================================================
# Test 3: tool 例外 → failure hook → state["tool_errors"] 記録
# ============================================================

def test_tool_failure_records_error():
    print("== tool 例外: failure hook で state['tool_errors'] 記録 ==")
    state = _fresh_state()
    provider = _FakeProvider([
        _forced_tool_msg("wait", {
            "tool_intent": "待機",
            "tool_expected_outcome": "何もしない",
            "message": "待ちます",
        }),
    ])
    rt = _build_runtime(state, provider, fail_tool_name="wait")

    summary = rt.run_turn_with_forced_tool(
        forced_tool_name="wait",
        user_input="fail テスト",
    )

    errors = state.get("tool_errors", [])
    err = errors[-1] if errors else {}
    return all([
        _assert(len(errors) == 1, f"tool_errors に 1 件: {len(errors)}"),
        _assert(err.get("tool") == "wait", "tool=wait"),
        _assert("simulated failure" in err.get("error", ""),
                "エラーメッセージ保存"),
    ])


# ============================================================
# Test 4: UPS v2 retro_log_entry_id → observation で遡及 E2
# ============================================================

def test_ups_retro_e2_flow():
    print("== UPS v2 retro: action 実行 → pending_add(retro) → observe で +40% ==")
    state = _fresh_state()
    # 過去 log に e2=40% の entry を配置 (retro 対象)
    log_entry_id = "sess_0001"
    state["log"] = [{
        "id": log_entry_id, "time": "09:00", "tool": "output_display",
        "intent": "発話", "e2": "40%",
    }]
    # output_display action 実行直後の pending (retro_log_entry_id 付き)
    pending_add(
        state,
        source_action="output_display",
        expected_observation="ゆうの返答",
        lag_kind="minutes",
        content="ゆうへの発話",
        cycle_id=5,
        channel="device",
        expiry_policy="time",
        ttl_cycles=20,
        retro_log_entry_id=log_entry_id,
    )

    # external_message 到着 (ゆうの返答) → pending_observe で遡及 E2 発火
    updated = pending_observe(
        state,
        observed_content="はーい、おはよう",
        channel="device",
        cycle_id=6,
        match_source_actions=["output_display"],
    )

    return all([
        _assert(len(updated) == 1, "1 件 observe"),
        _assert(state["log"][0]["e2"] == "80%",
                f"遡及 E2 +40%: {state['log'][0]['e2']}"),
        _assert(updated[0]["gap"] == 0.0, "pending gap=0"),
    ])


# ============================================================
# Test 5: pending_prune (time expiry + dynamic_n + protected)
# ============================================================

def test_pending_prune_mixed_policies():
    print("== pending_prune: time/dynamic_n/protected の同時淘汰 ==")
    state = _fresh_state()
    state["log"] = [{"cycle": i} for i in range(10)]  # dynamic_n = 3

    # protected: 常に残る
    pending_add(state, source_action="living_presence",
                expected_observation="外部声", lag_kind="unknown",
                content="ゆうの声", cycle_id=0, channel="device",
                expiry_policy="protected")
    # time expiry: ttl=5 で origin_cycle=0 → cycle 10 で削除対象
    pending_add(state, source_action="output_display",
                expected_observation="反応", lag_kind="minutes",
                content="古い発話", cycle_id=0, channel="device",
                expiry_policy="time", ttl_cycles=5,
                retro_log_entry_id="old")
    # dynamic_n 対象: 4 件 (gap 0.9, 0.5, 0.3, 0.1)
    for gap in [0.9, 0.5, 0.3, 0.1]:
        pending_add(state, source_action="reflection",
                    expected_observation="x", lag_kind="cycles",
                    content=f"ref g={gap}", cycle_id=0, channel="self",
                    initial_gap=gap, semantic_merge=True)

    dropped = pending_prune(state, current_cycle=10, dynamic_n=3)

    remaining_sources = [p.get("source_action") for p in state["pending"]]
    return all([
        _assert("living_presence" in remaining_sources, "protected 残る"),
        _assert("output_display" not in remaining_sources,
                "time expired 削除"),
        _assert(remaining_sources.count("reflection") == 3,
                f"dynamic_n=3 で上位 3 残る (実={remaining_sources.count('reflection')})"),
        _assert(dropped >= 2, f"削除 >= 2 (実={dropped})"),
    ])


# ============================================================
# Test 6: post hook が update_unresolved_intents 経由で UPS pending 追加
# ============================================================

def test_post_hook_adds_ups_unresolved_pending():
    print("== post hook: update_unresolved_intents 経由で UPS v2 pending 追加 ==")
    state = _fresh_state()
    provider = _FakeProvider([
        _forced_tool_msg("reflect", {
            "tool_intent": "最近の行動を振り返る",
            "tool_expected_outcome": "洞察が得られる",
            "message": "内省します",
        }),
    ])
    rt = _build_runtime(state, provider)

    rt.run_turn_with_forced_tool(
        forced_tool_name="reflect",
        user_input="reflection テスト",
    )

    ups_unresolved = [
        p for p in state.get("pending", [])
        if p.get("type") == "pending" and p.get("semantic_merge") is True
    ]
    entry = ups_unresolved[0] if ups_unresolved else {}
    return all([
        _assert(len(ups_unresolved) == 1,
                f"UPS semantic_merge pending 1 件: {len(ups_unresolved)}"),
        _assert(entry.get("source_action") == "reflect",
                f"source_action=reflect (tool 名由来)"),
        _assert(entry.get("observation_lag_kind") == "cycles",
                "lag_kind=cycles (unresolved 由来の default)"),
        _assert(entry.get("expected_channel") == "self",
                "expected_channel=self (内省 pending)"),
        _assert(abs(entry.get("gap", 0) - 0.2) < 0.01,
                "gap=0.2 (E3=80 由来)"),
    ])


# ============================================================
# Test 7: chain 実行 (複数 run_turn_with_forced_tool)
# ============================================================

def test_chain_execution_multiple_tools():
    print("== chain: 複数 tool を連続 run_turn で実行、各 post hook 発火 ==")
    state = _fresh_state()
    # 2 tool 連続: update_self → output_display
    provider = _FakeProvider([
        _forced_tool_msg("update_self", {
            "key": "mood", "value": "curious",
            "tool_intent": "感情更新", "tool_expected_outcome": "mood=curious",
            "message": "mood を更新",
        }),
        _forced_tool_msg("output_display", {
            "content": "curious な気分です",
            "tool_intent": "感情発話", "tool_expected_outcome": "ゆうに伝わる",
            "message": "感情を共有",
        }),
    ])
    rt = _build_runtime(state, provider)

    # 第 1 turn
    s1 = rt.run_turn_with_forced_tool(forced_tool_name="update_self",
                                        user_input="自己更新")
    # 第 2 turn (chain 次)
    s2 = rt.run_turn_with_forced_tool(forced_tool_name="output_display",
                                        user_input="発話")

    return all([
        _assert(len(s1.tool_invocations) == 1, "1 回目: 1 tool"),
        _assert(len(s2.tool_invocations) == 1, "2 回目: 1 tool"),
        _assert(len(state.get("action_ledger", [])) == 2,
                "ledger に 2 件 (各 post hook 発火)"),
        _assert(state["action_ledger"][0]["tool"] == "update_self",
                "1 件目 update_self"),
        _assert(state["action_ledger"][1]["tool"] == "output_display",
                "2 件目 output_display"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("基本統合: tool + post hook", test_basic_integration_fire_tool_eval),
        ("3 層欠損: pre hook deny", test_approval_missing_denies_tool),
        ("tool 例外: failure hook", test_tool_failure_records_error),
        ("UPS v2 retro E2 flow", test_ups_retro_e2_flow),
        ("pending_prune: mixed policies", test_pending_prune_mixed_policies),
        ("post hook: UPS unresolved 追加",
         test_post_hook_adds_ups_unresolved_pending),
        ("chain: 複数 tool 連続 + 各 hook", test_chain_execution_multiple_tools),
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
