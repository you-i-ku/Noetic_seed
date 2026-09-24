"""0b の観察、fire 出口、非介入の差分テスト。外部 I/O と LLM は固定する。

main の起動処理は実行せず、AST から fire 本体と観察関数をそのまま接続する。
候補パーサ・controller_select・prompt 組立・ConversationRuntime は実コード。
"""
import ast
import copy
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import controller as controller_module
from core import metrics
from core.info_gain import CYCLE_KEY
from core.parser import parse_candidates
from core.prompt import build_prompt_propose
from core.prompt_assembly import assemble_system_prompt
from core.providers.base import AssistantMessage, BaseProvider, ToolUseBlock
from core.runtime.conversation import ConversationRuntime
from core.runtime.hooks import HookRunner
from core.runtime.permissions import PermissionEnforcer, PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec

ROOT = Path(__file__).resolve().parent.parent
MAIN_SOURCE = (ROOT / "main.py").read_text(encoding="utf-8")
BASELINE_MAIN_SOURCE = globals().get("BASELINE_MAIN_SOURCE", MAIN_SOURCE)
BASELINE_CONTROLLER_SOURCE = globals().get("BASELINE_CONTROLLER_SOURCE")


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 24, 12, 0, 0, tzinfo=tz)


@pytest.fixture
def fixed_io(monkeypatch, tmp_path):
    monkeypatch.setattr("core.config.BASE_DIR", tmp_path)
    monkeypatch.setattr("core.config.MEMORY_DIR", tmp_path)
    monkeypatch.setattr(metrics, "get_code_version", lambda: "test-version")
    for path in ("core.memory.load_all_subjective_entries", "core.memory.load_all_memories",
                 "core.memory_links.list_links", "core.controller.load_all_subjective_entries",
                 "core.controller.load_all_memories", "core.controller.list_links"):
        monkeypatch.setattr(path, lambda *a, **kw: [])
    monkeypatch.setattr(controller_module.time, "time", lambda: 1000.0)
    monkeypatch.setattr(controller_module, "predict_next_conditioned_batch", lambda *a: None)
    monkeypatch.setattr(controller_module, "_detect_attractor_redundancy", lambda c: {
        "redundancy_pairs": [], "cluster_count": len(c), "redundant_tool_set": set(),
        "diversity_score": 1.0,
    })
    monkeypatch.setattr(controller_module, "compute_efe_components", lambda *a, **kw: {
        "G": 2.0 if "output_display" in kw["hypothetical_next"]["candidate"]["tools"] else 1.0,
    })
    monkeypatch.setattr("core.llm._reload_active_config", lambda: None)
    monkeypatch.setattr("core.auth.reload_secrets", lambda: None)
    monkeypatch.setattr("core.ws_server.get_stream_snapshot", lambda: ([], 0, False))
    monkeypatch.setattr("core.prompt.datetime", FixedDateTime)
    for path in ("core.world_model.update_basin_state", "core.transfer_entropy.maybe_te_maintenance",
                 "core.memory_links.prune_weak_links"):
        monkeypatch.setattr(path, lambda *a, **kw: None)
    return tmp_path


def _event(number, run="r", utterances=None, inputs=None):
    return {"event_type": "outward_attempt", "run_id": run, "attempt_id": f"{run}_{number}",
            "time": "2026-09-24 12:00:00", "utterances": utterances or [], "inputs": inputs or []}


@pytest.mark.parametrize("tool,args,expected", [
    ("output_display", None, "act"), ("elyth_mark_read", None, "act"),
    ("elyth_info", None, "sense"), ("x_like", None, "act"),
    ("WebSearch", None, "sense"), ("camera_stream_stop", None, "control"),
    ("camera_stream", None, "sense"), ("output_display_fake", None, "internal"),
    ("websearch", None, "internal"), ("reflect", None, "internal"),
    ("http_request", None, "undetermined"), ("http_request", {}, "sense"),
    ("http_request", {"method": " get "}, "sense"),
    ("http_request", {"method": "POST"}, "act"),
    ("view_image", None, "undetermined"), ("listen_audio", None, "undetermined"),
    ("view_image", {"path": "https://example.test/image.png"}, "sense"),
    ("listen_audio", {"path": "http://example.test/audio.wav"}, "sense"),
    ("view_image", {"path": "sandbox/image.png"}, "internal"),
    ("listen_audio", {"path": "sandbox/audio.wav"}, "internal"),
    ("listen_audio", {"path": "../outside.wav"}, "undetermined"),
])
def test_classification(tool, args, expected, fixed_io):
    """似た名前・既読/情報取得・引数依存を一律に分類する誤実装を検出する。"""
    assert metrics.classify_outward(tool, args) == expected


def test_chain_tail_and_selection_alignment(fixed_io):
    """先頭だけの分類、未判定のact扱い、G配列の順序違いを検出する。"""
    candidates = [{"tool": "reflect", "tools": ["reflect", "output_display"]},
                  {"tool": "elyth_info", "tools": ["elyth_info"]},
                  {"tool": "http_request", "tools": ["http_request", "view_image"]}]
    event = {}
    metrics.observe_outward_candidates(event, candidates)
    event.update(candidate_tools=[c["tools"] for c in candidates], g_values=[7, 2, 9], selected_idx=1)
    metrics.observe_outward_selection(event)
    assert event["proposed_act"] and event["proposed_sense"]
    assert event["proposed_undetermined"] == 2 and not event["selected_act"]
    assert event["best_outward_g_minus_selected_g"] == 5
    event["selected_idx"] = 0
    metrics.observe_outward_selection(event)
    assert event["selected_act"] and event["best_outward_g_minus_selected_g"] == 0


@pytest.mark.parametrize("output,is_error,status,connected", [
    ("送信キューに登録 (channel=device、受付時の接続 2 件、実際に届いたかは未確認): hi", False, "ok", 2),
    ("tool execution error: output_display: 接続している受け手が 0 件", True, "runtime_error", 0),
    ("[REJECTED] denied", True, "rejected", None),
    ("エラー: missing", False, "tool_error", None),
    ("Error: unavailable", False, "tool_error", None),
])
def test_execution_status(output, is_error, status, connected):
    """文字列エラーを成功発話にする、拒否をruntime_errorにする、接続数を到達数扱いする誤実装を検出する。"""
    result = metrics.build_outward_execution("output_display", {"channel": " device "},
                                             output, is_error, "display")
    assert result == {"tool": "output_display", "channel": "device", "status": status,
                      "connected_at_enqueue": connected, "category": "act"}


def test_emit_preserves_snapshot_and_counts_all_events(fixed_io):
    """outward emitでcycle snapshotを更新する、JSONLを上書きする、indexをcycle数とする誤実装を検出する。"""
    state = {"cycle_id": 1, "run_id": "r", "energy": 50, "entropy": 0.65,
             "self": {}, "pending": [], "predictor_confidence": {}, "action_ledger": []}
    metrics.emit_cycle_metrics(state, {}, {})
    assert CYCLE_KEY in state
    before = copy.deepcopy(state)
    event = _event(1)
    metrics.emit_outward_attempt(event)
    metrics.emit_outward_attempt(_event(2))
    assert state == before and state[CYCLE_KEY] == before[CYCLE_KEY]
    lines = [json.loads(s) for s in (fixed_io / metrics.METRICS_FILE_NAME).read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 and [e["attempt_id"] for e in lines[1:]] == ["r_1", "r_2"]
    assert json.loads((fixed_io / metrics.INDEX_FILE_NAME).read_text(encoding="utf-8"))[metrics.METRICS_FILE_NAME]["count"] == 3


class FakeProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, calls, requests, fail=False):
        super().__init__(model="test", api_key="")
        self.calls, self.requests, self.fail = calls, requests, fail

    def supports_tool_use(self):
        return True

    def stream(self, request):
        self.requests.append(copy.deepcopy(request))
        if self.fail:
            raise RuntimeError("runtime failure")
        return AssistantMessage(tool_uses=self.calls)


def _main_factory(env, source):
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    names = {"_record_outward_input", "_observe_outward", "_run_observed_fire", "_run_one_fire"}
    nodes = [copy.deepcopy(n) for n in main.body if
             (isinstance(n, ast.FunctionDef) and n.name in names) or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
              and t.id in {"_outward_seq", "_outward_fire_no", "_outward_pending_inputs"} for t in n.targets))]
    has_observer = any(isinstance(n, ast.FunctionDef) and n.name == "_run_observed_fire" for n in nodes)
    factory = ast.parse("def factory():\n    pressure = 20.0\n").body[0]
    factory.body += nodes
    factory.body += ast.parse("return {" + "'plain': _run_one_fire," +
                             ("'observed': _run_observed_fire, 'input': _record_outward_input" if has_observer
                              else "'observed': _run_one_fire, 'input': None") + "}").body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])),
                 str(ROOT / "main.py"), "exec"), env)
    return env["factory"]()


class SeqProvider(FakeProvider):
    """呼ばれるたびに次の段の tool_uses を返す (0d-1 の複数段テスト用)。"""

    def stream(self, request):
        self.requests.append(copy.deepcopy(request))
        uses = self.calls.pop(0) if self.calls else []
        if isinstance(uses, Exception):
            raise uses
        return AssistantMessage(tool_uses=uses)


def _fire_harness(monkeypatch, tmp_path, source=MAIN_SOURCE, mode="normal", selector=None,
                  stages=None, proposal=None, env_overrides=None, hook_runner=None):
    state = {"cycle_id": 3, "run_id": "r", "session_id": "s", "self": {},
             "energy": 50, "entropy": 0.65, "subjective_entries": [], "raw_events": [],
             "pending": [], "files_read": [], "files_written": [], "summaries": [],
             "action_ledger": [], "world_model": None, CYCLE_KEY: {"sentinel": 7}}
    calls = [ToolUseBlock(id="r1", name="output_display", input={"channel": "device"}),
             ToolUseBlock(id="r2", name="x_like", input={}),
             ToolUseBlock(id="r3", name="elyth_post", input={})]
    if mode == "not_invoked":
        calls = []
    registry = ToolRegistry()
    outputs = {"reflect": "reflected", "output_display": "送信キューに登録 (channel=device、受付時の接続 2 件、実際に届いたかは未確認): hi",
               "x_like": "liked", "elyth_post": "エラー: failed"}
    for name, output in outputs.items():
        registry.register(ToolSpec(name, name, {"type": "object"}, PermissionMode.READ_ONLY,
                                    lambda a, text=output: a.get("echo", text)))
    requests = []
    provider = (SeqProvider(copy.deepcopy(stages), requests) if stages is not None
                else FakeProvider(calls, requests, mode == "runtime_error"))
    runtime = ConversationRuntime(provider, registry,
                                  hook_runner or HookRunner(), PermissionEnforcer(PermissionMode.ALLOW))
    tools = {n: {"desc": n} for n in outputs}
    prompts, candidates_seen, selected_seen = [], [], []
    proposal = proposal or "1. [考える] → reflect (pe2=40, pec=0.2)\n2. [応答] → output_display (pe2=70, pec=0.5)"

    def llm(prompt, **kw):
        prompts.append(prompt)
        if mode == "llm1_error":
            raise RuntimeError("llm1 failure")
        return proposal

    def select(candidates, ctrl, current_state, **kw):
        selected = (selector or controller_module.controller_select)(candidates, ctrl, current_state, **kw)
        candidates_seen.append(copy.deepcopy(candidates))
        selected_seen.append(copy.deepcopy(selected))
        return selected

    cycle_emit = Mock(side_effect=lambda s, *a: s.__setitem__(CYCLE_KEY, {"cycle": s["cycle_id"]}))
    monkeypatch.setattr(metrics, "emit_cycle_metrics", cycle_emit)
    env = dict(state=state, copy=copy, json=json, re=re, datetime=FixedDateTime,
               time=SimpleNamespace(sleep=Mock()), BASE_DIR=tmp_path,
               _refresh_state=Mock(), _wm_log=Mock(), broadcast_log=Mock(), broadcast_state=Mock(),
               broadcast_self=Mock(), controller=lambda *a: {"allowed_tools": set(outputs), "tool_rank": {}},
               TOOLS=tools, LEVEL_TOOLS={}, llm_cfg={}, _rt_registry=registry, _runtime=runtime,
               build_prompt_propose=build_prompt_propose, assemble_system_prompt=assemble_system_prompt,
               call_llm=llm, prompt_budget={"completion_reserve": 1000}, append_debug_log=Mock(),
               parse_candidates=(lambda *a: []) if mode == "no_candidate" else parse_candidates,
               _intent_conditioned_scores=controller_module._intent_conditioned_scores,
               controller_select=select, _get_channel=lambda n: {"output_display": "display", "elyth_post": "elyth", "x_like": "x"}.get(n, "internal"),
               _hook_ctx={"state_before": {}, "evaluations": []}, _pending_observations=[],
               _extract_action_key=lambda n, a: n, cap_tool_result=lambda text: text,
               save_state=Mock(), calc_state_change_bonus=lambda *a: 0.0, _update_energy=Mock(),
               make_perspective=lambda **kw: kw,
               event_emitter=SimpleNamespace(fire_event=lambda s, e: s["raw_events"].append(e)),
               maybe_compress_log=Mock(), pending_prune=Mock(), load_pref=lambda: {})
    env.update(env_overrides or {})
    functions = _main_factory(env, source)
    return functions, state, prompts, requests, candidates_seen, selected_seen, cycle_emit


def _outward_lines(directory):
    """0b の観察行だけ。0d-1 の tool_invocation 行は数えない。"""
    return [e for e in (json.loads(s) for s in (directory / metrics.METRICS_FILE_NAME)
                        .read_text(encoding="utf-8").splitlines())
            if e["event_type"] == "outward_attempt"]


def _run(function):
    return function("test", False, {"post_fire_reset": 0.3}, 10, FixedDateTime.now())


@pytest.mark.parametrize("mode,end", [("normal", "completed"), ("llm1_error", "llm1_error"),
                                     ("not_invoked", "not_invoked"), ("runtime_error", "runtime_error"),
                                     ("no_candidate", "no_candidate")])
def test_fire_exit_once_and_full_invocations(mode, end, monkeypatch, fixed_io):
    """cycle末だけのemit、途中終了の欠落、最後のinvocationだけの観察、likeの発話扱いを検出する。"""
    funcs, state, *rest = _fire_harness(monkeypatch, fixed_io, mode=mode)
    if mode == "no_candidate":
        with pytest.raises(IndexError):
            _run(funcs["observed"])
    else:
        _run(funcs["observed"])
    lines = _outward_lines(fixed_io)
    assert len(lines) == 1 and lines[0]["attempt_id"] == "r_1"
    event = lines[0]
    assert event["end"] == end and event["cycle_id"] == 3
    if mode == "normal":
        assert [(r["tool"], r["status"]) for r in event["exec"]] == [
            ("output_display", "ok"), ("x_like", "ok"), ("elyth_post", "tool_error"), ("reflect", "not_invoked")]
        assert event["utterances"] == [{"seq": 1, "channel": "device", "tool": "output_display"}]
        assert event["exec"][0]["connected_at_enqueue"] == 2
    if mode in ("llm1_error", "not_invoked", "runtime_error"):
        assert rest[-1].call_count == 0 and state[CYCLE_KEY] == {"sentinel": 7}


def test_observation_does_not_change_behavior(monkeypatch, fixed_io):
    """観察追加が候補・G・選択・実prompt・state・cycle emitの回数を変える誤実装を検出する。"""
    baseline_selector = None
    if BASELINE_CONTROLLER_SOURCE is not None:
        node = next(n for n in ast.parse(BASELINE_CONTROLLER_SOURCE).body
                    if isinstance(n, ast.FunctionDef) and n.name == "controller_select")
        namespace = dict(vars(controller_module))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "old_controller", "exec"), namespace)
        baseline_selector = namespace["controller_select"]
    before = _fire_harness(monkeypatch, fixed_io, source=BASELINE_MAIN_SOURCE, selector=baseline_selector)
    result_before = _run(before[0]["plain"])
    after = _fire_harness(monkeypatch, fixed_io)
    result_after = _run(after[0]["observed"])
    assert result_before == result_after
    assert before[1] == after[1]
    assert before[2] == after[2] and before[2]
    assert [r.system_prompt for r in before[3]] == [r.system_prompt for r in after[3]]
    assert before[4:6] == after[4:6] and before[4]
    assert before[-1].call_count == after[-1].call_count == 1
    assert "candidate_tools" not in after[1]["last_selection_log"]


def test_emit_failure_isolated_and_input_buffer_retained(monkeypatch, fixed_io):
    """emit例外でfire結果を変える、未書出し入力を消す、毎回seqをリセットする誤実装を検出する。"""
    funcs, state, *_ = _fire_harness(monkeypatch, fixed_io, mode="not_invoked")
    funcs["input"]("device", "chat")
    funcs["input"]("claude", "mcp")
    real_emit = metrics.emit_outward_attempt
    monkeypatch.setattr(metrics, "emit_outward_attempt", Mock(side_effect=OSError("disk failure")))
    before = copy.deepcopy(state)
    assert _run(funcs["observed"]) is None
    assert state[CYCLE_KEY] == before[CYCLE_KEY]
    monkeypatch.setattr(metrics, "emit_outward_attempt", real_emit)
    _run(funcs["observed"])
    _run(funcs["observed"])
    events = _outward_lines(fixed_io)
    assert [e["attempt_id"] for e in events] == ["r_2", "r_3"]
    assert events[0]["inputs"] == [
        {"seq": 1, "channel": "device", "source": "chat", "cycle_at_record": 3},
        {"seq": 2, "channel": "claude", "source": "mcp", "cycle_at_record": 3}]
    assert events[1]["inputs"] == []


def test_input_and_utterance_sequence_across_fires(monkeypatch, fixed_io):
    """入力と発話を別seqにする、fireでリセットする、入力観察をstateへ書く誤実装を検出する。"""
    funcs, state, *_ = _fire_harness(monkeypatch, fixed_io)
    _run(funcs["observed"])
    before = copy.deepcopy(state)
    funcs["input"]("device", "chat")
    funcs["input"]("elyth", "mcp")
    assert state == before
    _run(funcs["observed"])
    events = _outward_lines(fixed_io)
    assert events[0]["utterances"][0]["seq"] == 1
    assert [(i["seq"], i["channel"], i["source"], i["cycle_at_record"])
            for i in events[1]["inputs"]] == [(2, "device", "chat", 4), (3, "elyth", "mcp", 4)]
    assert events[1]["utterances"][0]["seq"] == 4
    result = metrics.summarize_outward(events, k=1)["following_input"]
    assert [r["seq"] for r in result["matched"]] == [1]
    assert [r["seq"] for r in result["incomplete"]] == [4]


def test_observer_processing_failure_isolated(monkeypatch, fixed_io):
    """分類処理の例外でtool実行・state・fire戻り値を変える誤実装を検出する。"""
    baseline = _fire_harness(monkeypatch, fixed_io)
    result_before = _run(baseline[0]["plain"])
    monkeypatch.setattr(metrics, "build_outward_execution", Mock(side_effect=ValueError("observer failure")))
    observed = _fire_harness(monkeypatch, fixed_io)
    result_after = _run(observed[0]["observed"])
    assert result_before == result_after and baseline[1] == observed[1]
    events = _outward_lines(fixed_io)
    assert len(events) == 1 and events[0]["end"] == "completed"


def test_main_input_and_fire_wiring():
    """helper単体だけ動いてchat/MCP消化やmicro-loopに接続されていない誤実装を検出する。"""
    tree = ast.parse(MAIN_SOURCE)
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)]
    for target, source in (("chat_text", "chat"), ("_line", "mcp")):
        loop = next(n for n in loops if isinstance(n.target, ast.Name) and n.target.id == target)
        calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "_record_outward_input"]
        assert len(calls) == 1 and ast.literal_eval(calls[0].args[1]) == source
    micro = next(n for n in loops if isinstance(n.target, ast.Name) and n.target.id == "_micro_iter")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "_run_observed_fire" for n in ast.walk(micro))


@pytest.mark.parametrize("offset,channel,seq,matched", [
    (0, "device", 11, True), (0, "device", 9, False),
    (1, "device", 11, True), (2, "device", 11, True),
    (3, "device", 11, False), (1, "claude", 11, False), (1, "device", 10, False),
])
def test_following_input_boundaries(offset, channel, seq, matched):
    """同fireの前後・別channel・K境界・K+1をcycle_idだけで数える誤実装を検出する。"""
    events = [_event(i) for i in range(4)]
    events[0]["utterances"] = [{"seq": 10, "channel": "device", "tool": "output_display"}]
    events[offset]["inputs"] = [{"seq": seq, "channel": channel, "source": "mcp"}]
    result = metrics.summarize_outward(events, k=2)["following_input"]
    assert result["count"] == int(matched) and result["denominator"] == 1
    records = result["matched"] if matched else result["unmatched"]
    assert [(r["attempt_id"], r["seq"]) for r in records] == [("r_0", 10)]


def test_following_input_mixed_censoring_and_runs():
    """末尾を分母に含める、入力の重複割当を禁止する、likeや別runを数える誤実装を検出する。"""
    events = [_event(i) for i in range(4)]
    events[0]["utterances"] = [
        {"seq": 1, "channel": "device", "tool": "output_display"},
        {"seq": 2, "channel": "device", "tool": "output_display"},
        {"seq": 3, "channel": "x", "tool": "x_post"},
        {"seq": 4, "channel": "device", "tool": "x_like"}]
    events[1]["inputs"] = [{"seq": 10, "channel": "device", "source": "chat"}]
    events[3]["utterances"] = [{"seq": 11, "channel": "device", "tool": "output_display"}]
    events += [_event(0, run="other", inputs=[{"seq": 30, "channel": "x", "source": "mcp"}])]
    result = metrics.summarize_outward(events, k=2)["following_input"]
    assert (result["count"], result["denominator"], result["rate"]) == (2, 3, 2 / 3)
    assert [r["seq"] for r in result["matched"]] == [1, 2]
    assert [r["seq"] for r in result["unmatched"]] == [3]
    assert [r["seq"] for r in result["incomplete"]] == [11]
    assert metrics.summarize_outward([])["following_input"]["rate"] is None


def test_summary_stage_counts():
    """cycleイベントをfire分母に混ぜる、複数実行をfire件数にする誤実装を検出する。"""
    a, b = _event(1), _event(2)
    a.update(proposed_act=True, proposed_sense=True, proposed_undetermined=2, selected_act=False,
             exec=[{"status": "ok"}, {"status": "ok"}, {"status": "tool_error"}])
    b.update(proposed_act=True, selected_act=True, exec=[{"status": "not_invoked"}])
    result = metrics.summarize_outward([a, {"event_type": "cycle"}, b])
    assert result["attempts"] == 2
    assert result["proposed_act"]["rate"] == 1.0
    assert result["selected_act"]["rate"] == 0.5
    assert result["proposed_sense"]["count"] == 1
    assert result["proposed_undetermined"]["tools"] == 2
    assert result["execution"]["ok"] == {"count": 1, "denominator": 2, "rate": 0.5, "invocations": 2}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
