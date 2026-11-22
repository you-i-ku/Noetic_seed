"""HUMAN_FRAME_PLAN v0.4: 条件付き note、観察専用の検出、改名。"""
import ast
import copy
import json
import re
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.approval_callback import make_approval_callback, _format_preview
from core.runtime.approval_rules import make_policy_fn
from core.runtime.hooks import HookRunner, make_pre_tool_use_approval_check
from core.runtime.registry import ToolRegistry, ToolError
from core.runtime.tools import (
    register_all, ensure_approval_props,
    ensure_noetic_file_hints, ensure_noetic_bash_hint,
)
from core.runtime.legacy_bridge import register_legacy_bridge
from core.runtime.tools.noetic_ext import register_noetic_tools, NOETIC_TOOL_NAMES
from core.runtime.conversation import ConversationRuntime
from core.runtime.permissions import PermissionEnforcer, PermissionMode
from core.runtime.tool_schema import ToolSpec
from test_runtime_core import MockProvider
from test_prompt_assembly import _fresh_state, _sample_tools
from core.prompt_assembly import assemble_system_prompt
from test_outward_metrics import fixed_io  # noqa: F401
from test_failure_contract_foundation import persistence_io  # noqa: F401


REASONS = {"tool_intent": "自分の理由", "tool_expected_outcome": "自分の予想"}
RULES = [
    {"tools": ["bash"], "action": "approve"},
    {"tools": ["write_file"], "body_modify": True, "action": "approve"},
    {"tools": ["write_file", "read_file"], "action": "auto"},
    {"tools": ["bash"], "action": "auto"},  # 最初の一致を覆してはいけない
]


@pytest.mark.parametrize("tool,args,required,reason", [
    ("bash", {}, True, "rules[0]"),
    ("write_file", {"path": "core/test.py"}, True, "rules[1]"),
    ("write_file", {"path": "sandbox/test.txt"}, False, "rules[2]"),
    ("read_file", {"path": "core/test.py"}, False, "rules[2]"),
    ("unknown", {}, True, "default: approve"),
])
@pytest.mark.parametrize("note", [None, "", " \n\t", "説明あり"])
def test_policy_and_hook_share_decision(tmp_path, tool, args, required, reason, note):
    policy = make_policy_fn(RULES, workspace_root=tmp_path)
    inp = {**REASONS, **args, "note": note, "message": "本文では補わない"}
    decision, source = policy.evaluate(tool, inp)
    assert decision is required and reason in source
    assert policy(tool, inp) is required
    result = make_pre_tool_use_approval_check(policy_fn=policy)(tool, inp)
    assert result.denied is (required and not str(note or "").strip())
    if result.denied:
        assert reason in result.messages[0] and "note が空" in result.messages[0]
        if "rules[1]" in reason:
            assert "body_modify=True" in result.messages[0]


@pytest.mark.parametrize("tool", ["bash", "read_file", "write_file", "unknown"])
@pytest.mark.parametrize("missing", [None, "tool_intent", "tool_expected_outcome"])
def test_auto_all_never_evaluates_policy(tool, missing):
    policy = Mock(side_effect=AssertionError("policy called"))
    policy.evaluate.side_effect = AssertionError("reason evaluated")
    inp = dict(REASONS)
    if missing:
        inp[missing] = " "
    result = make_pre_tool_use_approval_check(
        auto_approve_all=True, policy_fn=policy)(tool, inp)
    assert result.denied is (missing is not None)
    request = Mock(side_effect=AssertionError("UI called"))
    cb = make_approval_callback(auto_approve_all=True, policy_fn=policy,
                                request_approval_fn=request)
    assert cb(tool, inp, []) is True
    policy.assert_not_called()
    policy.evaluate.assert_not_called()
    request.assert_not_called()


@pytest.mark.parametrize("mode", ["deny", "warn", "auto_fill"])
@pytest.mark.parametrize("required", [True, False])
def test_missing_policies_only_require_note_when_needed(mode, required):
    policy = make_policy_fn([], default_action="approve" if required else "auto")
    inp = {**REASONS, "message": "送信本文"}
    result = make_pre_tool_use_approval_check(mode, policy_fn=policy)("tool", inp)
    assert result.denied is (required and mode == "deny")
    if required and mode == "auto_fill":
        assert result.updated_input["note"].startswith("[auto_fill]")
        assert result.updated_input["message"] == "送信本文"
    else:
        assert result.updated_input is None
    assert bool(result.messages) is required
    assert "note" not in inp


def test_final_prompt_has_no_approval_audience():
    prompt = assemble_system_prompt(_fresh_state(), _sample_tools(), force_tool="read_file")
    assert "[強制実行指示]" in prompt and "2 フィールド" in prompt
    assert "tool_intent" in prompt and "tool_expected_outcome" in prompt
    assert prompt.count("note") == 1 and "note は自由に書ける欄" in prompt
    for text in ("確認相手", "口調", "message", "許可してください", "3 フィールド"):
        assert text not in prompt


@pytest.mark.parametrize("provider_name", ["anthropic", "openai", "claude_code"])
def test_all_registered_schemas_preserve_business_message(tmp_path, provider_name):
    from tools import TOOLS

    reg = ToolRegistry()
    register_all(reg, tmp_path)
    original = {s.name: copy.deepcopy(s.input_schema) for s in reg.list()}
    # main と同じ順序で実物を登録し、後付け説明も provider 境界まで通す。
    register_legacy_bridge(reg, TOOLS, skip_names=NOETIC_TOOL_NAMES)
    register_noetic_tools(reg, TOOLS)
    ensure_approval_props(reg)
    ensure_noetic_file_hints(reg)
    ensure_noetic_bash_hint(reg)
    assert ensure_approval_props(reg) == 0
    for name, meta in TOOLS.items():
        assert reg.get(name).handler is meta["func"]
    provider = MockProvider([])
    provider.name = provider_name
    runtime = ConversationRuntime(provider=provider, tool_registry=reg,
        permission_enforcer=PermissionEnforcer(PermissionMode.ALLOW))
    runtime._call_llm()
    assert len(provider.calls) == 1
    published = [s.get("function", s) for s in provider.calls[0].tools]
    assert {s["name"] for s in published} == set(reg.all_names())
    assert len(published) == len(reg.list())
    internal = {"mic_record", "camera_stream", "screen_peek", "secret_write", "reboot", "http_request"}
    for tool in published:
        spec = reg.get(tool["name"])
        schema = tool.get("input_schema", tool.get("parameters"))
        assert tool["description"] == spec.description
        assert schema == spec.input_schema
        _assert_no_approval_notice(tool["description"])
        _assert_no_approval_notice(schema)
        props, required = schema["properties"], schema["required"]
        assert {"tool_intent", "tool_expected_outcome"} <= set(required), spec.name
        assert props["note"] == {"type": "string", "description": "自由に書ける欄"}
        assert "note" not in required
        if spec.name in internal:
            assert "message" in props and "message" not in required
            assert props["message"]["description"] == "操作に添える説明 (省略可)"
        elif spec.name in original and spec.name not in TOOLS:
            old = original[spec.name]
            assert props.get("message") == old.get("properties", {}).get("message")
            assert ("message" in required) == ("message" in old.get("required", []))
        else:
            assert "message" not in props, spec.name
    # 共通 schema を使い回し、他の legacy tool に message が漏れる誤実装を検出。
    assert "message" not in reg.get("elyth_post").input_schema["properties"]
    for name in ("write_file", "edit_file"):
        description = reg.get(name).description
        assert "現在の作業ディレクトリ" in description
        assert "PLAN §" not in description and "profile 配下" not in description
        assert "profile 外" not in description
    name_description = reg.get("update_self").description + json.dumps(
        reg.get("update_self").input_schema, ensure_ascii=False)
    assert "変更不可" not in name_description and "不変" not in name_description
    from core.runtime.tools import task, ui
    with pytest.raises(ToolError, match="message is required"):
        task.task_update({"task_id": "missing", "note": "N"})
    with pytest.raises(ToolError, match="message is required"):
        ui.send_user_message({"note": "N"})


def _assert_no_approval_notice(value):
    """事実の『確認』や TestingPermission の機能説明は対象外。"""
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_approval_notice(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_approval_notice(child)
    elif isinstance(value, str):
        assert not re.search(
            r"承認|確認(?:が|を)?(?:要|必須|必要|求め|取る)|確認の(?:対象|ため)|"
            r"内部の確認|許可ダイアログ|依頼理由|"
            r"(?:requires?|needs?)\s+(?:approval|confirmation|permission)|"
            r"(?:approval|confirmation)\s+(?:required|needed)", value, re.I), value


@pytest.mark.parametrize("text", [
    "承認必須", "確認が要る", "確認が必要", "確認の対象", "内部の確認に添える説明",
    "MediaProjection 許可ダイアログあり", "requires approval", "confirmation required",
])
def test_approval_notice_check_detects_regressions(text):
    with pytest.raises(AssertionError):
        _assert_no_approval_notice({"properties": {"message": {"description": text}}})


@pytest.mark.parametrize("approved", [True, False])
def test_rejection_resubmission_then_confirmation(approved):
    policy = make_policy_fn(RULES)
    hooks = HookRunner()
    hooks.register_pre(make_pre_tool_use_approval_check(policy_fn=policy))
    handler = Mock(return_value="executed")
    request = Mock(return_value=approved)
    callback = make_approval_callback(pause_on_await=False, policy_fn=policy,
                                      request_approval_fn=request)
    reg = ToolRegistry()
    reg.register(ToolSpec(name="bash", description="", input_schema={"type": "object"},
                         required_permission=PermissionMode.DANGER_FULL_ACCESS, handler=handler))
    runtime = ConversationRuntime(provider=MockProvider([]), tool_registry=reg,
        permission_enforcer=PermissionEnforcer(PermissionMode.PROMPT),
        hook_runner=hooks, approval_callback=callback)
    first = {**REASONS, "message": "業務本文"}
    rec = runtime._execute_tool_use("first", "bash", first, push_session=False)
    assert "[REJECTED]" in rec.output and "rules[0]" in rec.output and "note が空" in rec.output
    handler.assert_not_called()
    request.assert_not_called()
    second = {**first, "note": "確認用自由欄"}
    rec = runtime._execute_tool_use("second", "bash", second, push_session=False)
    request.assert_called_once()
    preview = request.call_args.args[1]
    assert "'message': '業務本文'" in preview.split("② why:")[0]
    assert "'note'" not in preview.split("② why:")[0]
    assert preview.split("③ to you (= 協力者):\n")[1] == "   確認用自由欄"
    if approved:
        handler.assert_called_once_with(second)
        assert rec.output == "executed"
    else:
        handler.assert_not_called()
        assert "[REJECTED] approval denied" in rec.output


def test_message_is_not_a_fallback_for_note():
    preview = _format_preview("tool", {**REASONS, "message": "旧名で補わない"}, [])
    assert preview.endswith("   (空)")


def test_business_message_is_delivered_unchanged(monkeypatch):
    from core.runtime.tools import task, ui
    sender = Mock()
    monkeypatch.setitem(ui._ui_bridge, "send_user", sender)
    assert ui.send_user_message({"message": "送信本文", "note": "自由欄"}) == "Sent (normal): 送信本文"
    sender.assert_called_once_with("送信本文", [], "normal")
    update = Mock(return_value=True)
    monkeypatch.setattr(task._registry, "update", update)
    assert task.task_update({"task_id": "t", "message": "作業指示", "note": "自由欄"}) == "Sent message to task t"
    update.assert_called_once_with("t", "作業指示")


@pytest.mark.parametrize("auto_all,outer_count", [(False, 1), (True, 0)])
def test_internal_confirmation_remains_separate(monkeypatch, auto_all, outer_count):
    """外側の確認と内部確認を統合しない。内部の message は固有の説明。"""
    from tools import secret_tools
    inner = Mock(return_value=False)
    monkeypatch.setattr(secret_tools, "request_approval", inner)
    outer = Mock(return_value=True)
    policy = make_policy_fn([], default_action="approve")
    hook = make_pre_tool_use_approval_check(auto_approve_all=auto_all, policy_fn=policy)
    callback = make_approval_callback(pause_on_await=False, auto_approve_all=auto_all,
                                      policy_fn=policy, request_approval_fn=outer)
    inp = {**REASONS, "name": "example", "content": "test", "message": "内部用説明"}
    if not auto_all:
        inp["note"] = "外側自由欄"
    assert not hook("secret_write", inp).denied
    assert callback("secret_write", inp, [])
    with pytest.raises(ToolError, match="キャンセル"):
        secret_tools.secret_write(inp)
    assert outer.call_count == outer_count and inner.call_count == 1
    assert "内部用説明" in inner.call_args.args[1]
    assert "外側自由欄" not in inner.call_args.args[1]
    # 実ファイルへの書込なし。内部の拒否も ToolError として伝わる。


def test_main_wires_same_policy_before_hook():
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    calls = {name: [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == name]
             for name in ("make_policy_fn", "make_pre_tool_use_approval_check", "make_approval_callback")}
    policy, hook, cb = (calls[n][0] for n in calls)
    assert policy.lineno < hook.lineno < cb.lineno
    assert ast.unparse(next(k.value for k in policy.keywords if k.arg == "workspace_root")) == "BASE_DIR"
    for call in (hook, cb):
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert kw["policy_fn"] == "_policy_fn"
        assert kw["auto_approve_all"] == "_approval_cfg.get('auto_approve_all', False)"


def _identity_events(directory):
    path = directory / "metrics_events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["event_type"].startswith("identity_")] if path.exists() else []


@pytest.mark.parametrize("source", ["LLM①", "LLM②", "both", "neither"])
@pytest.mark.parametrize("term", ["AIアシスタント", "AI assistant", "AIAssistant"])
def test_identity_detection_fire_is_observation_only(monkeypatch, fixed_io, source, term):
    """Execute main's actual fire: wrong side, missing wire, memory injection and duplicates fail."""
    from core import metrics
    from test_outward_metrics import _fire_harness, _run, FakeProvider
    original = FakeProvider.stream
    def stream(provider, request):
        message = original(provider, request)
        message.text = f"引用: {term} / {term}" if source in ("LLM②", "both") else "本文"
        return message
    monkeypatch.setattr(FakeProvider, "stream", stream)
    proposal = "1. [考える] → reflect (pe2=40, pec=0.2)\n2. [応答] → output_display (pe2=70, pec=0.5)"
    if source in ("LLM①", "both"):
        proposal += f"\n引用: {term} / {term}"
    baseline = _fire_harness(monkeypatch, fixed_io, proposal=proposal)
    with monkeypatch.context() as patch:
        patch.setattr(metrics, "emit_identity_term_detected", lambda **kw: None)
        baseline_result = _run(baseline[0]["observed"])
    observed = _fire_harness(monkeypatch, fixed_io, proposal=proposal)
    assert _run(observed[0]["observed"]) == baseline_result
    assert observed[1] == baseline[1]  # includes raw_events / subjective_entries
    assert observed[2] == baseline[2]
    assert [(r.system_prompt, r.messages) for r in observed[3]] == [
        (r.system_prompt, r.messages) for r in baseline[3]]
    assert "[SYSTEM] 検出" not in repr(observed[1])
    events = _identity_events(fixed_io)
    if source == "neither":
        assert events == []
    else:
        assert len(events) == 1
        event = events[0]
        assert event["event_type"] == "identity_term_detected"
        assert (event["run_id"], event["cycle_id"]) == ("r", 4)
        assert event["detector_version"] == "substring-v1"
        sources = ["LLM①", "LLM②"] if source == "both" else [source]
        assert event["detections"] == [{"term": term, "source": side} for side in sources]


def test_identity_detection_emit_exception_does_not_stop_fire(monkeypatch, fixed_io):
    from core import metrics
    from test_outward_metrics import _fire_harness, _run
    observed = _fire_harness(monkeypatch, fixed_io)
    monkeypatch.setattr(metrics, "emit_identity_term_detected", Mock(side_effect=OSError("offline")))
    _run(observed[0]["observed"])
    assert observed[1]["cycle_id"] == 4 and observed[1]["raw_events"]


def _name_runtime():
    from types import SimpleNamespace
    from tools import TOOLS
    reg = ToolRegistry()
    register_noetic_tools(reg, TOOLS)
    return ConversationRuntime(SimpleNamespace(name="anthropic"), reg, HookRunner(),
                               PermissionEnforcer(PermissionMode.ALLOW))


def _live_name(st):
    state = st.load_state()
    state.update(run_id="disk-run", cycle_id=2)
    state["self"]["name"] = "before"
    st.bind_live_state(state)
    assert st.save_state(state)
    # Deliberately newer than disk: an extra load/copy loses the run/cycle and this field.
    state.update(run_id="live-run", cycle_id=8, live_only="keep")
    return state


def test_identity_name_change_uses_saved_live_state(persistence_io):
    from datetime import datetime
    from core.runtime.registry import current_tool_id
    st = persistence_io
    state = _live_name(st)
    rt = _name_runtime()
    for tool_id, value in (("rename-1", "after"), ("rename-2", "after"), ("rename-3", "third")):
        rec = rt._execute_tool_use(tool_id, "update_self", {"key": "name", "value": value,
                                                          "confidence": 0.2, "tool_id": "spoof"})
        assert not rec.is_error and rec.output == f"self[name] = {value}"
        assert current_tool_id.get() == ""
    events = _identity_events(st.MEMORY_DIR)
    assert len(events) == 2
    for event, old, new, tool_id in zip(events, ["before", "after"], ["after", "third"],
                                       ["rename-1", "rename-3"]):
        assert event["event_type"] == "identity_name_changed"
        assert (event["old_name"], event["new_name"], event["tool_id"]) == (old, new, tool_id)
        assert (event["run_id"], event["cycle_id"]) == ("live-run", 9)
        datetime.fromisoformat(event["time"])
    assert st.load_state() is state and state["self"]["name"] == "third"
    disk = json.loads(st.STATE_FILE.read_text(encoding="utf-8"))
    assert disk["self"]["name"] == "third" and disk["live_only"] == "keep"
    assert "name" not in state.get("_efe_self_confidence", {})
    assert "identity_name_changed" not in json.dumps(state)
    st.bind_live_state(None)
    assert st.load_state()["self"]["name"] == "third"


@pytest.mark.parametrize("value", ["", " \t", "AIアシスタント", "AI assistant", "AIAssistant"])
def test_identity_name_rejection_is_tool_error_without_event(persistence_io, value):
    st = persistence_io
    state = _live_name(st)
    before = copy.deepcopy(state)
    rec = _name_runtime()._execute_tool_use("reject", "update_self", {"key": "name", "value": value})
    assert rec.is_error and rec.error_kind == "tool_failure"
    assert not rec.output.startswith("tool execution error:")
    assert state == before and _identity_events(st.MEMORY_DIR) == []


@pytest.mark.parametrize("failure", ["false", "raise"])
def test_identity_name_failed_save_does_not_emit(persistence_io, monkeypatch, failure):
    from tools import builtin
    from core import metrics
    st = persistence_io
    state = _live_name(st)
    emit = Mock()
    monkeypatch.setattr(metrics, "emit_identity_name_changed", emit)
    def save(current):
        assert current is state
        assert not emit.called  # Detect emission before persistence.
        if failure == "raise":
            raise OSError("save failed")
        return False
    monkeypatch.setattr(builtin, "save_state", save)
    _name_runtime()._execute_tool_use("failed", "update_self", {"key": "name", "value": "after"})
    emit.assert_not_called()
    assert _identity_events(st.MEMORY_DIR) == []
    assert json.loads(st.STATE_FILE.read_text(encoding="utf-8"))["self"]["name"] == "before"


def test_identity_name_observation_exception_does_not_undo_save(persistence_io, monkeypatch):
    from core import metrics
    st = persistence_io
    _live_name(st)
    monkeypatch.setattr(metrics, "emit_identity_name_changed", Mock(side_effect=OSError("offline")))
    rec = _name_runtime()._execute_tool_use("saved", "update_self", {"key": "name", "value": "after"})
    assert not rec.is_error
    assert json.loads(st.STATE_FILE.read_text(encoding="utf-8"))["self"]["name"] == "after"


@pytest.mark.parametrize("kind", ["name", "term"])
@pytest.mark.parametrize("failure", [None, "append", "index"])
def test_identity_metrics_append_and_index_failures(monkeypatch, tmp_path, kind, failure):
    from core import metrics
    monkeypatch.setattr("core.config.MEMORY_DIR", tmp_path)
    if failure:
        method = "_atomic_append_jsonl" if failure == "append" else "_update_metrics_index"
        monkeypatch.setattr(metrics, method, Mock(side_effect=OSError(failure)))
    if kind == "name":
        result = metrics.emit_identity_name_changed(run_id="r", cycle_id=5,
                    old_name="before", new_name="after", tool_id="t")
    else:
        result = metrics.emit_identity_term_detected(run_id="r", cycle_id=5,
                    llm1_text="AIアシスタント AI assistant", llm2_text="AIAssistant AI assistant")
    assert result is (failure != "append")
    events = _identity_events(tmp_path)
    assert len(events) == (0 if failure == "append" else 1)
    if events and kind == "term":
        assert events[0]["detections"] == [
            {"term": "AIアシスタント", "source": "LLM①"},
            {"term": "AI assistant", "source": "LLM①"},
            {"term": "AI assistant", "source": "LLM②"},
            {"term": "AIAssistant", "source": "LLM②"},
        ]
    if failure is None:
        index = json.loads((tmp_path / metrics.INDEX_FILE_NAME).read_text(encoding="utf-8"))
        assert index[metrics.METRICS_FILE_NAME]["count"] == 1
