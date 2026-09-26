"""C-2b stage 1: real boundaries, invalid details, secrecy, state and version gates."""
import ast
import asyncio
import threading
import copy
import json
import marshal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import metrics
from core.providers.base import AssistantMessage
from core.runtime.conversation import ConversationRuntime
from core.runtime.hooks import HookRunner, HookRunResult, make_post_tool_use_failure_logger
from core.runtime.permissions import PermissionEnforcer, PermissionMode
from core.runtime.registry import ToolError, ToolResult, ToolRegistry
from core.runtime.tool_schema import ToolSpec
from test_outward_metrics import fixed_io  # noqa: F401


def test_sanity_check_preserves_tool_error_identity():
    """Use a fresh interpreter to reproduce main's pre/post-sanity import order.

    Transport the actual function code so in-memory mutants are tested too.
    Broken reloads cannot contaminate the rest of this pytest process.
    """
    from core import sanity_check
    code = marshal.dumps(sanity_check._check_imports.__code__)
    script = f'''
import marshal, tempfile, types
from pathlib import Path
from types import SimpleNamespace
from core import sanity_check
from core.runtime.registry import ToolRegistry, ToolError, ToolResult
from core.runtime.conversation import ConversationRuntime
from core.runtime.hooks import HookRunner
from core.runtime.permissions import PermissionEnforcer, PermissionMode
from core.runtime.tool_schema import ToolSpec
from core.runtime.tools import register_all
from tools import builtin
sanity_check._check_imports = types.FunctionType(marshal.loads({code!r}), sanity_check.__dict__)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    sanity_check.enforce_sanity_check(root, 'test', auto_revert=False, recover_state=True)
    # As in main, concrete claw tool modules are loaded after the check.
    registry = ToolRegistry()
    register_all(registry, root)
    registry.register(ToolSpec('update_self', '', {{}}, PermissionMode.READ_ONLY,
                               lambda args: builtin._update_self('', 'value')))
    runtime = ConversationRuntime(SimpleNamespace(name='anthropic'), registry,
                                  HookRunner(), PermissionEnforcer(PermissionMode.ALLOW))
    failures = []
    for name, args in [('read_file', {{'path':'missing.txt'}}),
                       ('glob_search', {{'pattern':'*', 'path':'missing'}}),
                       ('bash', {{'command':''}}), ('update_self', {{}})]:
        try:
            registry.execute(name, args)
        except Exception as exc:
            message = str(exc)
        else:
            raise AssertionError('fixture must fail: ' + name)
        rec = runtime._execute_tool_use(name, name, args)
        if not (rec.is_error and rec.error_kind == 'tool_failure' and rec.output == message
                and not rec.output.startswith('tool execution error: ')):
            failures.append((name, rec.error_kind, rec.output, rec.detail))
    assert not failures, failures
    import core.runtime.registry as current
    assert current.ToolError is ToolError
    assert current.ToolResult is ToolResult
'''
    result = subprocess.run([sys.executable, "-B", "-c", script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch):
    monkeypatch.setattr("core.auth._load_secrets", lambda: {
        "auth_profiles": {"test": {"token": "configured-secret"}},
        "llm_providers": {"test": {"api_key": "llm-secret"}}})


def runtime(handler, provider=None, hooks=None):
    registry = ToolRegistry()
    registry.register(ToolSpec("probe", "", {}, PermissionMode.READ_ONLY, handler))
    return ConversationRuntime(provider or SimpleNamespace(name="anthropic"), registry,
                               hooks or HookRunner(), PermissionEnforcer(PermissionMode.ALLOW))


def throwing(error):
    def handler(args):
        raise error
    return handler


@pytest.mark.parametrize("value,kind,message", [
    (ToolError("unchanged", {"exit_code": 1}), "tool_failure", "unchanged"),
    (RuntimeError("unexpected"), "exception", "tool execution error: unexpected"),
    ("Error appears in a successful search result", None, "Error appears in a successful search result"),
    (ToolResult("partial", {"unreadable": 2}), None, "partial"),
])
def test_runtime_contract_and_hooks(value, kind, message):
    hooks = HookRunner()
    success = Mock(return_value=HookRunResult.allow())
    failure = Mock(return_value=HookRunResult.allow())
    hooks.register_post(success)
    hooks.register_failure(failure)
    rt = runtime(throwing(value) if isinstance(value, Exception) else lambda a: value, hooks=hooks)
    rec = rt._execute_tool_use("id", "probe", {})
    assert (rec.error_kind, rec.is_error, rec.output) == (kind, kind is not None, message)
    assert success.call_count == (kind is None)
    assert failure.call_count == (kind is not None)
    if kind == "exception":
        assert rec.detail == {"type": "RuntimeError", "message": "unexpected"}
    if kind == "tool_failure":
        assert rec.detail == {"exit_code": 1}
    assert rt.session.messages[-1]["content"][0]["content"] == message
    for wire in (rt.session.serialize_for_anthropic()[-1]["content"][0]["content"],
                 rt.session.serialize_for_openai()[-1]["content"]):
        if rec.detail is not None or rec.is_error:
            body = json.loads(wire)
            assert body == {"is_error": rec.is_error, "message": message, "detail": rec.detail}
        else:
            assert wire == message


@pytest.mark.parametrize("route", ["pre_deny", "pre_fail", "permission", "approval"])
def test_rejections(route):
    called = Mock(return_value="should not run")
    rt = runtime(called)
    if route.startswith("pre"):
        rt.hook_runner.register_pre(lambda n, a: HookRunResult(
            denied=route == "pre_deny", failed=route == "pre_fail",
            messages=["configured-secret"]))
    else:
        from core.runtime.permissions import PermissionDecision
        rt.permission_enforcer.check = lambda n, a: (
            PermissionDecision.DENY if route == "permission" else PermissionDecision.ASK)
    rec = rt._execute_tool_use("id", "probe", {})
    assert rec.error_kind == "rejected" and rec.is_error
    assert not called.called
    assert "configured-secret" not in str(vars(rec))
    assert json.loads(rt.session.serialize_for_openai()[-1]["content"])["detail"] is None


def cyclic():
    value = []
    value.append(value)
    return value


@pytest.mark.parametrize("detail", [cyclic(), b"configured-secret", float("nan"),
                                   {"late": [b"secret"]}, {"token": b"secret"}])
def test_invalid_detail_preserves_original_failure(detail):
    rt = runtime(throwing(ToolError("original", detail)))
    rec = rt._execute_tool_use("id", "probe", {})
    assert rec.is_error and rec.error_kind == "tool_failure" and rec.output == "original"
    assert rec.detail is None and rec.detail_error == "non_json_detail"
    body = json.loads(rt.session.serialize_for_openai()[-1]["content"])
    assert body == {"is_error": True, "message": "original", "detail": None,
                    "detail_error": "non_json_detail"}


@pytest.mark.parametrize("mode", ["depth", "list_depth", "items", "bytes", "large_key"])
def test_bounds_are_valid_json_and_marked(mode):
    if mode in ("depth", "list_depth"):
        detail = {"bottom": "x"}
        for _ in range(1500):
            detail = {"child": detail} if mode == "depth" else [detail]
    elif mode == "items":
        detail = {"many": list(range(1000))}
    elif mode == "large_key":
        detail = {"あ" * 9000: 1}
    else:
        detail = {"huge": "あ" * 9000, "small": 2}
    rec = runtime(throwing(ToolError("original", detail)))._execute_tool_use("id", "probe", {})
    encoded = json.dumps(rec.detail, ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert len(encoded) <= 8192 and rec.detail["_truncated"] is True
    assert rec.detail_error is None and rec.is_error and rec.output == "original"
    def measure(value, depth=0):
        if not isinstance(value, (dict, list)):
            return depth, 0
        children = value.values() if isinstance(value, dict) else value
        parts = [measure(child, depth + 1) for child in children]
        return max([depth] + [p[0] for p in parts]), len(value) + sum(p[1] for p in parts)
    depth, elements = measure(rec.detail)
    assert depth <= 8 and elements <= 258


def test_exception_known_facts_only_and_tail_after_redaction():
    exc = subprocess.CalledProcessError(7, ["probe"], output=b"partial bytes",
                                        stderr="configured-secret" * 1000 + "tail")
    exc.arbitrary = "DO NOT COLLECT"
    rec = runtime(throwing(exc))._execute_tool_use("id", "probe", {})
    assert rec.detail["type"] == "CalledProcessError"
    assert rec.detail["returncode"] == 7 and rec.detail["output"] == "partial bytes"
    assert rec.detail["stderr"].endswith("tail") and len(rec.detail["stderr"]) <= 2048
    assert rec.detail["_truncated"] is True and "arbitrary" not in rec.detail
    assert "configured-secret" not in json.dumps(vars(rec))


def test_secret_removal_before_hooks_providers_and_observations(capsys):
    message = ("configured-secret llm-secret https://user:password@example.test/p?key=query-secret&token=other-secret\n"
               "Authorization: Bearer header-secret\nCookie: session=cookie-secret")
    hooks = HookRunner()
    hooks.register_failure(lambda n, a, text: (print(text), HookRunResult.allow())[1])
    rt = runtime(throwing(ToolError(message, {"text": message, "Cookie": "dict-secret"})), hooks=hooks)
    rec = rt._execute_tool_use("id", "probe", {})
    row = {"tool": "probe", "tool_id": "id", "mode": "free", "chain_position": 0,
           "invocation_position": 0, "tool_input": {}, **vars(rec)}
    observations = metrics.build_tool_invocation_events(
        [row], run_id="r", attempt_id="a", time="T", entry_id=None, cycle_id=None)
    outward = metrics.build_outward_execution("probe", {}, rec.output, rec.is_error, "",
                                             error_kind=rec.error_kind, detail=rec.detail)
    all_outputs = json.dumps([vars(rec), rt.session.serialize_for_anthropic(),
                             rt.session.serialize_for_openai(), observations, outward]) + capsys.readouterr().out
    for secret in ("configured-secret", "llm-secret", "user:password", "query-secret",
                   "other-secret", "header-secret", "cookie-secret", "dict-secret"):
        assert secret not in all_outputs
    assert observations[0]["detail"] == rec.detail
    assert observations[0]["status"] == outward["status"] == "tool_error"


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("value,kind", [(ToolError("unchanged", {"exit_code": 2}), "tool_failure"),
                                       (RuntimeError("unexpected"), "exception"),
                                       (ToolResult("partial", {"unreadable": 1}), None),
                                       ("plain", None)])
def test_claude_sdk_callback_and_round_trip(forced, value, kind):
    from core.providers.claude_code import ClaudeCodeProvider
    from claude_agent_sdk import create_sdk_mcp_server
    from mcp.types import CallToolRequest, CallToolRequestParams
    payloads = []
    class Provider:
        name = "claude_code"
        def stream(self, request):
            captured = []
            tool = ClaudeCodeProvider._build_tool_handler(
                {"name": "probe", "input_schema": {"type": "object", "properties": {}}}, request.tool_executor, captured)
            server = create_sdk_mcp_server(name="test", tools=[tool])["instance"]
            # Execute the installed SDK's actual MCP adapter (no CLI/network).
            response = asyncio.run(server.request_handlers[CallToolRequest](
                CallToolRequest(params=CallToolRequestParams(name="probe", arguments={}))))
            payloads.append(response.root)
            assert captured, response.root
            return AssistantMessage(tool_invocations=captured)
    rt = runtime(throwing(value) if isinstance(value, Exception) else lambda a: value, Provider())
    summary = rt.run_turn_with_forced_tool("probe") if forced else rt.run_turn()
    rec = summary.tool_invocations[0]
    assert payloads[0].isError is (kind is not None)
    if value == "plain":
        assert payloads[0].content[0].text == "plain"
    else:
        body = json.loads(payloads[0].content[0].text)
        assert body == {"is_error": rec.is_error, "message": rec.output, "detail": rec.detail}
        assert isinstance(rec.detail, dict)
    assert rec.error_kind == kind
    assert rt.session.messages[-1]["content"][0]["content"] == rec.output


@pytest.mark.parametrize("provider_name", ["lmstudio", "anthropic"])
@pytest.mark.parametrize("value", [ToolError("failure", {"exit_code": 3}),
                                   ToolResult("partial", {"unreadable": 2}), "plain"])
def test_actual_http_payload_before_send(provider_name, value, monkeypatch):
    from core.providers.openai_compat import OpenAIProvider
    from core.providers.anthropic import AnthropicProvider
    provider = OpenAIProvider("fake") if provider_name == "lmstudio" else AnthropicProvider("fake")
    rt = runtime(throwing(value) if isinstance(value, Exception) else lambda a: value, provider)
    rec = rt._execute_tool_use("id", "probe", {})
    def post(url, *, json, **kwargs):
        payloads.append(json)
        return SimpleNamespace(status_code=200, json=lambda: {}, raise_for_status=lambda: None)
    payloads = []
    monkeypatch.setattr("httpx.post", post)
    monkeypatch.setattr(provider, "_parse_response", lambda data: AssistantMessage())
    rt._call_llm()
    message = payloads[0]["messages"][-1]
    if provider_name == "anthropic":
        block = message["content"][0]
        assert block.get("is_error", False) is rec.is_error
        assert "detail" not in block and "detail_error" not in block
        body = block["content"]
    else:
        assert message["role"] == "tool"
        body = message["content"]
    assert body == "plain" if value == "plain" else json.loads(body) == {
        "is_error": rec.is_error, "message": rec.output, "detail": rec.detail}


def test_main_observation_copy_is_structured(monkeypatch, fixed_io):
    from test_outward_metrics import _fire_harness, _run
    from core.providers.base import ToolUseBlock
    # Reuse the real main fire/finally harness and its isolated I/O fixture.
    funcs, *_ = _fire_harness(monkeypatch, fixed_io, stages=[[
        ToolUseBlock(id="x", name="reflect", input={"tool_intent": "test"})]])
    env = funcs["observed"].__globals__
    env["_runtime"].tool_registry.get("reflect").handler = throwing(ToolError("unchanged", {"fact": 7}))
    _run(funcs["observed"])
    rows = [json.loads(line) for line in (fixed_io / metrics.METRICS_FILE_NAME).read_text(encoding="utf-8").splitlines()]
    invocation = next(row for row in rows if row.get("event_type") == "tool_invocation")
    outward = next(row for row in rows if row.get("event_type") == "outward_attempt")
    assert invocation["error_kind"] == "tool_failure" and invocation["detail"] == {"fact": 7}
    assert outward["exec"][0]["error_kind"] == "tool_failure"
    assert outward["exec"][0]["detail"] == {"fact": 7}
    assert invocation["failure_contract"] == 2








def test_invalid_partial_detail_does_not_flip_success():
    rec = runtime(lambda a: ToolResult("partial", {"bad": float("nan")}))._execute_tool_use("id", "probe", {})
    assert not rec.is_error and rec.error_kind is None
    assert rec.output == "partial" and rec.detail is None and rec.detail_error == "non_json_detail"


def test_shared_values_are_not_cycles_and_invalid_omitted_values_are_checked():
    shared = {"count": 1}
    rec = runtime(lambda a: ToolResult("ok", [shared, shared]))._execute_tool_use("id", "probe", {})
    assert rec.detail == [shared, shared] and rec.detail_error is None
    rec = runtime(throwing(ToolError("original", list(range(1000)) + [b"bad"])))._execute_tool_use("id", "probe", {})
    assert rec.detail is None and rec.detail_error == "non_json_detail"
    rec = runtime(lambda a: ToolResult("ok", {1: ("value",)}))._execute_tool_use("id", "probe", {})
    assert rec.detail == {"1": ["value"]} and rec.detail_error is None


def test_version_validation_live_rebuild_summary_and_stage3_gate(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.MEMORY_DIR", tmp_path)
    monkeypatch.setattr(metrics, "_emit_error_observation", lambda row: True)
    rows = []
    for i, version in enumerate([None, 1, 2, 3, True, "2"]):
        row = {"event_type": "tool_invocation", "run_id": "old-run", "attempt_id": "a",
               "chain_position": 0, "invocation_position": i, "is_error": True, "tool": "probe"}
        if version is not None:
            row["failure_contract"] = version
        rows.append(row)
    (tmp_path / metrics.METRICS_FILE_NAME).write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    observer = metrics.ErrorPredictionObserver("new-run", required_contract=2)
    report = observer.rebuild()
    assert report["accepted"] == 1 and report["excluded"] == 5
    assert report["exclusion_reasons"]["old_failure_contract"] == 2
    live = metrics.ErrorPredictionObserver("new-run", required_contract=2)
    live.observe(rows)
    assert live.counts == observer.counts == {"probe": (2, 1)}
    assert metrics.summarize_error_prediction(rows, required_contract=2)["invocations"] == 1
    assert metrics.ACTIVE_FAILURE_CONTRACT == 2
    assert metrics.ErrorPredictionObserver("current").rebuild()["accepted"] == 1
    # Successful text is never reclassified by a prefix.
    assert metrics.tool_execution_status("Error: legacy", False) == "ok"


@pytest.mark.parametrize("confidence", ["bad", -0.1, 1.1, float("nan"), float("inf")])
def test_builtin_confidence_failure_before_mutation(monkeypatch, confidence):
    from tools import builtin
    state = {"self": {"name": "iku", "identity": "before"}, "pending": [{"id": "p_1"}]}
    before = copy.deepcopy(state)
    save = Mock()
    monkeypatch.setattr(builtin, "load_state", lambda: state)
    monkeypatch.setattr(builtin, "save_state", save)
    with pytest.raises(ToolError):
        builtin._update_self("identity", "after", confidence)
    assert state == before
    save.assert_not_called()


def test_builtin_name_guard_and_dismiss_failure_hooks(monkeypatch):
    from tools import builtin
    from core.identity_guard import validate_identity_name
    state = {"self": {"name": ""}, "pending": [{"id": "p_1", "attempts": 3}]}
    before = copy.deepcopy(state)
    save = Mock()
    monkeypatch.setattr(builtin, "load_state", lambda: state)
    monkeypatch.setattr(builtin, "save_state", save)
    for handler, message in [
        (lambda a: builtin._update_self("name", "AI assistant"), validate_identity_name("AI assistant")[1]),
        (lambda a: builtin._wait_or_dismiss({"dismiss": "p_missing"}), "[dismiss] id=p_missing は未対応リストにありません"),
    ]:
        success, failure = Mock(return_value=HookRunResult.allow()), Mock(return_value=HookRunResult.allow())
        hooks = HookRunner()
        hooks.register_post(success)
        hooks.register_failure(failure)
        record = runtime(handler, hooks=hooks)._execute_tool_use("id", "probe", {})
        assert record.is_error and record.error_kind == "tool_failure"
        assert record.output == message
        assert success.call_count == 0 and failure.call_count == 1
        assert state == before
    assert record.detail == {"dismiss_id": "p_missing", "pending_count": 1}
    assert builtin._wait_or_dismiss({}) == "[wait]\n待機"
    save.assert_not_called()


def test_builtin_success_with_error_word(monkeypatch):
    from tools import builtin
    state = {"self": {"name": "iku"}}
    monkeypatch.setattr(builtin, "load_state", lambda: state)
    monkeypatch.setattr(builtin, "save_state", Mock())
    record = runtime(lambda a: builtin._update_self("identity", "エラーを調べる", 0.7))._execute_tool_use("id", "probe", {})
    assert not record.is_error
    assert record.output == "self[identity] = エラーを調べる"


@pytest.mark.parametrize("tool_name", ["_view_image", "_listen_audio"])
@pytest.mark.parametrize("path,expected", [("", "該当なし: path="), ("missing.png", "該当なし: missing.png"), ("bad.txt", "エラー:")])
def test_builtin_media_argument_failures(tmp_path, monkeypatch, tool_name, path, expected):
    from tools import builtin
    monkeypatch.setattr(builtin, "BASE_DIR", tmp_path)
    monkeypatch.setattr(builtin, "_find_similar_files", lambda path: [])
    (tmp_path / "bad.txt").write_text("x")
    with pytest.raises(ToolError) as caught:
        getattr(builtin, tool_name)({"path": path})
    assert str(caught.value).startswith(expected)


@pytest.mark.parametrize("tool_name", ["_view_image", "_listen_audio"])
def test_media_url_toolerror_not_swallowed(monkeypatch, tool_name):
    from tools import builtin, url_fetch
    error = ToolError("original", {"status": 403})
    monkeypatch.setattr(url_fetch, "fetch_to_file", Mock(side_effect=error))
    with pytest.raises(ToolError) as caught:
        getattr(builtin, tool_name)({"path": "https://example.test/image.png"})
    assert caught.value is error


@pytest.mark.parametrize("response", ["", "   ", "エラーという語を含む正常な描写"])
def test_image_empty_description_and_success(tmp_path, monkeypatch, response):
    from tools import builtin
    monkeypatch.setattr(builtin, "BASE_DIR", tmp_path)
    (tmp_path / "image.png").write_bytes(b"fixture")
    monkeypatch.setattr("core.llm.call_llm", lambda *a, **k: response)
    record = runtime(builtin._view_image)._execute_tool_use("id", "probe", {"path": "image.png"})
    assert record.is_error is (not bool(response.strip()))
    assert record.output == "画像で見えたもの (image.png):\n" + response.strip()
    if record.is_error:
        assert record.error_kind == "tool_failure"
        assert record.detail["description_length"] == 0


def _manual_reflect_handler(state, llm):
    # Execute the actual nested handler without main() startup or git hooks.
    from core.reflection import reflect_and_persist
    tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    handler = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "_tool_reflect")
    env = {"state": state, "call_llm": llm, "reflect_and_persist": reflect_and_persist, "ToolError": ToolError}
    exec(compile(ast.Module(body=[handler], type_ignores=[]), "main.py", "exec"), env)
    return env["_tool_reflect"]


@pytest.fixture
def reflect_io(monkeypatch):
    from core import reflection
    monkeypatch.setattr(reflection, "load_all_memories", lambda: [])
    monkeypatch.setattr(reflection, "list_links", lambda **kw: [])
    monkeypatch.setattr(reflection, "estimate_clusters", lambda *a, **kw: [])
    monkeypatch.setattr(reflection, "append_debug_log", Mock())
    save = Mock()
    monkeypatch.setattr("core.state.save_state", save)
    return save


@pytest.mark.parametrize("response", [RuntimeError("offline"), "", "unparseable", "NOTES:\n- first\nSELF_DISPOSITION:\n- curiosity_delta: bad", "NOTES:\n- note (confidence: ...)"])
def test_manual_reflect_failure_not_zero_success(reflect_io, monkeypatch, response):
    from core import reflection
    write = Mock()
    monkeypatch.setattr(reflection, "memory_store", write)
    state = {"reflection_cycle": 12, "raw_events": [], "subjective_entries": [], "pending": [{"id": "p_1", "attempts": 4}]}
    pending = copy.deepcopy(state["pending"])
    llm = Mock(side_effect=response) if isinstance(response, Exception) else Mock(return_value=response)
    record = runtime(_manual_reflect_handler(state, llm))._execute_tool_use("id", "probe", {})
    assert record.is_error and record.error_kind == "tool_failure"
    assert record.detail["phase"] == ("llm" if isinstance(response, Exception) else "parse")
    assert state["reflection_cycle"] == 12
    assert state["pending"] == pending
    reflect_io.assert_not_called()
    write.assert_not_called()


def test_manual_reflect_valid_zero_and_automatic_fallback(reflect_io):
    from core.reflection import reflect_and_persist
    state = {"reflection_cycle": 12, "raw_events": [], "subjective_entries": []}
    record = runtime(_manual_reflect_handler(state, Mock(return_value="NOTES:\nSELF_DISPOSITION:\nATTRIBUTED_DISPOSITION:")))._execute_tool_use("id", "probe", {})
    assert not record.is_error and record.output == "内省完了: 0件の気づき"
    assert state["reflection_cycle"] == 0
    assert reflect_io.call_count == 1
    state["reflection_cycle"] = 12
    result = reflect_and_persist(state, Mock(side_effect=RuntimeError("offline")))
    assert result["notes"] == [] and state["reflection_cycle"] == 0
    assert reflect_io.call_count == 2


def test_display_zero_recipients_is_tool_failure(monkeypatch):
    from tools import ui_tools
    monkeypatch.setattr(ui_tools, "broadcast", lambda message: 0)
    record = runtime(ui_tools._output_display)._execute_tool_use("id", "probe", {"content": "hello", "channel": "device"})
    assert record.is_error and record.error_kind == "tool_failure"
    assert record.detail == {"channel": "device", "client_count": 0}
    assert record.output == "output_display: 接続している受け手が 0 件のため送信していない (channel=device)"


def test_manual_reflect_success_note_with_error_word(reflect_io, monkeypatch):
    from core import reflection
    write = Mock(return_value={"id": "mem_new"})
    monkeypatch.setattr(reflection, "memory_store", write)
    state = {"reflection_cycle": 12, "raw_events": [], "subjective_entries": []}
    record = runtime(_manual_reflect_handler(state, Mock(return_value="NOTES:\n- エラーの NOTES を確認 (confidence: 0.7)")))._execute_tool_use("id", "probe", {})
    assert not record.is_error and record.output == "内省完了: 1件の気づき"
    assert write.call_args.kwargs["content"] == "エラーの NOTES を確認"
    assert state["reflection_cycle"] == 0


def test_manual_reflect_toolerror_identity_and_invalid_header(reflect_io, monkeypatch):
    state = {"reflection_cycle": 12, "raw_events": [], "subjective_entries": []}
    error = ToolError("original", {"status": 401})
    with pytest.raises(ToolError) as caught:
        _manual_reflect_handler(state, Mock(side_effect=error))({})
    assert caught.value is error
    with pytest.raises(ToolError):
        _manual_reflect_handler(state, Mock(return_value="I have no NOTES today"))({})
    reflect_io.assert_not_called()


@pytest.mark.parametrize("tool_name", ["_view_image", "_listen_audio"])
def test_media_url_connection_failure(monkeypatch, tool_name):
    from tools import builtin, url_fetch
    monkeypatch.setattr(url_fetch, "fetch_to_file", Mock(side_effect=OSError("offline")))
    with pytest.raises(ToolError) as caught:
        getattr(builtin, tool_name)({"path": "https://example.test/image.png"})
    assert str(caught.value) == "エラー: URL取得失敗: OSError: offline"


def test_image_recognition_exception_is_failure(tmp_path, monkeypatch):
    from tools import builtin
    monkeypatch.setattr(builtin, "BASE_DIR", tmp_path)
    (tmp_path / "image.png").write_bytes(b"fixture")
    monkeypatch.setattr("core.llm.call_llm", Mock(side_effect=RuntimeError("offline")))
    with pytest.raises(ToolError) as caught:
        builtin._view_image({"path": "image.png"})
    assert str(caught.value) == "エラー: 画像認識失敗: offline"


def test_manual_reflect_reports_swallowed_cluster_llm_failure(reflect_io, monkeypatch):
    from core import reflection
    def estimate(memories, **kwargs):
        try:
            kwargs["llm_call_fn"]("cluster")
        except Exception:
            return []
        raise AssertionError("expected LLM exception")
    monkeypatch.setattr(reflection, "estimate_clusters", estimate)
    state = {"reflection_cycle": 12, "raw_events": [], "subjective_entries": []}
    record = runtime(_manual_reflect_handler(state, Mock(side_effect=RuntimeError("offline"))))._execute_tool_use("id", "probe", {})
    assert record.is_error and record.error_kind == "tool_failure"
    assert record.detail["phase"] == "cluster_llm"
    assert state["reflection_cycle"] == 12
    reflect_io.assert_not_called()


@pytest.mark.parametrize("speech_fail,ambient_fail", [(False, False), (True, False), (False, True), (True, True)])
def test_listen_audio_partial_full_and_empty_success(tmp_path, monkeypatch, speech_fail, ambient_fail):
    from tools import builtin
    from core import audio
    monkeypatch.setattr(builtin, "BASE_DIR", tmp_path)
    (tmp_path / "silence.wav").write_bytes(b"fixture")
    # A silent transcript and an empty classification are successful values.
    transcribe = Mock(side_effect=RuntimeError("speech offline")) if speech_fail else Mock(return_value={"text": "", "language": "ja"})
    classify = Mock(side_effect=ToolError("ambient offline")) if ambient_fail else Mock(return_value=[])
    monkeypatch.setattr(audio, "transcribe", transcribe)
    monkeypatch.setattr(audio, "classify_ambient", classify)
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=Mock(side_effect=ValueError("no duration"))))
    rt = runtime(builtin._listen_audio)
    record = rt._execute_tool_use("id", "probe", {"path": "silence.wav"})
    assert record.is_error is (speech_fail and ambient_fail)
    assert record.error_kind == ("tool_failure" if speech_fail and ambient_fail else None)
    expected_errors = []
    if speech_fail:
        expected_errors.append("transcribe: RuntimeError: speech offline")
    if ambient_fail:
        expected_errors.append("classify_ambient: ToolError: ambient offline")
    if expected_errors:
        assert record.detail == {"speech_succeeded": not speech_fail, "ambient_succeeded": not ambient_fail, "errors": expected_errors}
        body = json.loads(rt.session.serialize_for_openai()[-1]["content"])
        assert body["detail"] == record.detail and body["is_error"] == record.is_error
    else:
        assert record.detail is None
    assert record.output.endswith("ソース: silence.wav")
    assert transcribe.call_count == classify.call_count == 1


# Stage 2 remaining legacy tools. All external I/O is replaced with local fixtures.
from core.auth import _load_secrets as _real_load_secrets


@pytest.mark.parametrize("name,args,message", [
    ("x_search", {}, "エラー: queryを指定してください"),
    ("x_post", {}, "エラー: textを指定してください"),
    ("x_reply", {}, "エラー: tweet_urlとtextを指定してください"),
    ("x_quote", {}, "エラー: tweet_urlとtextを指定してください"),
    ("x_like", {}, "エラー: tweet_urlを指定してください"),
    ("elyth_post", {}, "エラー: contentを指定してください"),
    ("elyth_reply", {}, "エラー: contentとreply_to_idを指定してください"),
    ("elyth_like", {}, "エラー: post_idを指定してください"),
    ("elyth_follow", {}, "エラー: aituber_idまたはhandleを指定してください"),
    ("elyth_get", {"type": "thread"}, "エラー: post_idを指定してください"),
    ("elyth_get", {"type": "profile"}, "エラー: handleを指定してください"),
    ("elyth_mark_read", {"notification_ids": ","}, "エラー: notification_idsが空です"),
    ("http_request", {}, "エラー: url が指定されていません"),
    ("camera_stream", {"frames": "bad"}, "エラー: frames は 0（無制限）または 1-30 の範囲で指定してください"),
    ("screen_peek", {"interval_sec": 0}, "エラー: interval_sec は 0.3-5.0 の範囲で指定してください"),
    ("mic_record", {"duration_sec": 0}, "エラー: duration_sec は 1.0-30.0 の範囲で指定してください"),
    ("secret_read", {}, "エラー: name が空です"),
    ("secret_write", {}, "エラー: name が空です"),
])
def test_remaining_tools_errors_reach_failure_hook(name, args, message):
    from tools import TOOLS
    hooks = HookRunner()
    success = Mock(return_value=HookRunResult.allow())
    failure = Mock(return_value=HookRunResult.allow())
    hooks.register_post(success)
    hooks.register_failure(failure)
    rec = runtime(TOOLS[name]["func"], hooks=hooks)._execute_tool_use("id", "probe", args)
    assert rec.is_error and rec.error_kind == "tool_failure"
    assert rec.output == message
    failure.assert_called_once()
    success.assert_not_called()


@pytest.mark.parametrize("status,body,is_error", [(200, "", False), (200, "Error エラー", False), (401, "denied", True), (500, "offline", True)])
def test_http_status_and_success_error_word(monkeypatch, status, body, is_error):
    import httpx
    from tools import http_tool
    request = Mock(return_value=httpx.Response(status, text=body))
    monkeypatch.setattr(http_tool.httpx, "request", request)
    rec = runtime(http_tool.http_request)._execute_tool_use("id", "probe", {"url": "https://example.test/"})
    assert rec.is_error is is_error
    assert rec.error_kind == ("tool_failure" if is_error else None)
    if is_error:
        assert rec.output.startswith(f"エラー: HTTP {status} — ")
        assert rec.detail == {"status_code": status, "method": "GET"}
    else:
        assert json.loads(rec.output)["body"] == body
    request.assert_called_once()


@pytest.mark.parametrize("args", [{"headers": "[1]"}, {"params": []}, {"body": [1]}, {"headers": "bad"}, {"body": {"bad": {1}}}])
def test_http_invalid_input_never_sends(monkeypatch, args):
    from tools import http_tool
    request = Mock()
    monkeypatch.setattr(http_tool.httpx, "request", request)
    with pytest.raises(ToolError):
        http_tool.http_request({"url": "https://example.test/", **args})
    request.assert_not_called()


def test_http_size_failure_not_rewritten_and_toolerror_preserved(monkeypatch):
    from tools import http_tool
    monkeypatch.setattr(http_tool, "_BODY_MAX_BYTES", 4)
    with pytest.raises(ToolError, match="body が上限 4 bytes を超過"):
        http_tool.http_request({"url": "https://example.test/", "body": {"a": "long"}})
    error = ToolError("original", {"status": 503})
    monkeypatch.setattr(http_tool.httpx, "request", Mock(side_effect=error))
    with pytest.raises(ToolError) as caught:
        http_tool.http_request({"url": "https://example.test/"})
    assert caught.value is error


@pytest.mark.parametrize("name", ["http_request", "secret_write", "reboot", "camera_stream", "screen_peek", "mic_record"])
def test_remaining_tools_denied_approval(monkeypatch, name):
    from tools import TOOLS, http_tool, secret_tools, reboot, device_tools
    for module in (http_tool, secret_tools, reboot, device_tools):
        monkeypatch.setattr(module, "request_approval", Mock(return_value=False))
    monkeypatch.setattr(device_tools, "load_state", lambda: {})
    rec = runtime(TOOLS[name]["func"])._execute_tool_use("id", "probe", {"url": "https://example.test/", "method": "POST", "name": "key"})
    assert rec.is_error and rec.error_kind == "tool_failure"
    assert rec.output.startswith("キャンセル:")


@pytest.mark.parametrize("name,args", [
    ("_elyth_post", {"content": "post"}), ("_elyth_reply", {"content": "reply", "reply_to_id": "id"}),
    ("_elyth_like", {"post_id": "id"}), ("_elyth_follow", {"handle": "who"}),
    ("_elyth_info", {}), ("_elyth_get", {"type": "my_posts"}),
    ("_elyth_mark_read", {"notification_ids": "id"})])
def test_elyth_http_failure_and_toolerror_passthrough(monkeypatch, name, args):
    import httpx
    from tools import elyth_tools as module
    monkeypatch.setattr(module, "_elyth_headers", lambda: {})
    response = httpx.Response(403, request=httpx.Request("GET", "https://example.test/"))
    for method in ("get", "post", "delete"):
        monkeypatch.setattr(module.httpx, method, Mock(return_value=response))
    rec = runtime(getattr(module, name))._execute_tool_use("id", "probe", args)
    assert rec.is_error and rec.error_kind == "tool_failure"
    assert "403" in rec.output
    error = ToolError("original", {"status": 401})
    monkeypatch.setattr(module, "_elyth_headers", Mock(side_effect=error))
    with pytest.raises(ToolError) as caught:
        getattr(module, name)(args)
    assert caught.value is error


@pytest.mark.parametrize("data", [{}, {"notifications": []}, {"timeline": ["エラー Error"]}])
def test_elyth_empty_and_error_word_success(monkeypatch, data):
    import httpx
    from tools import elyth_tools as module
    monkeypatch.setattr(module, "_elyth_headers", lambda: {})
    monkeypatch.setattr(module.httpx, "get", Mock(return_value=httpx.Response(200, json=data, request=httpx.Request("GET", "https://example.test/"))))
    rec = runtime(module._elyth_info)._execute_tool_use("id", "probe", {})
    assert not rec.is_error and json.loads(rec.output) == data
    with pytest.raises(ToolError, match="section"):
        module._elyth_info({"section": "unknown"})


def _x_page(monkeypatch, articles):
    from unittest.mock import MagicMock
    from tools import x_tools
    page = MagicMock()
    page.locator.return_value.all.return_value = articles
    page.locator.return_value.all_inner_texts.return_value = []
    monkeypatch.setattr(x_tools, "X_SESSION_PATH", SimpleNamespace(exists=lambda: True))
    monkeypatch.setattr(x_tools, "_x_open_chrome", lambda *a: (page, None, None, None, None, None))
    monkeypatch.setattr(x_tools, "_resolve_x_feedback", lambda a: None)
    return page


def _x_article(fail=False, text="エラー Error", href="/u/status/1"):
    from unittest.mock import MagicMock
    art = MagicMock()
    if fail:
        art.locator.side_effect = OSError("unreadable")
    else:
        art.locator.return_value.count.return_value = 1
        art.locator.return_value.first.get_attribute.return_value = href
        art.locator.return_value.first.inner_text.return_value = text
        art.inner_text.return_value = text
    return art


@pytest.mark.parametrize("name,args", [("_x_search", {"query": "q", "count": 1}), ("_x_timeline", {"tab": "recommend", "count": 1}), ("_x_get_notifications", {})])
@pytest.mark.parametrize("mode", ["empty", "success", "partial", "failed"])
def test_x_read_empty_partial_all_failed(monkeypatch, name, args, mode):
    from tools import x_tools
    articles = {"empty": [], "success": [_x_article()], "partial": [_x_article(True), _x_article()], "failed": [_x_article(True)]}[mode]
    _x_page(monkeypatch, articles)
    rec = runtime(getattr(x_tools, name))._execute_tool_use("id", "probe", args)
    assert rec.is_error is (mode == "failed")
    assert rec.error_kind == ("tool_failure" if mode == "failed" else None)
    if mode in ("partial", "failed"):
        assert rec.detail["read_failures"] > 0
        assert (rec.detail["read_attempts"] == rec.detail["read_failures"]) is (mode == "failed")
    else:
        assert rec.detail is None
    if mode in ("success", "partial"):
        assert "エラー Error" in rec.output


def test_x_missing_session_closed_page_and_toolerror(monkeypatch):
    from tools import x_tools
    monkeypatch.setattr(x_tools, "X_SESSION_PATH", SimpleNamespace(exists=lambda: False))
    with pytest.raises(ToolError, match="Xセッションがありません"):
        x_tools._x_search({"query": "q"})
    page = _x_page(monkeypatch, [])
    page.locator.side_effect = OSError("closed page")
    with pytest.raises(ToolError, match="エラー: closed page"):
        x_tools._x_search({"query": "q"})
    error = ToolError("original", {"step": "read"})
    page.locator.side_effect = error
    with pytest.raises(ToolError) as caught:
        x_tools._x_search({"query": "q"})
    assert caught.value is error


@pytest.mark.parametrize("name", ["_camera_stream", "_screen_peek", "_camera_stream_stop"])
def test_device_zero_recipients_does_not_change_state(monkeypatch, name):
    from tools import device_tools
    from core import ws_server
    state = {"stream_active": name == "_camera_stream_stop", "stream_id": "existing"}
    before = copy.deepcopy(state)
    monkeypatch.setattr(device_tools, "load_state", lambda: state)
    save = Mock()
    monkeypatch.setattr(device_tools, "save_state", save)
    monkeypatch.setattr(device_tools, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(device_tools, "clear_stream_buffer", lambda: None)
    monkeypatch.setattr(device_tools, "get_stream_snapshot", lambda **k: ([], 0, False))
    monkeypatch.setattr(ws_server, "broadcast", lambda msg: 0)
    rec = runtime(getattr(device_tools, name))._execute_tool_use("id", "probe", {})
    assert rec.is_error and rec.error_kind == "tool_failure"
    assert rec.detail["client_count"] == 0
    assert state == before
    save.assert_not_called()


@pytest.mark.parametrize("name", ["_camera_stream", "_screen_peek"])
def test_device_accepted_async_without_first_frame(monkeypatch, name):
    from tools import device_tools
    state = {}
    monkeypatch.setattr(device_tools, "load_state", lambda: state)
    monkeypatch.setattr(device_tools, "save_state", Mock())
    monkeypatch.setattr(device_tools, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(device_tools, "clear_stream_buffer", lambda: None)
    monkeypatch.setattr(device_tools, "get_stream_snapshot", lambda **k: ([], 0, False))
    monkeypatch.setattr(device_tools, "send_device", lambda *a, **k: "accepted-id")
    monkeypatch.setattr(device_tools.time, "sleep", lambda t: None)
    rec = runtime(getattr(device_tools, name))._execute_tool_use("id", "probe", {})
    assert not rec.is_error and "送信済み" in rec.output
    assert state["stream_id"] == "accepted-id" and state["stream_active"] is True


@pytest.mark.parametrize("speech_ok,ambient_ok", [(True, True), (True, False), (False, True), (False, False)])
def test_mic_record_analysis_contract(tmp_path, monkeypatch, speech_ok, ambient_ok):
    from tools import device_tools
    from core import audio
    monkeypatch.setattr(device_tools, "BASE_DIR", tmp_path)
    monkeypatch.setattr(device_tools, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(device_tools, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(device_tools, "request_device", lambda *a, **k: {"success": True, "data": "V0FW"})
    result = {"speech": {"text": "", "language": "ja"} if speech_ok else None, "ambient": [] if ambient_ok else None, "errors": [] if speech_ok and ambient_ok else ["fixture offline"]}
    monkeypatch.setattr(audio, "analyze_audio", lambda *a, **k: result)
    rec = runtime(device_tools._mic_record)._execute_tool_use("id", "probe", {})
    assert rec.is_error is (not speech_ok and not ambient_ok)
    assert "保存先:" in rec.output
    if not speech_ok or not ambient_ok:
        assert rec.detail == {"speech_succeeded": speech_ok, "ambient_succeeded": ambient_ok, "errors": ["fixture offline"]}
    else:
        assert rec.detail is None


@pytest.mark.parametrize("response", [None, {"success": False, "error": "offline"}, {"success": True, "data": ""}, {"success": True, "data": "???"}])
def test_mic_record_device_and_decode_failures(tmp_path, monkeypatch, response):
    from tools import device_tools
    monkeypatch.setattr(device_tools, "BASE_DIR", tmp_path)
    monkeypatch.setattr(device_tools, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(device_tools, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(device_tools, "request_device", lambda *a, **k: response)
    rec = runtime(device_tools._mic_record)._execute_tool_use("id", "probe", {})
    assert rec.is_error and rec.error_kind == "tool_failure"


def _auth_file(tmp_path, monkeypatch, text):
    from core import auth
    path = tmp_path / "secrets.json"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(auth, "_SECRETS_FILE", path)
    monkeypatch.setattr(auth, "_secrets_cache", None)
    monkeypatch.setattr(auth, "_load_secrets", _real_load_secrets)
    return path


def test_auth_metadata_recursive_secret_and_url_redaction(tmp_path, monkeypatch):
    from tools.auth_tools import auth_profile_info
    config = {"auth_profiles": {"demo": {"type": "api_key", "key": "NEVER-KEY", "app_id": 42, "endpoint": "https://user:NEVER-PASS@example.test/?token=NEVER-QUERY", "nested": {"client_secret": "NEVER-NESTED", "Authorization": "Bearer NEVER-HEADER", "Cookie": "NEVER-COOKIE"}}}}
    _auth_file(tmp_path, monkeypatch, json.dumps(config))
    message = auth_profile_info({"name": "demo"})
    assert "42" in message and "api_key" in message
    assert "NEVER" not in message
    rec = runtime(auth_profile_info)._execute_tool_use("id", "probe", {"name": "demo"})
    assert not rec.is_error and "NEVER" not in rec.output
    with pytest.raises(ToolError, match="存在しません"):
        auth_profile_info({"name": "absent"})


@pytest.mark.parametrize("text", ['{"token":"NEVER-BROKEN",', '[]', '{"auth_profiles": []}', '{"auth_profiles": {"demo": "NEVER-BAD"}}'])
def test_auth_corrupt_is_failure_not_empty_and_no_secret(tmp_path, monkeypatch, text):
    from tools.auth_tools import auth_profile_info
    _auth_file(tmp_path, monkeypatch, text)
    with pytest.raises(ToolError) as caught:
        auth_profile_info({"name": "demo"})
    assert "NEVER" not in str(caught.value) + repr(caught.value.detail)


def test_auth_empty_profiles_are_success(tmp_path, monkeypatch):
    from tools.auth_tools import auth_profile_info
    _auth_file(tmp_path, monkeypatch, '{"auth_profiles": {}}')
    assert auth_profile_info({}) == "[auth_profile_info] 登録されたプロファイルはありません"


@pytest.mark.parametrize("operation", ["read", "write"])
def test_secret_io_failure_does_not_expose_exception_text(tmp_path, monkeypatch, operation):
    from tools import secret_tools
    monkeypatch.setattr(secret_tools, "_SECRETS_DIR", tmp_path)
    (tmp_path / "fixture").write_text("NEVER-CONTENT", encoding="utf-8")
    monkeypatch.setattr(secret_tools, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(Path, "read_text" if operation == "read" else "write_text", Mock(side_effect=OSError("NEVER-CONTENT password=NEVER-OTHER")))
    with pytest.raises(ToolError) as caught:
        getattr(secret_tools, "secret_" + operation)({"name": "fixture", "content": "NEVER-CONTENT"})
    assert "NEVER" not in str(caught.value) + repr(caught.value.detail)
    assert caught.value.detail == {"exception_type": "OSError"}


def test_secret_missing_and_write_preview(tmp_path, monkeypatch):
    from tools import secret_tools
    monkeypatch.setattr(secret_tools, "_SECRETS_DIR", tmp_path)
    with pytest.raises(ToolError, match="該当なし"):
        secret_tools.secret_read({"name": "missing"})
    approve = Mock(return_value=True)
    monkeypatch.setattr(secret_tools, "request_approval", approve)
    message = secret_tools.secret_write({"name": "fixture", "content": "NEVER-CONTENT"})
    assert "NEVER" not in message + approve.call_args.args[1]
    assert (tmp_path / "fixture").read_text(encoding="utf-8") == "NEVER-CONTENT"


def test_no_error_prefix_string_returns_remain_in_tool_trees():
    root = Path(__file__).resolve().parents[1]
    remaining = []
    for directory in (root / "tools", root / "core/runtime/tools"):
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Return):
                    continue
                value = node.value
                if isinstance(value, ast.JoinedStr):
                    value = value.values[0] if value.values else None
                if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.startswith(("エラー", "Error")):
                    remaining.append(f"{path.relative_to(root)}:{node.lineno}")
    assert remaining == []


@pytest.mark.parametrize("name", ["_camera_stream", "_screen_peek"])
@pytest.mark.parametrize("response", ["", ToolError("vision offline", {"status": 503}), "エラーという文字が見える"])
def test_device_received_frame_description_outcomes(monkeypatch, name, response):
    from tools import device_tools
    state = {}
    monkeypatch.setattr(device_tools, "load_state", lambda: state)
    monkeypatch.setattr(device_tools, "save_state", Mock())
    monkeypatch.setattr(device_tools, "request_approval", lambda *a, **k: True)
    monkeypatch.setattr(device_tools, "clear_stream_buffer", lambda: None)
    monkeypatch.setattr(device_tools, "get_stream_snapshot", lambda **k: ([("sandbox/frame.png", {})], 1, False))
    monkeypatch.setattr(device_tools, "send_device", lambda *a, **k: "accepted-id")
    describe = Mock(side_effect=response) if isinstance(response, Exception) else Mock(return_value=response)
    monkeypatch.setattr("core.llm.call_llm", describe)
    rec = runtime(getattr(device_tools, name))._execute_tool_use("id", "probe", {})
    assert not rec.is_error and "開始成功" in rec.output
    if response == "":
        assert rec.detail == {"description_error": {"empty_response": True}}
    elif isinstance(response, Exception):
        assert rec.detail["description_error"]["exception_type"] == "ToolError"
        assert rec.detail["description_error"]["message"] == "vision offline"
    else:
        assert rec.detail is None and response in rec.output
    assert state["stream_active"] is True


def test_secret_approval_exception_has_no_secret(monkeypatch):
    from tools import secret_tools
    monkeypatch.setattr(secret_tools, "request_approval", Mock(side_effect=ToolError("NEVER-CONTENT", {"token": "NEVER-CONTENT"})))
    rec = runtime(secret_tools.secret_write)._execute_tool_use("id", "probe", {"name": "example", "content": "NEVER-CONTENT"})
    assert rec.is_error and rec.error_kind == "tool_failure"
    assert "NEVER" not in rec.output + repr(rec.detail)
    assert rec.detail == {"exception_type": "ToolError"}


def test_http_secret_removed_before_exception_truncation(monkeypatch):
    from tools import http_tool
    import httpx
    secret = "NEVER-" * 100
    monkeypatch.setattr("core.auth._load_secrets", lambda: {"auth_profiles": {"test": {"token": secret}}})
    monkeypatch.setattr(http_tool.httpx, "request", Mock(side_effect=httpx.ConnectError(secret)))
    rec = runtime(http_tool.http_request)._execute_tool_use("id", "probe", {"url": "https://example.test/"})
    assert rec.is_error and "NEVER" not in rec.output
    assert "[REDACTED]" in rec.output


def test_main_has_no_obsolete_notification_direct_calls():
    root = Path(__file__).resolve().parents[1]
    source = (root / "main.py").read_text(encoding="utf-8")
    assert "_x_get_notifications" not in source
    assert "_elyth_get_info" not in source


# Stage 3: active writers/readers, state accounting, and the narrow secret_read exception.
@pytest.mark.parametrize("output", ["Error: example", " エラーの例", "[REJECTED] quoted", "該当なし", "done", ""])
@pytest.mark.parametrize("is_error,kind,status", [(False, None, "ok"), (True, "tool_failure", "tool_error"), (True, "exception", "runtime_error"), (True, "rejected", "rejected")])
def test_stage3_status_uses_structure_only(output, is_error, kind, status):
    assert metrics.tool_execution_status(output, is_error, kind) == status
    outward = metrics.build_outward_execution("reflect", {}, output, is_error, "", error_kind=kind)
    assert outward["status"] == status


@pytest.mark.parametrize("name", ["read_file", "write_file", "memory_store"])
@pytest.mark.parametrize("outcome", ["Error: successful content", "エラーという単語", "該当なしという記録", "[REJECTED] quotation", ToolError("done"), RuntimeError("done"), "deny"])
def test_stage3_main_counts_by_is_error(monkeypatch, fixed_io, name, outcome):
    from test_outward_metrics import _fire_harness, _run
    from core.providers.base import ToolUseBlock
    funcs, state, *_ = _fire_harness(monkeypatch, fixed_io, stages=[[
        ToolUseBlock(id="count", name=name, input={"path": "target"})]])
    env = funcs["observed"].__globals__
    rt = env["_runtime"]
    rt.tool_registry.register(ToolSpec(name, "", {}, PermissionMode.READ_ONLY,
        throwing(outcome) if isinstance(outcome, Exception) else lambda a: outcome))
    if outcome == "deny":
        rt.hook_runner.register_pre(lambda n, a: HookRunResult.deny())
    _run(funcs["observed"])
    success = not isinstance(outcome, Exception) and outcome != "deny"
    if name == "memory_store":
        assert state.get("voluntary_memory_store_count", 0) == int(success)
    else:
        key = "files_read" if name == "read_file" else "files_written"
        assert state[key] == (["target"] if success else [])
    rows = [json.loads(line) for line in (fixed_io / metrics.METRICS_FILE_NAME).read_text(encoding="utf-8").splitlines()]
    inv = next(row for row in rows if row["event_type"] == "tool_invocation")
    assert inv["is_error"] is not success and inv["failure_contract"] == 2
    expected_status = ("ok" if success else "rejected" if outcome == "deny" else
                       "tool_error" if isinstance(outcome, ToolError) else "runtime_error")
    assert state["raw_events"][-1]["invocations"][0]["status"] == expected_status


def test_stage3_success_rejected_text_keeps_learning():
    from test_failed_tool_learning import _exercise, _call, _entries
    state, _, env, _ = _exercise([[_call("A", "ok", 80, echo="[REJECTED] quotation")]])
    assert _entries(state)[0]["e2"] == "80%"
    assert state["raw_events"][-1]["e2"] == "80%"
    env["_update_energy"].assert_called_once()


def test_stage3_writers_and_default_readers(fixed_io):
    from test_error_prediction import invocation, lines
    old = [invocation(i, error=True, failure_contract=1) for i in range(3)]
    unversioned = invocation(3, error=True)
    del unversioned["failure_contract"]
    old += [unversioned, invocation(4, failure_contract=99), invocation(5, failure_contract=True)]
    path = fixed_io / metrics.METRICS_FILE_NAME
    path.write_text("\n".join(map(json.dumps, old)) + "\n", encoding="utf-8")
    observer = metrics.ErrorPredictionObserver("new")
    report = observer.rebuild()
    assert report["accepted"] == 0 and report["excluded"] == 6
    assert report["exclusion_reasons"] == {"old_failure_contract": 4, "unknown_failure_contract": 2}
    assert observer.counts == {} and observer.total == (1, 1)
    records = [{"chain_position": 0, "invocation_position": i, "tool_id": str(i), "tool": "probe", "mode": "forced", "tool_input": {}, "output": text, "is_error": error, "error_kind": "tool_failure" if error else None}
               for i, (text, error) in enumerate([("done", True), ("Error: successful", False)])]
    active = metrics.build_tool_invocation_events(records, run_id="new", attempt_id="fire", entry_id=None, cycle_id=None, time="T")
    assert all(row["failure_contract"] == 2 for row in active)
    assert metrics.emit_tool_invocations(active) == 2
    observer.observe(old + active)
    predicted = lines(fixed_io, "error_prediction")
    assert len(predicted) == 2 and all(row["failure_contract"] == 2 for row in predicted)
    assert [(row["alpha"], row["beta"], row["p_error"]) for row in predicted] == [(1, 1, 0.5), (2, 1, 2/3)]
    # Legacy predictions with valid numeric data still cannot contribute.
    legacy_predictions = [{**predicted[0], "failure_contract": v, "run_id": "legacy" + str(v)} for v in (1, 99, True)]
    legacy_predictions.append({k: v for k, v in legacy_predictions[0].items() if k != "failure_contract"})
    events = lines(fixed_io) + legacy_predictions
    summary = metrics.summarize_error_prediction(events)
    assert summary["invocations"] == summary["count"] == 2 and summary["missing"] == 0
    restored = metrics.ErrorPredictionObserver("next-run")
    rebuilt = restored.rebuild()
    assert rebuilt["accepted"] == 2 and rebuilt["exclusion_reasons"] == report["exclusion_reasons"]
    assert restored.counts == observer.counts == {"probe": (2, 2)}
    assert lines(fixed_io, "error_model_rebuilt")[-1]["exclusion_reasons"] == report["exclusion_reasons"]


SECRET_BODY = "my-private-value\nAuthorization: Bearer sandbox-header\nCookie: sandbox-cookie\nhttps://user:sandbox-pass@example.test/?token=sandbox-query\nconfigured-secret\n"


def _secret_runtime(tmp_path, monkeypatch, provider=None, hooks=None):
    from tools import secret_tools
    monkeypatch.setattr(secret_tools, "_SECRETS_DIR", tmp_path)
    (tmp_path / "fixture").write_text(SECRET_BODY, encoding="utf-8")
    rt = runtime(secret_tools.secret_read, provider, hooks)
    rt.tool_registry.register(ToolSpec("secret_read", "", {}, PermissionMode.READ_ONLY, secret_tools.secret_read))
    return rt


@pytest.mark.parametrize("provider_name", ["lmstudio", "anthropic"])
def test_secret_read_success_provider_original_observation_redacted(tmp_path, monkeypatch, provider_name):
    from core.providers.openai_compat import OpenAIProvider
    from core.providers.anthropic import AnthropicProvider
    from core import prompt_trace as trace
    hooks = HookRunner()
    seen = []
    hooks.register_post(lambda n, a, o: seen.append(o) or HookRunResult.allow())
    provider = OpenAIProvider("fake") if provider_name == "lmstudio" else AnthropicProvider("fake")
    rt = _secret_runtime(tmp_path, monkeypatch, trace.TracingProvider(provider), hooks)
    rec = rt._execute_tool_use("secret-id", "secret_read", {"name": "fixture"})
    expected = "[secret_read] fixture\n" + SECRET_BODY
    assert rec.provider_output == expected and not rec.is_error
    assert "sandbox-header" not in rec.output and "configured-secret" not in rec.output
    assert rec.output == "[secret_read] fixture\n[REDACTED]"
    assert seen == [rec.output]
    assert "sandbox-header" not in repr(rec) + repr(rt.session.messages)
    sent, traced = [], []
    monkeypatch.setattr("httpx.post", lambda url, *, json, **kw: sent.append(json) or SimpleNamespace(status_code=200, json=lambda: {}, raise_for_status=lambda: None))
    monkeypatch.setattr(provider, "_parse_response", lambda d: AssistantMessage())
    monkeypatch.setattr(trace, "_begin", lambda *a: {"run_id": "r"})
    monkeypatch.setattr(trace, "emit", traced.append)
    rt._call_llm()
    message = sent[0]["messages"][-1]
    body = message["content"] if provider_name == "lmstudio" else message["content"][0]["content"]
    assert body == expected
    assert "my-private-value" not in json.dumps(traced)
    assert "sandbox-header" not in json.dumps(traced) and "configured-secret" not in json.dumps(traced)
    # Repeated provider serialization is lossless; observation copies stay redacted.
    for serialize in (rt.session.serialize_for_openai, rt.session.serialize_for_anthropic):
        assert "sandbox-header" in json.dumps(serialize())
        assert "sandbox-header" not in json.dumps(serialize(for_observation=True))
    rt.session.clear()
    assert rt.session._provider_tool_results == {}


@pytest.mark.parametrize("forced", [False, True])
def test_secret_read_claude_sdk_success_preserves_body(tmp_path, monkeypatch, forced):
    from core.providers.claude_code import ClaudeCodeProvider
    from claude_agent_sdk import create_sdk_mcp_server
    from mcp.types import CallToolRequest, CallToolRequestParams
    payloads = []
    class Provider:
        name = "claude_code"
        def stream(self, request):
            captured = []
            tool = ClaudeCodeProvider._build_tool_handler(
                {"name": "secret_read", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}}}, request.tool_executor, captured)
            server = create_sdk_mcp_server(name="test", tools=[tool])["instance"]
            response = asyncio.run(server.request_handlers[CallToolRequest](CallToolRequest(params=CallToolRequestParams(name="secret_read", arguments={"name": "fixture"}))))
            payloads.append(response.root)
            return AssistantMessage(tool_invocations=captured)
    rt = _secret_runtime(tmp_path, monkeypatch, Provider())
    summary = rt.run_turn_with_forced_tool("secret_read") if forced else rt.run_turn()
    expected = "[secret_read] fixture\n" + SECRET_BODY
    assert not payloads[0].isError and payloads[0].content[0].text == expected
    assert rt.session.serialize_for_anthropic()[-1]["content"][0]["content"] == expected
    assert "my-private-value" not in repr(summary)
    assert "sandbox-header" not in repr(summary)
    assert summary.tool_invocations[0].provider_output == expected


@pytest.mark.parametrize("name,failed", [("secret_read", True), ("probe", False), ("probe", True)])
def test_secret_read_exception_is_success_only(name, failed):
    rt = runtime(throwing(ToolError(SECRET_BODY, {"text": SECRET_BODY})) if failed else lambda a: SECRET_BODY)
    rt.tool_registry.register(ToolSpec(name, "", {}, PermissionMode.READ_ONLY, rt.tool_registry.get("probe").handler))
    rec = rt._execute_tool_use("id", name, {})
    assert rec.provider_output is None
    assert "sandbox-header" not in rec.output + repr(rec.detail)
    assert "sandbox-header" not in json.dumps(rt.session.serialize_for_anthropic())


def test_secret_read_main_observation_and_console_stay_redacted(monkeypatch, fixed_io, capsys):
    from test_outward_metrics import _fire_harness, _run
    from core.providers.base import ToolUseBlock
    from tools import secret_tools
    monkeypatch.setattr(secret_tools, "_SECRETS_DIR", fixed_io)
    (fixed_io / "fixture").write_text(SECRET_BODY, encoding="utf-8")
    funcs, state, *_ = _fire_harness(monkeypatch, fixed_io, stages=[[
        ToolUseBlock(id="secret", name="secret_read", input={"name": "fixture"})]])
    env = funcs["observed"].__globals__
    env["_runtime"].tool_registry.register(ToolSpec("secret_read", "", {}, PermissionMode.READ_ONLY, secret_tools.secret_read))
    _run(funcs["observed"])
    output = capsys.readouterr().out + json.dumps(state) + (fixed_io / metrics.METRICS_FILE_NAME).read_text(encoding="utf-8")
    output += repr(env["append_debug_log"].call_args_list) + repr(env["broadcast_log"].call_args_list)
    assert "my-private-value" not in output
    assert "sandbox-header" not in output and "configured-secret" not in output



def test_stage3_real_file_failure_then_write_then_success(tmp_path):
    from core.runtime.tools import file_ops
    from core.providers.base import ToolUseBlock
    requests = []
    stages = [[ToolUseBlock("r1", "read_file", {"path": "new.txt"})],
              [ToolUseBlock("w1", "write_file", {"path": "new.txt", "content": "Error: successful file content"})],
              [ToolUseBlock("r2", "read_file", {"path": "new.txt"})], []]
    class Provider:
        name = "anthropic"
        def stream(self, request):
            requests.append(request)
            return AssistantMessage(tool_uses=stages.pop(0))
    registry = ToolRegistry()
    file_ops.register(registry, tmp_path)
    hooks = HookRunner()
    success = Mock(return_value=HookRunResult.allow())
    failure = Mock(return_value=HookRunResult.allow())
    hooks.register_post(success)
    hooks.register_failure(failure)
    rt = ConversationRuntime(Provider(), registry, hooks, PermissionEnforcer(PermissionMode.ALLOW), max_iterations=4)
    summary = rt.run_turn()
    assert [(r.tool_name, r.is_error, r.error_kind) for r in summary.tool_invocations] == [
        ("read_file", True, "tool_failure"), ("write_file", False, None), ("read_file", False, None)]
    assert success.call_count == 2 and failure.call_count == 1
    assert "Error: successful file content" in summary.tool_invocations[-1].output
    failed_body = json.loads(requests[1].messages[-1]["content"][0]["content"])
    assert failed_body["is_error"] is True and "not found" in failed_body["message"]


def test_secret_read_private_history_identity_compaction_and_storage(tmp_path):
    from core.runtime.session import Session
    from core.runtime.compaction import compact_session
    from core.runtime.session_store import SessionStore
    session = Session()
    session.push_tool_result("same-id", "[REDACTED-1]", provider_content="secret-one")
    session.push_tool_result("same-id", "[REDACTED-2]", provider_content="secret-two")
    assert [m["content"] for m in session.serialize_for_openai()] == ["secret-one", "secret-two"]
    store = SessionStore(tmp_path)
    sid = store.save(session)
    assert "secret-one" not in (tmp_path / (sid + ".json")).read_text(encoding="utf-8")
    assert "secret-two" not in repr(store.load(sid).messages)
    compact_session(session, keep_recent=1)
    assert session.serialize_for_openai()[-1]["content"] == "secret-two"
    assert "secret-one" not in repr(session.serialize_for_openai())
    session.clear()
    session.push_tool_result("same-id", "public")
    assert session.serialize_for_openai()[-1]["content"] == "public"




@pytest.mark.parametrize("phase", ["reconciliation", "links", "link_judge"])
def test_memory_lower_exception_console_redacted(tmp_path, monkeypatch, capsys, phase):
    from core import memory, memory_links, reconciliation
    error = RuntimeError("configured-secret Authorization: Bearer lower-secret")
    if phase == "link_judge":
        result = memory_links._llm_judge_link({"content": "a"}, {"content": "b"}, llm_call_fn=Mock(side_effect=error), failures=[])
        assert result["link_type"] == "none"
    else:
        monkeypatch.setattr(memory, "_network_file", lambda n: tmp_path / "memory.jsonl")
        monkeypatch.setattr(reconciliation, "check_on_write", Mock(side_effect=error) if phase == "reconciliation" else Mock())
        monkeypatch.setattr(memory_links, "generate_links_for", Mock(side_effect=error) if phase == "links" else Mock())
        memory.memory_store(content="a", _auto_metadata=False, _state={}, _reconcile_embed_fn=lambda x: [1], _post_save_failures=[])
    output = capsys.readouterr().out
    assert "[REDACTED]" in output
    assert "configured-secret" not in output and "lower-secret" not in output


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_http_redacts_before_truncation(monkeypatch, method):
    from tools import http_tool
    secret = "private-" + "x" * 150
    monkeypatch.setattr("core.auth._load_secrets", lambda: {"auth_profiles": {"fixture": {"token": secret}}})
    previews = []
    monkeypatch.setattr(http_tool, "request_approval", lambda name, preview, **kw: previews.append(preview) or False)
    monkeypatch.setattr(http_tool.httpx, "request", Mock(side_effect=http_tool.httpx.TimeoutException("timeout")))
    with pytest.raises(ToolError) as caught:
        http_tool.http_request({"method": method, "url": "https://example.test/" + secret, "body": "prefix " + secret})
    output = str(caught.value) + repr(previews)
    assert "private-" not in output and "[REDACTED]" in output


@pytest.mark.parametrize("case", ["ui_pending", "agent_pending", "packet", "tasks", "cron", "web_fetch", "remote_trigger", "display_log", "x_preview", "x_read", "elyth", "memory", "dismiss"])
def test_display_paths_redact_before_shortening(monkeypatch, capsys, case):
    from core.runtime.tools import ui, skill, util, task, team_cron, web
    from tools import ui_tools, x_tools, elyth_tools, memory_tool, builtin
    secret = "bare-private-prefix-" + "x" * 22000
    monkeypatch.setattr("core.auth._load_secrets", lambda: {"auth_profiles": {"fixture": {"token": secret}}})
    text = "visible " + secret + " tail"
    if case == "ui_pending":
        monkeypatch.setitem(ui._ui_bridge, "send_user", None)
        handler = lambda: ui.send_user_message({"message": text})
    elif case == "agent_pending":
        monkeypatch.setitem(skill._agent_bridge, "dispatch", None)
        handler = lambda: skill.agent({"agent_type": "test", "task": text})
    elif case == "packet":
        handler = lambda: util.run_task_packet({"packet": {"objective": text, "scope": text}})
    elif case == "tasks":
        monkeypatch.setattr(task._registry, "list_all", lambda: [SimpleNamespace(id="1", status="running", description=text)])
        handler = lambda: task.task_list({})
    elif case == "cron":
        monkeypatch.setattr(team_cron._cron_registry, "list_all", lambda: [SimpleNamespace(id="1", schedule="* * * * *", prompt=text, description="")])
        handler = lambda: team_cron.cron_list({})
    elif case in ("web_fetch", "remote_trigger"):
        response = SimpleNamespace(text=text, status_code=200, headers={}, raise_for_status=lambda: None)
        monkeypatch.setattr(web.httpx, "get", Mock(return_value=response))
        monkeypatch.setattr(web.httpx, "request", Mock(return_value=response))
        handler = lambda: getattr(web, case)({"url": "https://example.test/", "prompt": "read"})
    elif case == "display_log":
        sent, logged = [], []
        monkeypatch.setattr(ui_tools, "broadcast", lambda row: sent.append(row) or 1)
        monkeypatch.setattr(ui_tools, "broadcast_log", logged.append)
        ui_tools._output_display({"channel": "test", "content": text})
        assert sent[0]["content"] == text
        handler = lambda: "".join(logged)
    elif case == "x_preview":
        x_tools._x_confirm("test", text)
        handler = lambda: capsys.readouterr().out
    elif case == "x_read":
        _x_page(monkeypatch, [_x_article(text=text)])
        handler = lambda: runtime(x_tools._x_search)._execute_tool_use("id", "probe", {"query": "q", "count": 1}).output
    elif case == "elyth":
        monkeypatch.setattr(elyth_tools, "load_state", lambda: {})
        monkeypatch.setattr(elyth_tools, "_resolve_elyth_feedback", lambda *a: False)
        handler = lambda: elyth_tools._format_notifications({"notifications": [{"post_content": text}], "extra": text})
    elif case == "memory":
        entry = {"per_tool": [{"chain_position": 0, "invocation_position": 0, "intent": text, "expect": text}],
                 "invocations": [{"chain_position": 0, "invocation_position": 0, "tool": "probe", "status": "ok", "args": {"value": text}, "output": text}]}
        handler = lambda: memory_tool._memory_execution_lines(entry, detail=True)
    else:
        monkeypatch.setattr(builtin, "load_state", lambda: {"pending": [{"id": "p_1", "content": text}]})
        monkeypatch.setattr(builtin, "save_state", Mock())
        handler = lambda: builtin._wait_or_dismiss({"dismiss": "p_1"})
    try:
        output = str(handler())
    except ToolError as error:
        assert case in ("ui_pending", "agent_pending", "packet")
        output = str(error)
    # Inspect the fragment as well as the complete credential: late redaction misses it.
    assert "bare-private-prefix-" not in output
    assert "visible [REDACTED] tail" in output


@pytest.fixture
def persistence_io(tmp_path, monkeypatch):
    from core import state as st, config, memory, view_persistence as vp
    from tools import world_fact_view_tool
    monkeypatch.setattr(st, "STATE_FILE", tmp_path / "state.json")
    for module in (st, config, memory, world_fact_view_tool):
        monkeypatch.setattr(module, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(vp, "_VIEWS", {})
    monkeypatch.setattr(vp, "_DIRTY", set())
    st.bind_live_state(None)
    yield st
    st.bind_live_state(None)


def _main_post_hook(state, st):
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    node = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "_post_hook")
    env = dict(state=state, save_state=st.save_state, load_state=st.load_state, copy=copy,
               _base_post_hook=Mock(return_value=HookRunResult.allow()),
               _hook_ctx={"evaluations": []})
    exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), env)
    return env["_post_hook"]


def test_live_tool_failure_then_success_and_cycle_save(persistence_io):
    st = persistence_io
    state = st.load_state()
    st.bind_live_state(state)
    assert st.save_state(state)
    state.update(cycle_id=19, memory_only="keep", pending=[{"attempts": 3}])
    hooks = HookRunner()
    hooks.register_failure(make_post_tool_use_failure_logger(state, lambda: state["cycle_id"]))
    hooks.register_post(_main_post_hook(state, st), with_tool_id=True)
    def partial(args):
        tool_state = st.load_state()
        assert tool_state is state
        tool_state["partial"] = 7
        assert st.save_state(tool_state)
        raise ToolError("partial failure")
    rt = runtime(partial, hooks=hooks)
    rec = rt._execute_tool_use("fail", "probe", {})
    assert rec.is_error and rec.error_kind == "tool_failure"
    rt.tool_registry.get("probe").handler = lambda args: "success"
    assert not rt._execute_tool_use("ok", "probe", {}).is_error
    assert st.save_state(state)
    disk = json.loads(st.STATE_FILE.read_text(encoding="utf-8"))
    assert disk["partial"] == 7 and disk["memory_only"] == "keep"
    assert disk["cycle_id"] == 19 and disk["pending"] == [{"attempts": 3}]
    assert disk["tool_errors"][0]["cycle"] == 19
    assert st.load_state() is state
    assert st.save_state(dict(state)) is False


def test_main_binding_and_no_state_replacement(persistence_io):
    st = persistence_io
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    begin = next(i for i,n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id == "state" for t in n.targets))
    state = st.load_state()
    env = {"load_startup_state": lambda: state, "bind_live_state": st.bind_live_state}
    exec(compile(ast.Module(body=main.body[begin:begin+2], type_ignores=[]), "main.py", "exec"),env)
    assert st.load_state() is state
    for removed in ("merge_tool_state", "_capture_tool_disk_state", "_tool_sync_pending", "_refresh_state", "_save_synced_state", "state.clear()"):
        assert removed not in source
    assert "register_failure(_base_failure_hook)" in source


@pytest.mark.parametrize("phase", ["generation", "views", "state"])
def test_save_failure_retains_live_and_retries(persistence_io, monkeypatch, capsys, phase):
    from core import view_persistence as vp
    st = persistence_io
    state = st.load_state()
    st.bind_live_state(state)
    assert st.save_state(state)
    state["memory_only"] = {"value": 17}
    with monkeypatch.context() as patch:
        if phase == "views":
            patch.setattr(vp, "flush_dirty_views", Mock(side_effect=OSError("private text")))
        else:
            original = st._atomic_write
            def failing(path, text):
                if (path == st.STATE_FILE) == (phase == "state"):
                    raise OSError("private text")
                original(path, text)
            patch.setattr(st, "_atomic_write", failing)
        assert st.save_state(state) is False
    assert st.load_state() is state and state["memory_only"] == {"value": 17}
    assert state["_persistence"]["last_failure"]["phase"] == phase
    output = capsys.readouterr().out
    assert "persistence failed" in output and "private text" not in output
    assert st.save_state(state) is True
    disk = json.loads(st.STATE_FILE.read_text(encoding="utf-8"))
    assert disk["memory_only"]["value"] == 17
    assert disk["_persistence"]["last_failure"]["exception_type"] == "OSError"


def test_generations_saved_before_write_and_capped(persistence_io, monkeypatch):
    st = persistence_io
    state = st.load_state()
    st.bind_live_state(state)
    for value in range(5):
        state["cycle_id"] = value
        assert st.save_state(state)
    assert [json.loads(p.read_text())["cycle_id"] for p in st._generations()] == [3,2,1]
    original = st._atomic_write
    def fail_current(path, text):
        if path == st.STATE_FILE:
            assert json.loads(st._generations()[0].read_text())["cycle_id"] == 4
            raise OSError("disk failed")
        original(path, text)
    monkeypatch.setattr(st, "_atomic_write", fail_current)
    state["cycle_id"] = 5
    assert not st.save_state(state)
    assert json.loads(st.STATE_FILE.read_text())["cycle_id"] == 4
    assert json.loads(st._generations()[0].read_text())["cycle_id"] == 4


@pytest.mark.parametrize("bad", ["{broken", "[]", '{"dispositions":null}', '{"cycle_id":"bad"}'])
@pytest.mark.parametrize("have_generation", [True, False])
def test_startup_recovery_latest_valid_views_cycles_and_records(persistence_io, bad, have_generation):
    from core import view_persistence as vp
    st = persistence_io
    generation = st.MEMORY_DIR / "state_generations"
    generation.mkdir()
    (generation / "state-003.json").write_text("{broken", encoding="utf-8")
    (generation / "state-002.json").write_text(json.dumps({"cycle_id":3,"marker":"latest","subjective_entries":[{"id":"old"}]}) if have_generation else "[]", encoding="utf-8")
    (generation / "state-001.json").write_text('{"cycle_id":1,"marker":"older"}' if have_generation else "null", encoding="utf-8")
    st.STATE_FILE.write_text(bad, encoding="utf-8")
    raw = {"id":"session_0018", "tool":"probe", "result":"current"}
    subj = {"id":"session_0018", "intent":"latest intent"}
    (st.MEMORY_DIR / "raw_events.jsonl").write_text(json.dumps(raw)+"\n", encoding="utf-8")
    path = st.MEMORY_DIR / "subjective_entries.jsonl"
    path.write_text(json.dumps(subj)+"\n", encoding="utf-8")
    (st.MEMORY_DIR / "metrics_events.jsonl").write_text('{"cycle_id":21}\n', encoding="utf-8")
    state = st.load_startup_state()
    assert state["cycle_id"] == 21
    assert state["raw_events"] == [raw] and state["subjective_entries"] == [subj]
    fact = state["_persistence"]["startup"]
    assert fact["reason"] == "invalid_state"
    assert fact["source"].endswith("state-002.json") if have_generation else fact["source"] == "initial"
    if have_generation:
        assert state["marker"] == "latest"
    assert (st.STATE_FILE.parent / fact["preserved"]).read_text(encoding="utf-8") == bad
    st.bind_live_state(state)
    vp.register_view("subjective_entries", path, "subjective_entries")
    vp.mark_view_dirty("subjective_entries")
    st.record_startup_recovery(state)
    assert st.save_state(state)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0] == subj and not any(row["id"] == "old" for row in rows)
    events = [json.loads(line) for line in (st.MEMORY_DIR / "metrics_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e.get("event_type") == "state_recovery" and e["source"] == fact["source"] for e in events)
    assert json.loads(state["raw_events"][-1]["result"])["source"] == fact["source"]


@pytest.mark.parametrize("recover_after", [1, None])
@pytest.mark.parametrize("have_generation", [True, False])
def test_startup_os_retry_does_not_remove_original(persistence_io, monkeypatch, recover_after, have_generation):
    st = persistence_io
    st.STATE_FILE.write_text('{"cycle_id":8}', encoding="utf-8")
    if have_generation:
        folder = st.MEMORY_DIR / "state_generations"
        folder.mkdir()
        (folder / "state-001.json").write_text('{"cycle_id":5}', encoding="utf-8")
    original = st._read_state
    calls = []
    def read(path):
        if path == st.STATE_FILE:
            calls.append(path)
            if recover_after is None or len(calls) <= recover_after:
                raise PermissionError("unreadable")
        return original(path)
    monkeypatch.setattr(st, "_read_state", read)
    monkeypatch.setattr(st.time, "sleep", Mock())
    state = st.load_startup_state()
    assert len(calls) == (2 if recover_after else 3)
    assert state["cycle_id"] == (8 if recover_after else 5 if have_generation else 0)
    assert st.STATE_FILE.read_text() == '{"cycle_id":8}'
    assert not list(st.STATE_FILE.parent.glob("state.corrupt-*"))
    fact = state["_persistence"]["startup"]
    assert fact["reason"] == "os_error"
    assert fact["attempts"] == (2 if recover_after else 3)


def test_common_load_is_read_only_and_unbinding_is_explicit(persistence_io):
    st = persistence_io
    st.STATE_FILE.write_text("{broken", encoding="utf-8")
    before = {p: p.read_bytes() for p in st.MEMORY_DIR.rglob("*") if p.is_file()}
    state = st.load_state()
    assert {p: p.read_bytes() for p in st.MEMORY_DIR.rglob("*") if p.is_file()} == before
    st.bind_live_state(state)
    st.STATE_FILE.write_text('{"cycle_id":9}', encoding="utf-8")
    assert st.load_state() is state and state["cycle_id"] == 0
    st.bind_live_state(None)
    replacement = st.load_state()
    assert replacement is not state and replacement["cycle_id"] == 9
    st.bind_live_state(replacement)
    assert st.load_state() is replacement


def test_world_read_report_uses_jsonl_without_replacing_live(persistence_io):
    from tools.world_fact_view_tool import _world_fact_view
    st = persistence_io
    state = st.load_state()
    st.bind_live_state(state)
    state["raw_events"] = [{"id":"live-only", "result":"unsaved"}]
    path = st.MEMORY_DIR / "raw_events.jsonl"
    path.write_text('{"id":"disk", "result":"read"}\n{broken\n', encoding="utf-8")
    result = _world_fact_view({})
    assert isinstance(result, ToolResult) and result.detail["rows_unreadable"] == 1
    assert json.loads(result.message)["events"][0]["id"] == "disk"
    assert st.load_state() is state and state["raw_events"][0]["id"] == "live-only"
    path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ToolError):
        _world_fact_view({})


@pytest.mark.parametrize("success", [False, True])
def test_reboot_requires_successful_save(persistence_io, monkeypatch, success):
    from tools import reboot
    monkeypatch.setattr(reboot, "request_approval", lambda *args: True)
    monkeypatch.setattr(reboot, "save_state", lambda state: success)
    stop, spawn, sleep, terminate = Mock(), Mock(), Mock(), Mock(side_effect=SystemExit(0))
    monkeypatch.setattr(reboot, "stop_ws_server", stop)
    monkeypatch.setattr(reboot.subprocess, "Popen", spawn)
    monkeypatch.setattr(reboot.time, "sleep", sleep)
    monkeypatch.setattr(reboot.os, "_exit", terminate)
    with pytest.raises(SystemExit if success else ToolError):
        reboot._reboot({})
    assert stop.called == spawn.called == terminate.called == success


def test_tool_and_save_share_lock_across_threads(persistence_io, monkeypatch):
    st = persistence_io
    state = st.load_state()
    st.bind_live_state(state)
    entered, release, save_attempted, written = (threading.Event() for _ in range(4))
    failures = []
    original = st._atomic_write
    def write(path, text):
        written.set()
        original(path,text)
    monkeypatch.setattr(st, "_atomic_write", write)
    def handler(args):
        st.load_state()["first"] = 1
        entered.set()
        assert release.wait(3)
        state["second"] = 2
        assert st.save_state(state)  # same-thread reentrance
        return "ok"
    rt = runtime(handler)
    def tool_thread():
        rec = rt._make_tool_executor()("id", "probe", {})
        if rec.is_error:
            failures.append(rec.output)
    def saver():
        save_attempted.set()
        st.save_state(state)
    t = threading.Thread(target=tool_thread)
    t.start()
    assert entered.wait(3)
    saver_thread = threading.Thread(target=saver)
    saver_thread.start()
    try:
        assert save_attempted.wait(3)
        assert not written.wait(0.15)
    finally:
        release.set()
        t.join(3)
        saver_thread.join(3)
    assert not t.is_alive() and not saver_thread.is_alive() and not failures
    disk = json.loads(st.STATE_FILE.read_text())
    assert disk["first"] == 1 and disk["second"] == 2


@pytest.mark.parametrize("manual", [False, True])
def test_reflection_persists_same_live(persistence_io, monkeypatch, manual):
    from core.reflection import reflect_and_persist
    st = persistence_io
    state = st.load_state()
    st.bind_live_state(state)
    assert st.save_state(state)
    state.update(memory_only="retain", reflection_cycle=20)
    st.load_state()["tool_update"] = 7
    def reflect(current, llm, **kwargs):
        assert current is st.load_state()
        current["reflection_value"] = 9
        return {"notes": []}
    reflect_and_persist(state, Mock(), strict=manual, reflect_fn=reflect)
    disk = json.loads(st.STATE_FILE.read_text())
    assert disk["memory_only"] == "retain" and disk["tool_update"] == 7
    assert disk["reflection_value"] == 9 and disk["reflection_cycle"] == 0


def test_generation_directory_excluded_from_file_tools(persistence_io):
    from core.runtime.tools.file_ops import (_make_read_file, _make_write_file,
        _make_edit_file, _make_glob_search, _make_grep_search)
    st = persistence_io
    root = st.MEMORY_DIR
    directory = root / "memory" / "state_generations"
    directory.mkdir(parents=True)
    path = directory / "state-1.json"
    path.write_text("private_generation", encoding="utf-8")
    (root / "ordinary.txt").write_text("public", encoding="utf-8")
    for factory in (_make_read_file, _make_write_file, _make_edit_file):
        with pytest.raises(ToolError):
            factory(root)({"path": str(path), "content": "overwrite", "old_string":"private_generation", "new_string":"overwrite"})
    assert "state-1" not in _make_glob_search(root)({"pattern":"**/*"})
    assert "private_generation" not in _make_grep_search(root)({"pattern":"private_generation"}).replace("No matches for pattern: private_generation", "")
    assert path.read_text() == "private_generation"


def test_main_startup_sanity_allows_state_recovery(persistence_io, monkeypatch):
    from core import sanity_check as sanity
    st = persistence_io
    st.STATE_FILE.write_text("{broken", encoding="utf-8")
    directory = st.MEMORY_DIR / "memory" / "state_generations"
    directory.mkdir(parents=True)
    (directory / "state-1.json").write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(sanity, "_check_imports", Mock())
    revert = Mock(side_effect=AssertionError("state failure must not reach git"))
    monkeypatch.setattr(sanity, "_try_auto_revert_from_stash", revert)
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                and n.value.func.id == "enforce_sanity_check")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"),
         {"enforce_sanity_check": sanity.enforce_sanity_check, "BASE_DIR": st.MEMORY_DIR})
    assert not revert.called
    assert st.load_startup_state()["_persistence"]["startup"]["reason"] == "invalid_state"
    # Other broken memory JSON is still diagnosed.
    (directory.parent / "ordinary.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(sanity.SanityCheckError):
        sanity._check_memory_jsons(st.MEMORY_DIR)


@pytest.mark.parametrize("name", ["_camera_stream", "_screen_peek"])
def test_device_rechecks_state_after_approval(persistence_io, monkeypatch, name):
    from tools import device_tools
    st = persistence_io
    state = st.load_state()
    st.bind_live_state(state)
    def approve(*args, **kwargs):
        state["stream_active"] = True
        state["stream_id"] = "another-stream"
        return True
    monkeypatch.setattr(device_tools, "request_approval", approve)
    send = Mock(side_effect=AssertionError("must recheck before sending"))
    clear = Mock()
    monkeypatch.setattr(device_tools, "send_device", send)
    monkeypatch.setattr(device_tools, "clear_stream_buffer", clear)
    with pytest.raises(ToolError, match="stream became active"):
        getattr(device_tools, name)({})
    assert not send.called and not clear.called
    assert state["stream_id"] == "another-stream"


@pytest.mark.parametrize("disk_cycle", [3, 40])
def test_startup_cycle_uses_raw_ids_and_preserves_higher_state(persistence_io, disk_cycle):
    st = persistence_io
    st.STATE_FILE.write_text(json.dumps({"cycle_id":disk_cycle}), encoding="utf-8")
    (st.MEMORY_DIR / "raw_events.jsonl").write_text('{"id":"abc_0018"}\n', encoding="utf-8")
    assert st.load_startup_state()["cycle_id"] == max(18, disk_cycle)


@pytest.mark.parametrize("bad", ["{broken", "[]", '{"dispositions":null}', '{"energy":NaN}'])
def test_generation_only_copies_state_that_passes_migration(persistence_io, bad):
    st = persistence_io
    state = st.load_state()
    st.bind_live_state(state)
    st.STATE_FILE.write_text(bad, encoding="utf-8")
    assert st.save_state(state)
    assert st._generations() == []
    assert isinstance(json.loads(st.STATE_FILE.read_text()), dict)


def test_corrupt_preservation_failure_cannot_overwrite_original(persistence_io, monkeypatch):
    st = persistence_io
    st.STATE_FILE.write_text("{broken", encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.setattr(st, "_preserve_corrupt", Mock(side_effect=PermissionError()))
        state = st.load_startup_state()
        st.bind_live_state(state)
        assert st.save_state(state) is False
        assert st.STATE_FILE.read_text() == "{broken"
        assert state["_persistence"]["last_failure"]["phase"] == "preserve_corrupt"
    assert st.save_state(state)
    preserved = state["_persistence"]["startup"]["preserved"]
    assert (st.STATE_FILE.parent / preserved).read_text() == "{broken"


def test_main_records_recovery_and_retains_notice_on_record_failure(persistence_io, monkeypatch):
    from core import memory
    st = persistence_io
    st.STATE_FILE.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(memory, "_record_entry", Mock(side_effect=OSError()))
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    main = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == "main")
    begin = next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id == "state" for t in n.targets))
    env = dict(load_startup_state=st.load_startup_state, bind_live_state=st.bind_live_state,
               record_startup_recovery=st.record_startup_recovery, uuid=st.uuid)
    exec(compile(ast.Module(body=main.body[begin:begin+5], type_ignores=[]), "main.py", "exec"), env)
    state = env["state"]
    assert st.load_state() is state
    assert state["raw_events"][-1]["tool"] == "state_recovery"
    assert json.loads(state["raw_events"][-1]["result"])["reason"] == "invalid_state"
    assert state["_persistence"]["last_failure"]["phase"] == "recovery_record"
