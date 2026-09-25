"""M0 の全文・対応 ID・非介入。実 LLM は呼ばない。"""
import ast
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_outward_metrics import (
    MAIN_SOURCE, _fire_harness, _outward_lines, _run, fixed_io,
)
from core import prompt_trace as trace
from core.providers.base import ApiRequest, AssistantMessage, ToolUseBlock


def _lines(directory):
    return [json.loads(line) for line in (directory / trace.TRACE_FILE_NAME)
            .read_text(encoding="utf-8").splitlines()]


def _event(attempt="r_1"):
    return {"run_id": "r", "attempt_id": attempt, "cycle_id": 3}


def _provider(response=None, stream=None, name="anthropic"):
    return SimpleNamespace(name=name, model="llm2-model", extra="delegated",
                           supports_tool_use=lambda: True, supports_vision=lambda: False,
                           stream=stream or Mock(return_value=response or AssistantMessage()))


def test_snapshot_identity_and_multiple_calls(fixed_io, monkeypatch):
    """浅いコピー、切詰め、段ごとの採番、応答時の ID 取り直しを落とす。"""
    executor = Mock()
    request = ApiRequest(system_prompt="全文\n" * 20000,
                         messages=[{"role": "user", "content": [{"text": "before"}]}],
                         tools=[{"schema": {"enum": ["before"]}}],
                         tool_choice={"name": "first"}, max_tokens=789, temperature=0.3,
                         image_paths=["画像.png"], tool_executor=executor)
    expected = copy.deepcopy(request)
    response = AssistantMessage(tool_uses=[ToolUseBlock("a", "first", {}), ToolUseBlock("b", "second", {})])
    emitted = []
    real_emit = trace.emit

    def emit(record):
        emitted.append(record)
        real_emit(record)

    monkeypatch.setattr(trace, "emit", emit)

    def stream(actual):
        assert actual is request and actual.tool_executor is executor
        assert _lines(fixed_io)[-1]["kind"] == "request"  # provider に入る前に保存済み。
        actual.messages[0]["content"][0]["text"] = "changed"
        actual.tools[0]["schema"]["enum"].append("changed")
        trace.set_chain_position(99, 999)
        return response

    provider = _provider(stream=stream)
    wrapped = trace.TracingProvider(provider)
    assert (wrapped.name, wrapped.model, wrapped.supports_tool_use(), wrapped.supports_vision(), wrapped.extra) == (
        "anthropic", "llm2-model", True, False, "delegated")
    with trace.trace_fire(_event()):
        for position in (0, 0, 1):
            trace.set_chain_position(position, 3)
            assert wrapped.stream(request) is response
    rows = _lines(fixed_io)
    assert [r["call_index"] for r in rows] == [0, 0, 1, 1, 2, 2]
    assert [r["chain_position"] for r in rows] == [0, 0, 0, 0, 1, 1]
    assert all(r["cycle_id"] == 3 for r in rows)
    for key in ("system_prompt", "messages", "tools", "tool_choice", "max_tokens", "temperature", "image_paths"):
        assert rows[0][key] == getattr(expected, key)
        assert emitted[0][key] == getattr(expected, key)
    assert "tool_executor" not in rows[0]
    assert rows[1]["tool_uses"] == [{"id": "a", "name": "first"}, {"id": "b", "name": "second"}]
    assert rows[0]["observation_scope"] == "provider_api_request_before_conversion"
    assert [r["trace_id"] for r in rows[::2]] == ["r_1:0", "r_1:1", "r_1:2"]
    assert all(a["trace_id"] == b["trace_id"] for a, b in zip(rows[::2], rows[1::2]))


def test_claude_code_ids_and_unobserved_scope(fixed_io):
    response = AssistantMessage(tool_invocations=[
        {"tool_id": "sdk-a", "tool_name": "reflect", "tool_input": {}},
        {"tool_id": "sdk-b", "tool_name": "output_display", "tool_input": {}},
    ])
    with trace.trace_fire(_event()):
        trace.TracingProvider(_provider(response, name="claude_code")).stream(ApiRequest())
    request, result = _lines(fixed_io)
    assert request["sdk_internal_inputs"] == "unobserved"
    assert result["tool_uses"] == []
    assert result["tool_invocations"] == [
        {"tool_id": "sdk-a", "tool_name": "reflect"},
        {"tool_id": "sdk-b", "tool_name": "output_display"}]


@pytest.mark.parametrize("failure", ["copy", "json", "write", "response_ids"])
def test_recording_failures_do_not_change_calls(failure, monkeypatch, fixed_io):
    """各失敗点で provider が一度だけ呼ばれ、応答同一性と採番が残る。"""
    class BadCopy:
        def __deepcopy__(self, memo):
            raise ValueError("copy failure")

    class BadResponse:
        @property
        def tool_uses(self):
            raise ValueError("id extraction failure")

    request = ApiRequest(messages=[BadCopy() if failure == "copy" else object()])
    if failure not in ("copy", "json"):
        request = ApiRequest()
    response = BadResponse() if failure == "response_ids" else AssistantMessage()
    provider = _provider(response)
    real_emit = trace.emit
    if failure == "write":
        monkeypatch.setattr(trace, "emit", Mock(side_effect=OSError("disk")))
    with trace.trace_fire(_event()):
        assert trace.TracingProvider(provider).stream(request) is response
        monkeypatch.setattr(trace, "emit", real_emit)
        trace.TracingProvider(_provider()).stream(ApiRequest(system_prompt="next"))
    assert provider.stream.call_count == 1 and provider.stream.call_args.args[0] is request
    rows = _lines(fixed_io)
    assert [(r["kind"], r["call_index"]) for r in rows[-2:]] == [("request", 1), ("response", 1)]
    if failure in ("copy", "json"):
        assert rows[0]["kind"] == "response" and rows[0]["call_index"] == 0
    elif failure == "response_ids":
        assert rows[0]["kind"] == "request" and rows[0]["call_index"] == 0


@pytest.mark.parametrize("write_error", [False, True])
@pytest.mark.parametrize("error", [RuntimeError("provider"), KeyboardInterrupt("interrupt")])
def test_provider_exception_preserved(error, write_error, monkeypatch, fixed_io):
    provider = _provider(stream=Mock(side_effect=error))
    if write_error:
        monkeypatch.setattr(trace, "emit", Mock(side_effect=OSError("disk")))
    with pytest.raises(type(error)) as caught:
        with trace.trace_fire(_event()):
            trace.TracingProvider(provider).stream(ApiRequest())
    assert caught.value is error and provider.stream.call_count == 1
    assert trace._active.get() is None
    if not write_error:
        request, result = _lines(fixed_io)
        assert result["kind"] == "error" and result["trace_id"] == request["trace_id"]
        assert result["exception_type"] == type(error).__name__ and result["exception_message"] == str(error)


def test_llm1_exact_prompt_config_and_hash(monkeypatch, fixed_io):
    monkeypatch.setattr("core.llm._get_active_provider_config", lambda: ("claude_code", "url", "secret", "llm1-model"))
    prompt, response = "入力\n" * 20000, "応答\nそのまま"
    call = Mock(return_value=response)
    with trace.trace_fire(_event()):
        assert trace.trace_llm1(call, prompt, cycle_id=7, max_tokens=123,
                               temperature=1.0, image_paths=["画像.png"]) is response
    call.assert_called_once_with(prompt, max_tokens=123, temperature=1.0, image_paths=["画像.png"])
    request, result = _lines(fixed_io)
    assert request["prompt"] == prompt and request["chain_position"] is None
    assert (request["provider"], request["model"], request["cycle_id"]) == ("claude_code", "llm1-model", 7)
    assert request["max_tokens"] == 123 and request["image_paths"] == ["画像.png"]
    assert result["response_length"] == len(response)
    assert result["response_sha256"] == hashlib.sha256(response.encode("utf-8")).hexdigest()
    assert "secret" not in (fixed_io / trace.TRACE_FILE_NAME).read_text(encoding="utf-8")


def test_partial_tail_does_not_swallow_next_record(fixed_io):
    path = fixed_io / trace.TRACE_FILE_NAME
    path.write_bytes(b'{"kind":"request","prompt":"broken')
    with trace.trace_fire(_event()):
        trace.TracingProvider(_provider()).stream(ApiRequest())
    lines = path.read_text(encoding="utf-8").splitlines()
    with pytest.raises(json.JSONDecodeError):
        json.loads(lines[0])
    assert [json.loads(s)["kind"] for s in lines[1:]] == ["request", "response"]


def test_main_forced_free_wiring_and_invocation_join(monkeypatch, fixed_io):
    """main が wrapper を使わない、chain を渡さない、別 attempt を作る実装を落とす。"""
    stages = [[ToolUseBlock("forced-id", "reflect", {"echo": "first-result"})],
              [ToolUseBlock("free-id", "output_display", {"channel": "device"})]]
    harness = _fire_harness(monkeypatch, fixed_io, stages=stages,
                            proposal="1. [考える] → reflect+output_display (pe2=40, pec=0.2)")
    _run(harness[0]["observed"])
    rows = _lines(fixed_io)
    requests = rows[::2]
    assert [(r["role"], r["chain_position"], r["call_index"]) for r in requests] == [
        ("llm1", None, 0), ("llm2", 0, 1), ("llm2", 1, 2)]
    assert requests[0]["prompt"] == harness[2][0]
    for recorded, actual in zip(requests[1:], harness[3]):
        for key in ("system_prompt", "messages", "tools", "tool_choice", "max_tokens", "temperature", "image_paths"):
            assert recorded[key] == getattr(actual, key)
    assert requests[1]["system_prompt"] != requests[2]["system_prompt"]
    assert requests[1]["messages"] != requests[2]["messages"]
    assert requests[1]["tool_choice"] == {"type": "tool", "name": "reflect"}
    assert requests[2]["tool_choice"] is None
    assert {r["attempt_id"] for r in rows} == {_outward_lines(fixed_io)[0]["attempt_id"]}
    invocations = [json.loads(s) for s in (fixed_io / "metrics_events.jsonl").read_text(encoding="utf-8").splitlines()
                   if json.loads(s)["event_type"] == "tool_invocation"]
    assert {(r["attempt_id"], r["tool_id"]) for r in invocations} == {("r_1", "forced-id"), ("r_1", "free-id")}
    assert [r["tool_uses"][0]["id"] for r in rows[3::2]] == ["forced-id", "free-id"]
    assert trace._active.get() is None
    tree = ast.parse(MAIN_SOURCE)
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "ConversationRuntime"
               and any(k.arg == "provider" and isinstance(k.value, ast.Name) and k.value.id == "_rt_provider"
                       for k in n.keywords) for n in ast.walk(tree))


@pytest.mark.parametrize("mode", ["normal", "llm1_error", "not_invoked", "runtime_error", "no_candidate"])
def test_main_next_fire_after_every_exit(mode, monkeypatch, fixed_io):
    harness = _fire_harness(monkeypatch, fixed_io, mode=mode)
    for _ in range(2):
        if mode == "no_candidate":
            with pytest.raises(IndexError):
                _run(harness[0]["observed"])
        else:
            _run(harness[0]["observed"])
        assert trace._active.get() is None
    rows = _lines(fixed_io)
    starts = [r for r in rows if r["kind"] == "request" and r["role"] == "llm1"]
    assert [(r["attempt_id"], r["call_index"]) for r in starts] == [("r_1", 0), ("r_2", 0)]
    assert len({r["trace_id"] for r in rows}) == len(rows) // 2
    for request, result in zip(rows[::2], rows[1::2]):
        for key in ("trace_id", "attempt_id", "run_id", "cycle_id", "role", "chain_position", "call_index"):
            assert request[key] == result[key]


def test_main_recording_failure_preserves_behavior(monkeypatch, fixed_io):
    """記録なし・正常記録・書込失敗で prompt/G/選択/state/戻り値/呼出数を比較。"""
    plain = _fire_harness(monkeypatch, fixed_io)
    before = _run(plain[0]["plain"])
    observed = _fire_harness(monkeypatch, fixed_io)
    after = _run(observed[0]["observed"])
    monkeypatch.setattr(trace, "emit", Mock(side_effect=OSError("disk")))
    broken = _fire_harness(monkeypatch, fixed_io)
    failed = _run(broken[0]["observed"])
    assert before == after == failed
    assert plain[1] == observed[1] == broken[1]
    assert plain[2] == observed[2] == broken[2] and len(plain[2]) == 1
    assert plain[4:6] == observed[4:6] == broken[4:6]
    for h in (observed, broken):
        assert len(h[3]) == len(plain[3])
        for a, b in zip(plain[3], h[3]):
            for key in ("system_prompt", "messages", "tools", "tool_choice", "max_tokens", "temperature", "image_paths"):
                assert getattr(a, key) == getattr(b, key)
        assert h[-1].call_count == plain[-1].call_count == 1
    events = _outward_lines(fixed_io)
    assert len(events) == 2 and all(e["end"] == "completed" for e in events)


def test_main_multiple_calls_in_one_stage(monkeypatch, fixed_io):
    """段を変えず反復した runtime 呼出も、それぞれの入力と番号で残す。"""
    harness = _fire_harness(monkeypatch, fixed_io,
                            stages=[[ToolUseBlock("first", "reflect", {})], []],
                            proposal="1. [考える] → reflect (pe2=40, pec=0.2)")
    harness[0]["observed"].__globals__["_runtime"].max_iterations = 2
    _run(harness[0]["observed"])
    requests = [r for r in _lines(fixed_io) if r["kind"] == "request" and r["role"] == "llm2"]
    assert [(r["chain_position"], r["call_index"]) for r in requests] == [(0, 1), (0, 2)]
    assert requests[0]["tool_choice"] == {"type": "tool", "name": "reflect"}
    assert requests[1]["tool_choice"] is None
    assert requests[0]["messages"] != requests[1]["messages"]


def test_response_write_failure_keeps_request(monkeypatch, fixed_io):
    real_emit = trace.emit

    def emit(record):
        if record["kind"] == "response":
            raise OSError("response disk failure")
        real_emit(record)

    monkeypatch.setattr(trace, "emit", emit)
    provider = _provider()
    with trace.trace_fire(_event()):
        result = trace.TracingProvider(provider).stream(ApiRequest(system_prompt="saved"))
    assert result is provider.stream.return_value and provider.stream.call_count == 1
    assert [(r["kind"], r["system_prompt"]) for r in _lines(fixed_io)] == [("request", "saved")]


def test_llm1_config_failure_and_original_exception(monkeypatch, fixed_io):
    """設定取得失敗で入力全文が欠ける、LLM 本体の例外が置換される実装を落とす。"""
    monkeypatch.setattr("core.llm._get_active_provider_config", Mock(side_effect=ValueError("config observation")))
    error = RuntimeError("llm1 failed")
    call = Mock(side_effect=error)
    prompt = "入力全文\n" * 20000
    with trace.trace_fire(_event()):
        with pytest.raises(RuntimeError) as caught:
            trace.trace_llm1(call, prompt, cycle_id=3, max_tokens=123,
                             temperature=1.0, image_paths=None)
    assert caught.value is error and call.call_count == 1
    request, result = _lines(fixed_io)
    assert request["kind"] == "request" and request["prompt"] == prompt
    assert request["provider"] is None and request["model"] is None
    assert request["provider_config_unavailable"] is True
    assert (request["max_tokens"], request["temperature"], request["image_paths"]) == (123, 1.0, None)
    assert result["kind"] == "error" and result["exception_message"] == "llm1 failed"
    assert request["trace_id"] == result["trace_id"]
