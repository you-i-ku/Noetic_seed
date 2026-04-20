"""Runtime Core 統合テスト: registry / hooks / session / conversation / providers。

1 ファイルにまとめて検証。使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_runtime_core.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from core.providers.base import (
    ApiRequest, AssistantMessage, BaseProvider, ToolUseBlock,
)
from core.providers.openai_compat import OpenAIProvider
from core.providers.anthropic import AnthropicProvider
from core.runtime.conversation import ConversationRuntime
from core.runtime.hooks import HookRunner, HookRunResult
from core.runtime.permissions import (
    PermissionDecision, PermissionEnforcer, PermissionMode, PermissionRules,
)
from core.runtime.registry import ToolRegistry
from core.runtime.session import Session
from core.runtime.tool_schema import ToolSpec


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# Registry
# ============================================================

def _spec(name, perm, out_prefix=""):
    return ToolSpec(
        name=name, description=f"{name} test",
        input_schema={"type": "object"},
        required_permission=perm,
        handler=lambda inp: f"{out_prefix}{name}({inp})",
    )


def test_registry_basic():
    print("== Registry: 登録/取得/実行/MCP名前正規化 ==")
    reg = ToolRegistry()
    reg.register(_spec("read_file", PermissionMode.READ_ONLY))
    reg.register(_spec("bash", PermissionMode.DANGER_FULL_ACCESS))
    r = [
        _assert(reg.has("read_file"), "has"),
        _assert(reg.get("bash").name == "bash", "get"),
        _assert("read_file" in reg.execute("read_file", {}), "execute"),
        _assert(ToolRegistry.mcp_tool_name("slack-server", "post")
                == "mcp__slack_server__post", "mcp名前正規化"),
        _assert(reg.is_mcp_tool("mcp__x__y") and
                not reg.is_mcp_tool("bash"), "is_mcp_tool"),
    ]
    try:
        reg.execute("none", {})
        r.append(_assert(False, "未登録で ValueError"))
    except ValueError:
        r.append(_assert(True, "未登録で ValueError"))
    return all(r)


def test_registry_filter():
    print("== Registry: list() フィルタ ==")
    reg = ToolRegistry()
    reg.register(_spec("read_file", PermissionMode.READ_ONLY))
    reg.register(_spec("write_file", PermissionMode.WORKSPACE_WRITE))
    reg.register(_spec("bash", PermissionMode.DANGER_FULL_ACCESS))
    r = [
        _assert(len(reg.list(max_permission=PermissionMode.READ_ONLY)) == 1,
                "RO: 1件"),
        _assert(len(reg.list(max_permission=PermissionMode.WORKSPACE_WRITE)) == 2,
                "WW: 2件"),
        _assert(len(reg.list()) == 3, "全件"),
        _assert(len(reg.list(allowlist=["bash"])) == 1, "allowlist"),
        _assert(len(reg.list(denylist=["bash"])) == 2, "denylist"),
    ]
    return all(r)


# ============================================================
# Hooks
# ============================================================

def test_hooks_basic():
    print("== Hooks: pre/post/failure 基本 ==")
    h = HookRunner()
    h.register_pre(lambda n, i: HookRunResult.allow(messages=["pre"]))
    h.register_post(lambda n, i, o: HookRunResult.allow(messages=["post"]))
    h.register_failure(lambda n, i, e: HookRunResult.allow(messages=["fail"]))
    r_pre = h.run_pre_tool_use("x", {})
    r_post = h.run_post_tool_use("x", {}, "out")
    r_fail = h.run_post_tool_use_failure("x", {}, "err")
    r = [
        _assert("pre" in r_pre.messages, "pre 呼出"),
        _assert("post" in r_post.messages, "post 呼出"),
        _assert("fail" in r_fail.messages, "failure 呼出"),
    ]
    return all(r)


def test_hooks_deny_stops():
    print("== Hooks: denied で後続停止 ==")
    h = HookRunner()
    called = []
    h.register_pre(lambda n, i: (called.append("h1"),
                                 HookRunResult.deny(messages=["no"]))[1])
    h.register_pre(lambda n, i: (called.append("h2"),
                                 HookRunResult.allow())[1])
    r = h.run_pre_tool_use("x", {})
    return all([
        _assert(called == ["h1"], "h2 は呼ばれない"),
        _assert(r.denied, "denied=True"),
    ])


def test_hooks_updated_input():
    print("== Hooks: updated_input で入力書換 ==")
    h = HookRunner()
    received = []
    h.register_pre(lambda n, i: HookRunResult(
        updated_input={**i, "extra": 1}
    ))
    h.register_pre(lambda n, i: (received.append(i),
                                 HookRunResult.allow())[1])
    r = h.run_pre_tool_use("x", {"a": 1})
    return all([
        _assert(received[0].get("extra") == 1, "後続 handler に伝播"),
        _assert(r.updated_input and r.updated_input.get("extra") == 1,
                "最終 updated_input"),
    ])


def test_hooks_exception():
    print("== Hooks: 例外で failed=True ==")
    h = HookRunner()
    def broken(n, i): raise RuntimeError("boom")
    h.register_pre(broken)
    r = h.run_pre_tool_use("x", {})
    return all([
        _assert(r.failed, "failed"),
        _assert(not r.denied, "denied ではない"),
    ])


# ============================================================
# Session
# ============================================================

def test_session_anthropic_serialize():
    print("== Session: Anthropic serialize ==")
    s = Session()
    s.push_user_text("hello")
    s.push_assistant_message(AssistantMessage(
        text="I'll help",
        tool_uses=[ToolUseBlock(id="t1", name="read_file",
                                input={"path": "a.txt"})],
    ))
    s.push_tool_result("t1", "CONTENT")
    out = s.serialize_for_anthropic()
    r = [
        _assert(len(out) == 3, "3 メッセージ"),
        _assert(out[0]["role"] == "user", "user"),
        _assert(out[1]["role"] == "assistant", "assistant"),
        _assert(any(b.get("type") == "tool_use"
                    for b in out[1]["content"]),
                "tool_use block"),
        _assert(out[2]["content"][0]["type"] == "tool_result",
                "tool_result block"),
    ]
    return all(r)


def test_session_openai_serialize():
    print("== Session: OpenAI serialize (変換) ==")
    s = Session()
    s.push_user_text("hello")
    s.push_assistant_message(AssistantMessage(
        tool_uses=[ToolUseBlock(id="t1", name="read_file",
                                input={"path": "a.txt"})],
    ))
    s.push_tool_result("t1", "CONTENT")
    out = s.serialize_for_openai()
    r = [
        _assert(len(out) == 3, "3 メッセージ"),
        _assert(out[1]["role"] == "assistant"
                and "tool_calls" in out[1], "tool_calls 形式"),
        _assert(out[1]["tool_calls"][0]["function"]["name"] == "read_file",
                "tool name"),
        _assert(out[2]["role"] == "tool"
                and out[2]["tool_call_id"] == "t1",
                "role=tool + tool_call_id"),
    ]
    return all(r)


# ============================================================
# Providers (httpx mock)
# ============================================================

def _patch_post(response_json: dict, capture: dict = None):
    original = httpx.post
    def mock(url, headers=None, json=None, timeout=None, **kw):
        if capture is not None:
            capture.update({"url": url, "headers": headers or {},
                            "json": json})
        return httpx.Response(status_code=200, json=response_json,
                              request=httpx.Request("POST", url))
    httpx.post = mock
    return original


def _restore_post(original):
    httpx.post = original


def test_openai_provider():
    print("== OpenAIProvider: tool_calls parse ==")
    resp = {
        "choices": [{
            "message": {"role": "assistant", "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file",
                                         "arguments": '{"path":"a.txt"}'}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    capture = {}
    original = _patch_post(resp, capture)
    try:
        p = OpenAIProvider(model="gpt-4", api_key="k",
                           base_url="https://x/v1")
        msg = p.stream(ApiRequest(
            system_prompt="sys",
            messages=[{"role": "user", "content": "x"}],
            tools=[{"type": "function",
                    "function": {"name": "read_file",
                                 "description": "d",
                                 "parameters": {}}}],
        ))
    finally:
        _restore_post(original)
    r = [
        _assert(msg.stop_reason == "tool_use", "stop_reason"),
        _assert(len(msg.tool_uses) == 1
                and msg.tool_uses[0].input == {"path": "a.txt"},
                "tool_use input parse"),
        _assert(capture["json"]["messages"][0]["role"] == "system",
                "system 先頭"),
        _assert("tools" in capture["json"], "tools payload"),
    ]
    return all(r)


def test_anthropic_provider():
    print("== AnthropicProvider: tool_use ContentBlock ==")
    resp = {
        "content": [
            {"type": "text", "text": "Let me read"},
            {"type": "tool_use", "id": "tu1", "name": "read_file",
             "input": {"path": "a.txt"}},
        ],
        "stop_reason": "tool_use",
    }
    capture = {}
    original = _patch_post(resp, capture)
    try:
        p = AnthropicProvider(model="claude-sonnet-4-6", api_key="k")
        msg = p.stream(ApiRequest(
            system_prompt="sys",
            messages=[{"role": "user", "content": "x"}],
        ))
    finally:
        _restore_post(original)
    r = [
        _assert(msg.text == "Let me read", "text"),
        _assert(len(msg.tool_uses) == 1, "1 tool_use"),
        _assert(msg.tool_uses[0].input == {"path": "a.txt"}, "input"),
        _assert(capture["json"]["system"] == "sys",
                "Anthropic system は別フィールド"),
        _assert(capture["headers"].get("x-api-key") == "k",
                "x-api-key ヘッダ"),
    ]
    return all(r)


# ============================================================
# ConversationRuntime (MockProvider end-to-end)
# ============================================================

class MockProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, responses):
        super().__init__(model="mock", api_key="")
        self._responses = list(responses)
        self.calls = []

    def supports_tool_use(self): return True

    def stream(self, req):
        self.calls.append(req)
        if self._responses:
            return self._responses.pop(0)
        return AssistantMessage(stop_reason="end_turn")


def _make_reg():
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="read_file", description="",
        input_schema={"type": "object"},
        required_permission=PermissionMode.READ_ONLY,
        handler=lambda i: f"READ {i.get('path','?')}",
    ))
    reg.register(ToolSpec(
        name="bash", description="",
        input_schema={"type": "object"},
        required_permission=PermissionMode.DANGER_FULL_ACCESS,
        handler=lambda i: f"RAN {i.get('command','?')}",
    ))
    return reg


def test_conv_text_only():
    print("== Conv: text only response → completed ==")
    p = MockProvider([AssistantMessage(text="hi", stop_reason="end_turn")])
    rt = ConversationRuntime(
        provider=p, tool_registry=_make_reg(),
        permission_enforcer=PermissionEnforcer(PermissionMode.ALLOW),
        max_iterations=5,
    )
    s = rt.run_turn("hello")
    return all([
        _assert(s.finish_reason == "completed", "completed"),
        _assert(s.iterations == 1, "1 回"),
        _assert(len(s.tool_invocations) == 0, "tool 0"),
    ])


def test_conv_single_tool():
    print("== Conv: tool 1 回 + 最終 text ==")
    p = MockProvider([
        AssistantMessage(tool_uses=[ToolUseBlock("t1", "read_file",
                                                 {"path": "a.txt"})],
                         stop_reason="tool_use"),
        AssistantMessage(text="done", stop_reason="end_turn"),
    ])
    rt = ConversationRuntime(
        provider=p, tool_registry=_make_reg(),
        permission_enforcer=PermissionEnforcer(PermissionMode.ALLOW),
        max_iterations=5,
    )
    s = rt.run_turn("read a")
    return all([
        _assert(s.iterations == 2, "2 iter"),
        _assert(len(s.tool_invocations) == 1, "1 tool"),
        _assert("READ a.txt" in s.tool_invocations[0].output, "output"),
        _assert(s.finish_reason == "completed", "completed"),
    ])


def test_conv_permission_deny():
    print("== Conv: rules.deny で実行されない ==")
    p = MockProvider([
        AssistantMessage(tool_uses=[ToolUseBlock("t1", "bash",
                                                 {"command": "x"})],
                         stop_reason="tool_use"),
        AssistantMessage(text="ok", stop_reason="end_turn"),
    ])
    enf = PermissionEnforcer(PermissionMode.ALLOW,
                             PermissionRules(deny=["bash"]))
    rt = ConversationRuntime(provider=p, tool_registry=_make_reg(),
                              permission_enforcer=enf, max_iterations=5)
    s = rt.run_turn("x")
    rec = s.tool_invocations[0]
    return all([
        _assert(rec.permission_decision == "deny", "DENY"),
        _assert(rec.is_error, "error"),
        _assert("RAN" not in rec.output, "実行されてない"),
    ])


def test_conv_approval_callback():
    print("== Conv: ASK → approval callback ==")
    called = []
    p = MockProvider([
        AssistantMessage(tool_uses=[ToolUseBlock("t1", "bash",
                                                 {"command": "ls"})],
                         stop_reason="tool_use"),
        AssistantMessage(text="ok", stop_reason="end_turn"),
    ])
    rt = ConversationRuntime(
        provider=p, tool_registry=_make_reg(),
        permission_enforcer=PermissionEnforcer(PermissionMode.PROMPT),
        approval_callback=lambda n, i, m: (called.append(n), True)[1],
        max_iterations=5,
    )
    s = rt.run_turn("ls")
    return all([
        _assert(s.tool_invocations[0].permission_decision == "ask", "ASK"),
        _assert(called == ["bash"], "callback 呼出"),
        _assert("RAN ls" in s.tool_invocations[0].output, "実行された"),
    ])


def test_conv_post_hook_eval():
    print("== Conv: post hook で E値評価シミュレート ==")
    evals = []
    h = HookRunner()
    h.register_post(lambda n, i, o: (evals.append(len(o)),
                                     HookRunResult.allow(
                                         messages=[f"len={len(o)}"]))[1])
    p = MockProvider([
        AssistantMessage(tool_uses=[ToolUseBlock("t1", "read_file",
                                                 {"path": "x"})],
                         stop_reason="tool_use"),
        AssistantMessage(text="ok", stop_reason="end_turn"),
    ])
    rt = ConversationRuntime(
        provider=p, tool_registry=_make_reg(), hook_runner=h,
        permission_enforcer=PermissionEnforcer(PermissionMode.ALLOW),
        max_iterations=5,
    )
    s = rt.run_turn("x")
    return all([
        _assert(len(evals) == 1 and evals[0] > 0, "post hook 実行"),
        _assert(any("len=" in m for m in s.tool_invocations[0].post_hook_messages),
                "messages 伝播"),
    ])


def test_conv_max_iterations():
    print("== Conv: max_iterations 到達 ==")
    p = MockProvider([
        AssistantMessage(tool_uses=[ToolUseBlock(f"t{i}", "read_file",
                                                 {"path": "x"})],
                         stop_reason="tool_use")
        for i in range(5)
    ])
    rt = ConversationRuntime(
        provider=p, tool_registry=_make_reg(),
        permission_enforcer=PermissionEnforcer(PermissionMode.ALLOW),
        max_iterations=3,
    )
    s = rt.run_turn("loop")
    return all([
        _assert(s.iterations == 3, "3 回で頭打ち"),
        _assert(s.finish_reason == "max_iterations", "max_iterations"),
    ])


def test_conv_tool_exception():
    print("== Conv: tool 例外 → post_tool_use_failure ==")
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="broken", description="",
        input_schema={"type": "object"},
        required_permission=PermissionMode.READ_ONLY,
        handler=lambda i: (_ for _ in ()).throw(RuntimeError("boom")),
    ))
    fails = []
    h = HookRunner()
    h.register_failure(lambda n, i, e:
        (fails.append(e), HookRunResult.allow())[1])
    p = MockProvider([
        AssistantMessage(tool_uses=[ToolUseBlock("t1", "broken", {})],
                         stop_reason="tool_use"),
        AssistantMessage(text="recovered", stop_reason="end_turn"),
    ])
    rt = ConversationRuntime(
        provider=p, tool_registry=reg, hook_runner=h,
        permission_enforcer=PermissionEnforcer(PermissionMode.ALLOW),
        max_iterations=5,
    )
    s = rt.run_turn("x")
    return all([
        _assert(s.tool_invocations[0].is_error, "error"),
        _assert("boom" in s.tool_invocations[0].output, "例外文字列"),
        _assert(len(fails) == 1, "failure hook"),
    ])


def main():
    tests = [
        test_registry_basic, test_registry_filter,
        test_hooks_basic, test_hooks_deny_stops,
        test_hooks_updated_input, test_hooks_exception,
        test_session_anthropic_serialize, test_session_openai_serialize,
        test_openai_provider, test_anthropic_provider,
        test_conv_text_only, test_conv_single_tool,
        test_conv_permission_deny, test_conv_approval_callback,
        test_conv_post_hook_eval, test_conv_max_iterations,
        test_conv_tool_exception,
    ]
    print(f"Running {len(tests)} test groups...\n")
    passed = 0
    for t in tests:
        if t():
            passed += 1
        print()
    print(f"=== Result: {passed}/{len(tests)} test groups passed ===")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
