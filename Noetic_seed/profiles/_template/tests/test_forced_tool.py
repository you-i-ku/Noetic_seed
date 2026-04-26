"""ConversationRuntime.run_turn_with_forced_tool (X-guided モード) テスト。

Phase 4 Step E-2a: controller 選択 + LLM args 生成のみ の強制 tool 実行を検証。

網羅項目:
  - ApiRequest.tool_choice 追加の provider 伝搬 (OpenAI / Anthropic)
  - tool_choice 省略時は既存挙動 (OpenAI="auto" / Anthropic=未指定)
  - ConversationRuntime.run_turn_with_forced_tool の tool 強制実行
  - 指定 tool 以外を LLM が返してもエラーにせず (hook 経由で扱う)
  - hook が通常通り発火
  - user_input が Session に積まれる

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_forced_tool.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from core.providers.anthropic import AnthropicProvider
from core.providers.base import ApiRequest, AssistantMessage, ToolUseBlock, BaseProvider
from core.providers.openai_compat import OpenAIProvider
from core.runtime.conversation import ConversationRuntime
from core.runtime.hooks import HookRunner, HookRunResult
from core.runtime.permissions import PermissionEnforcer, PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _patch_post(response_json: dict, capture: dict = None):
    original = httpx.post
    def mock(url, headers=None, json=None, timeout=None, **kw):
        if capture is not None:
            capture.update({"url": url, "json": json})
        return httpx.Response(status_code=200, json=response_json,
                              request=httpx.Request("POST", url))
    httpx.post = mock
    return original


def _restore_post(original):
    httpx.post = original


# ============================================================
# Provider: tool_choice 伝搬
# ============================================================

def test_openai_tool_choice_default():
    print("== OpenAI: tool_choice 省略 → 'auto' ==")
    resp = {"choices": [{"message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}]}
    capture = {}
    original = _patch_post(resp, capture)
    try:
        p = OpenAIProvider(model="m", api_key="k", base_url="https://x/v1")
        p.stream(ApiRequest(
            messages=[{"role": "user", "content": "x"}],
            tools=[{"type": "function",
                    "function": {"name": "read_file", "description": "d",
                                 "parameters": {}}}],
        ))
    finally:
        _restore_post(original)
    return _assert(capture["json"].get("tool_choice") == "auto",
                   f"tool_choice='auto' (実={capture['json'].get('tool_choice')})")


def test_openai_tool_choice_forced():
    print("== OpenAI: tool_choice 指定 → そのまま payload 反映 ==")
    resp = {"choices": [{"message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}]}
    capture = {}
    original = _patch_post(resp, capture)
    try:
        p = OpenAIProvider(model="m", api_key="k", base_url="https://x/v1")
        p.stream(ApiRequest(
            messages=[{"role": "user", "content": "x"}],
            tools=[{"type": "function",
                    "function": {"name": "read_file", "description": "d",
                                 "parameters": {}}}],
            tool_choice={"type": "function",
                         "function": {"name": "read_file"}},
        ))
    finally:
        _restore_post(original)
    tc = capture["json"].get("tool_choice")
    return all([
        _assert(isinstance(tc, dict), "dict で送信"),
        _assert(tc.get("type") == "function", "type=function"),
        _assert(tc.get("function", {}).get("name") == "read_file",
                "function.name=read_file"),
    ])


def test_anthropic_tool_choice_default():
    print("== Anthropic: tool_choice 省略 → payload に含まれない ==")
    resp = {"content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn"}
    capture = {}
    original = _patch_post(resp, capture)
    try:
        p = AnthropicProvider(model="m", api_key="k",
                              base_url="https://api.anthropic.com")
        p.stream(ApiRequest(
            messages=[{"role": "user", "content": "x"}],
            tools=[{"name": "read_file", "description": "d",
                    "input_schema": {}}],
        ))
    finally:
        _restore_post(original)
    return _assert("tool_choice" not in capture["json"],
                   "tool_choice key 無し")


def test_anthropic_tool_choice_forced():
    print("== Anthropic: tool_choice 指定 → payload に含む ==")
    resp = {"content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn"}
    capture = {}
    original = _patch_post(resp, capture)
    try:
        p = AnthropicProvider(model="m", api_key="k",
                              base_url="https://api.anthropic.com")
        p.stream(ApiRequest(
            messages=[{"role": "user", "content": "x"}],
            tools=[{"name": "read_file", "description": "d",
                    "input_schema": {}}],
            tool_choice={"type": "tool", "name": "read_file"},
        ))
    finally:
        _restore_post(original)
    tc = capture["json"].get("tool_choice", {})
    return all([
        _assert(tc.get("type") == "tool", "type=tool"),
        _assert(tc.get("name") == "read_file", "name=read_file"),
    ])


# ============================================================
# ConversationRuntime.run_turn_with_forced_tool
# ============================================================

class _FakeProvider(BaseProvider):
    """テスト用 mock provider。事前定義した AssistantMessage を返す。"""

    name = "openai_compat"  # tool_choice 形式を OpenAI として検証

    def __init__(self, assistant_msg: AssistantMessage,
                 capture: dict = None):
        super().__init__(model="fake", api_key="", base_url="")
        self._msg = assistant_msg
        self._capture = capture if capture is not None else {}

    def stream(self, request: ApiRequest) -> AssistantMessage:
        self._capture["system_prompt"] = request.system_prompt
        self._capture["messages"] = list(request.messages)
        self._capture["tools"] = list(request.tools)
        self._capture["tool_choice"] = request.tool_choice
        return self._msg


class _FakeAnthropicProvider(_FakeProvider):
    name = "anthropic"


def _spec(name, handler=None):
    return ToolSpec(
        name=name, description=f"{name} spec",
        input_schema={"type": "object"},
        required_permission=PermissionMode.WORKSPACE_WRITE,
        handler=handler or (lambda inp: f"{name}({inp})"),
    )


def test_forced_tool_basic():
    print("== run_turn_with_forced_tool: tool 強制実行 (単一 tool + required) ==")
    # LLM 応答: read_file tool_use を返す
    fake_msg = AssistantMessage(
        text="",
        tool_uses=[ToolUseBlock(id="c1", name="read_file",
                                 input={"path": "a.txt"})],
        stop_reason="tool_use",
    )
    capture = {}
    provider = _FakeProvider(fake_msg, capture)
    reg = ToolRegistry()
    reg.register(_spec("read_file"))
    reg.register(_spec("other_tool"))  # filter 効果検証用の余分 tool
    rt = ConversationRuntime(
        provider=provider, tool_registry=reg, max_iterations=1,
        permission_enforcer=PermissionEnforcer(
            mode=PermissionMode.WORKSPACE_WRITE),
    )
    summary = rt.run_turn_with_forced_tool(
        forced_tool_name="read_file",
        user_input="please read a.txt",
    )
    tc = capture.get("tool_choice")
    tools = capture.get("tools", [])
    tool_names = [
        t.get("function", {}).get("name") for t in tools
    ]
    return all([
        _assert(len(summary.tool_invocations) == 1, "1 tool 実行"),
        _assert(summary.tool_invocations[0].tool_name == "read_file",
                "tool name=read_file"),
        _assert("a.txt" in summary.tool_invocations[0].output,
                "tool output が Session に積まれた"),
        _assert(tc == "required",
                f"OpenAI tool_choice='required' (実={tc!r})"),
        _assert(len(tools) == 1,
                f"tools payload 1 個に絞られる (実={len(tools)})"),
        _assert(tool_names == ["read_file"],
                f"forced tool のみ送信 (実={tool_names})"),
    ])


def test_forced_tool_user_input_in_session():
    print("== run_turn_with_forced_tool: user_input が Session に積まれる ==")
    fake_msg = AssistantMessage(
        text="",
        tool_uses=[ToolUseBlock(id="c1", name="read_file",
                                 input={"path": "x"})],
        stop_reason="tool_use",
    )
    provider = _FakeProvider(fake_msg)
    reg = ToolRegistry()
    reg.register(_spec("read_file"))
    rt = ConversationRuntime(
        provider=provider, tool_registry=reg, max_iterations=1,
        permission_enforcer=PermissionEnforcer(
            mode=PermissionMode.WORKSPACE_WRITE),
    )
    rt.run_turn_with_forced_tool(
        forced_tool_name="read_file",
        user_input="candidate reason text",
    )
    first = rt.session.messages[0]
    return all([
        _assert(first["role"] == "user", "role=user"),
        _assert(first["content"][0]["text"] == "candidate reason text",
                "user_input 保持"),
    ])


def test_forced_tool_anthropic_choice_format():
    print("== run_turn_with_forced_tool: Anthropic 形式 tool_choice ==")
    fake_msg = AssistantMessage(
        text="",
        tool_uses=[ToolUseBlock(id="c1", name="read_file",
                                 input={"path": "x"})],
        stop_reason="tool_use",
    )
    capture = {}
    provider = _FakeAnthropicProvider(fake_msg, capture)
    reg = ToolRegistry()
    reg.register(_spec("read_file"))
    rt = ConversationRuntime(
        provider=provider, tool_registry=reg, max_iterations=1,
        permission_enforcer=PermissionEnforcer(
            mode=PermissionMode.WORKSPACE_WRITE),
    )
    rt.run_turn_with_forced_tool(forced_tool_name="read_file",
                                  user_input="x")
    tc = capture.get("tool_choice", {})
    return all([
        _assert(tc.get("type") == "tool", "type=tool (Anthropic 形式)"),
        _assert(tc.get("name") == "read_file", "name=read_file"),
    ])


def test_forced_tool_no_tool_returned():
    print("== run_turn_with_forced_tool: LLM が tool を返さなかった ==")
    # LLM が text だけ返す場合 (tool_choice 強制しても無視するケース)
    fake_msg = AssistantMessage(
        text="申し訳ないがツール使えない", tool_uses=[],
        stop_reason="end_turn",
    )
    provider = _FakeProvider(fake_msg)
    reg = ToolRegistry()
    reg.register(_spec("read_file"))
    rt = ConversationRuntime(
        provider=provider, tool_registry=reg, max_iterations=1,
        permission_enforcer=PermissionEnforcer(
            mode=PermissionMode.WORKSPACE_WRITE),
    )
    summary = rt.run_turn_with_forced_tool(
        forced_tool_name="read_file", user_input="x")
    return all([
        _assert(summary.finish_reason == "no_tool",
                f"finish_reason=no_tool (実={summary.finish_reason})"),
        _assert(len(summary.tool_invocations) == 0, "tool 実行ゼロ"),
    ])


def test_forced_tool_hooks_fired():
    print("== run_turn_with_forced_tool: pre/post hook 発火 ==")
    fake_msg = AssistantMessage(
        text="",
        tool_uses=[ToolUseBlock(id="c1", name="read_file",
                                 input={"path": "x",
                                        "tool_intent": "",
                                        "tool_expected_outcome": "",
                                        "message": ""})],
        stop_reason="tool_use",
    )
    provider = _FakeProvider(fake_msg)
    reg = ToolRegistry()
    reg.register(_spec("read_file"))

    pre_hits = {"n": 0}
    post_hits = {"n": 0}

    def _pre(tool_name, tool_input):
        pre_hits["n"] += 1
        return HookRunResult.allow()

    def _post(tool_name, tool_input, output):
        post_hits["n"] += 1
        return HookRunResult.allow()

    runner = HookRunner()
    runner.register_pre(_pre)
    runner.register_post(_post)

    rt = ConversationRuntime(
        provider=provider, tool_registry=reg,
        hook_runner=runner, max_iterations=1,
        permission_enforcer=PermissionEnforcer(
            mode=PermissionMode.WORKSPACE_WRITE),
    )
    rt.run_turn_with_forced_tool(
        forced_tool_name="read_file", user_input="x")
    return all([
        _assert(pre_hits["n"] == 1, "pre hook 1 回"),
        _assert(post_hits["n"] == 1, "post hook 1 回"),
    ])


def test_forced_tool_system_prompt_passed():
    print("== run_turn_with_forced_tool: system_prompt が provider に渡る ==")
    fake_msg = AssistantMessage(
        text="",
        tool_uses=[ToolUseBlock(id="c1", name="read_file",
                                 input={"path": "x"})],
        stop_reason="tool_use",
    )
    capture = {}
    provider = _FakeProvider(fake_msg, capture)
    reg = ToolRegistry()
    reg.register(_spec("read_file"))
    rt = ConversationRuntime(
        provider=provider, tool_registry=reg, max_iterations=1,
        system_prompt="SYSTEM_TEST",
        permission_enforcer=PermissionEnforcer(
            mode=PermissionMode.WORKSPACE_WRITE),
    )
    rt.run_turn_with_forced_tool(
        forced_tool_name="read_file", user_input="x")
    return _assert(capture.get("system_prompt") == "SYSTEM_TEST",
                   "system_prompt 伝搬")


# ============================================================
# 段階11-D Phase 8 hotfix ② (案 Y、ゆう原案 2026-04-26):
# forced_system_prompt の二重構造 (iteration 0 = 身体の意思 /
# iteration 1+ = 脳の探索) 検証
# ============================================================

class _MultiCallProvider(BaseProvider):
    """複数回 stream 呼出を別 capture として list に蓄積する mock。"""
    name = "openai_compat"

    def __init__(self, msgs: list, captures: list):
        super().__init__(model="fake", api_key="", base_url="")
        self._msgs = list(msgs)
        self._captures = captures
        self._idx = 0

    def stream(self, request: ApiRequest) -> AssistantMessage:
        cap = {
            "system_prompt": request.system_prompt,
            "tools": list(request.tools),
            "tool_choice": request.tool_choice,
        }
        self._captures.append(cap)
        msg = self._msgs[min(self._idx, len(self._msgs) - 1)]
        self._idx += 1
        return msg


def test_forced_tool_iteration_0_uses_forced_system_prompt():
    """forced_system_prompt 渡すと iteration 0 で req.system_prompt が override される (案 Y)"""
    print("== forced ② (案 Y): iteration 0 で forced_system_prompt が使われる ==")
    fake_msg = AssistantMessage(
        text="", tool_uses=[ToolUseBlock(id="c1", name="read_file",
                                          input={"path": "x"})],
        stop_reason="tool_use",
    )
    capture = {}
    provider = _FakeProvider(fake_msg, capture)
    reg = ToolRegistry()
    reg.register(_spec("read_file"))
    reg.register(_spec("other_tool"))
    rt = ConversationRuntime(
        provider=provider, tool_registry=reg, max_iterations=1,
        system_prompt="DEFAULT_PROMPT_ALL_TOOLS",
        permission_enforcer=PermissionEnforcer(
            mode=PermissionMode.WORKSPACE_WRITE),
    )
    rt.run_turn_with_forced_tool(
        forced_tool_name="read_file", user_input="x",
        forced_system_prompt="FORCED_PROMPT_ONE_TOOL",
    )
    return _assert(
        capture.get("system_prompt") == "FORCED_PROMPT_ONE_TOOL",
        "iteration 0 で forced_system_prompt 使用",
    )


def test_forced_tool_no_override_uses_default():
    """forced_system_prompt None なら従来挙動 (回帰ガード)"""
    print("== forced ② (案 Y): forced_system_prompt None で従来挙動 ==")
    fake_msg = AssistantMessage(
        text="", tool_uses=[ToolUseBlock(id="c1", name="read_file",
                                          input={"path": "x"})],
        stop_reason="tool_use",
    )
    capture = {}
    provider = _FakeProvider(fake_msg, capture)
    reg = ToolRegistry()
    reg.register(_spec("read_file"))
    rt = ConversationRuntime(
        provider=provider, tool_registry=reg, max_iterations=1,
        system_prompt="DEFAULT_PROMPT",
        permission_enforcer=PermissionEnforcer(
            mode=PermissionMode.WORKSPACE_WRITE),
    )
    rt.run_turn_with_forced_tool(
        forced_tool_name="read_file", user_input="x",
    )  # forced_system_prompt 省略
    return _assert(
        capture.get("system_prompt") == "DEFAULT_PROMPT",
        "forced_system_prompt 不在で self.system_prompt 維持 (回帰)",
    )


def test_forced_tool_iteration_1_returns_to_default():
    """max_iterations=2 で iteration 0 が forced、iteration 1 が default (案 Y 二重構造)"""
    print("== forced ② (案 Y): iteration 1+ で default system_prompt + filter 解除 ==")
    msgs = [
        AssistantMessage(
            text="", tool_uses=[ToolUseBlock(id="c1", name="read_file",
                                              input={"path": "x"})],
            stop_reason="tool_use",
        ),
        AssistantMessage(
            text="", tool_uses=[ToolUseBlock(id="c2", name="other_tool",
                                              input={"y": 1})],
            stop_reason="tool_use",
        ),
    ]
    captures: list = []
    provider = _MultiCallProvider(msgs, captures)
    reg = ToolRegistry()
    reg.register(_spec("read_file"))
    reg.register(_spec("other_tool"))
    rt = ConversationRuntime(
        provider=provider, tool_registry=reg, max_iterations=2,
        system_prompt="DEFAULT_PROMPT_ALL_TOOLS",
        permission_enforcer=PermissionEnforcer(
            mode=PermissionMode.WORKSPACE_WRITE),
    )
    rt.run_turn_with_forced_tool(
        forced_tool_name="read_file", user_input="x",
        forced_system_prompt="FORCED_PROMPT_ONE_TOOL",
    )
    # iteration 0: forced prompt + tool 1 個 + tool_choice="required"
    cap0 = captures[0] if len(captures) > 0 else {}
    cap1 = captures[1] if len(captures) > 1 else {}
    iter0_tools = cap0.get("tools", [])
    iter1_tools = cap1.get("tools", [])
    return all([
        _assert(len(captures) == 2, f"2 iterations 実行 (実={len(captures)})"),
        _assert(cap0.get("system_prompt") == "FORCED_PROMPT_ONE_TOOL",
                "iter 0: forced system_prompt"),
        _assert(len(iter0_tools) == 1,
                f"iter 0: tools 1 個 (実={len(iter0_tools)})"),
        _assert(cap0.get("tool_choice") == "required",
                "iter 0: tool_choice=required"),
        _assert(cap1.get("system_prompt") == "DEFAULT_PROMPT_ALL_TOOLS",
                "iter 1+: default system_prompt に戻る (脳の探索)"),
        _assert(len(iter1_tools) == 2,
                f"iter 1+: 全 tool 視界 (実={len(iter1_tools)})"),
        _assert(cap1.get("tool_choice") is None,
                f"iter 1+: tool_choice 解除 (実={cap1.get('tool_choice')!r})"),
    ])


def test_forced_tool_iteration_0_filter_active():
    """iteration 0 で filter が forced_tool 1 個に絞られる (function calling spec 経路の確認)"""
    print("== forced ② (案 Y): iteration 0 の tools payload も 1 個に絞られる ==")
    fake_msg = AssistantMessage(
        text="", tool_uses=[ToolUseBlock(id="c1", name="read_file",
                                          input={"path": "x"})],
        stop_reason="tool_use",
    )
    capture = {}
    provider = _FakeProvider(fake_msg, capture)
    reg = ToolRegistry()
    reg.register(_spec("read_file"))
    reg.register(_spec("other_tool"))
    reg.register(_spec("third_tool"))
    rt = ConversationRuntime(
        provider=provider, tool_registry=reg, max_iterations=1,
        system_prompt="DEFAULT_PROMPT",
        permission_enforcer=PermissionEnforcer(
            mode=PermissionMode.WORKSPACE_WRITE),
    )
    rt.run_turn_with_forced_tool(
        forced_tool_name="read_file", user_input="x",
        forced_system_prompt="FORCED_PROMPT",
    )
    tools = capture.get("tools", [])
    tool_names = [t.get("function", {}).get("name") for t in tools]
    return all([
        _assert(len(tools) == 1, f"tools 1 個 (実={len(tools)})"),
        _assert(tool_names == ["read_file"],
                f"forced tool のみ (実={tool_names})"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("OpenAI: tool_choice default", test_openai_tool_choice_default),
        ("OpenAI: tool_choice forced", test_openai_tool_choice_forced),
        ("Anthropic: tool_choice default", test_anthropic_tool_choice_default),
        ("Anthropic: tool_choice forced", test_anthropic_tool_choice_forced),
        ("forced: 基本 tool 強制", test_forced_tool_basic),
        ("forced: user_input session", test_forced_tool_user_input_in_session),
        ("forced: Anthropic 形式", test_forced_tool_anthropic_choice_format),
        ("forced: tool 無返却", test_forced_tool_no_tool_returned),
        ("forced: hooks 発火", test_forced_tool_hooks_fired),
        ("forced: system_prompt 伝搬", test_forced_tool_system_prompt_passed),
        ("forced ② (案Y): iter0 forced_system_prompt 使用",
         test_forced_tool_iteration_0_uses_forced_system_prompt),
        ("forced ② (案Y): override なし回帰",
         test_forced_tool_no_override_uses_default),
        ("forced ② (案Y): iter1+ default 復帰 (脳の探索)",
         test_forced_tool_iteration_1_returns_to_default),
        ("forced ② (案Y): iter0 tools 1 個絞り",
         test_forced_tool_iteration_0_filter_active),
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
