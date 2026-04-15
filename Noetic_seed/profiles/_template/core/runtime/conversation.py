"""ConversationRuntime — agent loop layer.

claw-code の rust/crates/runtime/src/conversation.rs:126-189 の Python port。

厳密 claw-code 準拠。Noetic 固有 (E値評価, 承認3層, pressure駆動) は
hook handler / approval_callback として外部から注入する形で接続する想定。
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.providers.base import ApiRequest, AssistantMessage, BaseProvider
from core.runtime.hooks import HookRunner
from core.runtime.permissions import (
    PermissionDecision,
    PermissionEnforcer,
    PermissionMode,
)
from core.runtime.registry import ToolRegistry
from core.runtime.session import Session


FinishReason = str


@dataclass
class ToolInvocationRecord:
    tool_id: str
    tool_name: str
    tool_input: dict
    output: str = ""
    is_error: bool = False
    permission_decision: Optional[str] = None
    pre_hook_messages: list = field(default_factory=list)
    post_hook_messages: list = field(default_factory=list)


@dataclass
class TurnSummary:
    messages: list = field(default_factory=list)
    tool_invocations: list = field(default_factory=list)
    usage: Optional[dict] = None
    iterations: int = 0
    finish_reason: FinishReason = "completed"
    assistant_messages: list = field(default_factory=list)


ApprovalCallback = Callable[[str, dict, list], bool]


class ConversationRuntime:
    """エージェントループ本体。"""

    def __init__(
        self,
        provider: BaseProvider,
        tool_registry: ToolRegistry,
        hook_runner: Optional[HookRunner] = None,
        permission_enforcer: Optional[PermissionEnforcer] = None,
        max_iterations: int = 1,
        system_prompt: str = "",
        approval_callback: Optional[ApprovalCallback] = None,
        max_tokens: int = 24000,
        temperature: float = 0.7,
    ):
        self.provider = provider
        self.tool_registry = tool_registry
        self.hook_runner = hook_runner or HookRunner()
        self.permission_enforcer = permission_enforcer or PermissionEnforcer(
            mode=PermissionMode.PROMPT
        )
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.approval_callback = approval_callback
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.session = Session()

    def run_turn(self, user_input: Optional[str] = None) -> TurnSummary:
        if user_input:
            self.session.push_user_text(user_input)

        summary = TurnSummary()

        for i in range(self.max_iterations):
            summary.iterations = i + 1

            msg = self._call_llm()
            summary.assistant_messages.append(msg)
            summary.usage = msg.usage
            self.session.push_assistant_message(msg)

            if not msg.tool_uses:
                summary.finish_reason = "completed"
                break

            for tu in msg.tool_uses:
                rec = self._execute_tool_use(tu.id, tu.name, tu.input)
                summary.tool_invocations.append(rec)
        else:
            summary.finish_reason = "max_iterations"

        summary.messages = [dict(m) for m in self.session.messages]
        return summary

    def _call_llm(self) -> AssistantMessage:
        tool_specs = self._build_tool_specs_for_provider()
        messages = self._serialize_messages()
        req = ApiRequest(
            system_prompt=self.system_prompt,
            messages=messages,
            tools=tool_specs,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return self.provider.stream(req)

    def _serialize_messages(self) -> list:
        name = getattr(self.provider, "name", "")
        if name == "anthropic":
            return self.session.serialize_for_anthropic()
        return self.session.serialize_for_openai()

    def _build_tool_specs_for_provider(self) -> list:
        specs = self.tool_registry.list()
        name = getattr(self.provider, "name", "")
        if name == "anthropic":
            return [s.to_anthropic_format() for s in specs]
        return [s.to_openai_format() for s in specs]

    def _execute_tool_use(self, tool_id: str, tool_name: str,
                          tool_input: dict) -> ToolInvocationRecord:
        rec = ToolInvocationRecord(
            tool_id=tool_id, tool_name=tool_name, tool_input=tool_input,
        )

        pre = self.hook_runner.run_pre_tool_use(tool_name, tool_input)
        rec.pre_hook_messages = list(pre.messages)
        current_input = pre.updated_input or tool_input

        if pre.denied:
            self._finalize(rec, "denied by pre hook", is_error=True)
            return rec
        if pre.failed:
            self._finalize(rec, "pre hook failed", is_error=True)
            return rec

        decision = self.permission_enforcer.check(tool_name, current_input)
        rec.permission_decision = decision.value

        if decision == PermissionDecision.DENY:
            self._finalize(rec, "permission denied", is_error=True)
            return rec

        if decision == PermissionDecision.ASK:
            approved = self._ask_approval(tool_name, current_input,
                                          rec.pre_hook_messages)
            if not approved:
                self._finalize(rec, "user rejected", is_error=True)
                return rec

        try:
            output = self.tool_registry.execute(tool_name, current_input)
        except Exception as e:
            err = f"tool execution error: {e}"
            self.hook_runner.run_post_tool_use_failure(
                tool_name, current_input, err
            )
            self._finalize(rec, err, is_error=True)
            return rec

        post = self.hook_runner.run_post_tool_use(
            tool_name, current_input, output
        )
        rec.post_hook_messages = list(post.messages)

        rec.tool_input = current_input
        self._finalize(rec, output, is_error=False)
        return rec

    def _finalize(self, rec: ToolInvocationRecord,
                  output: str, is_error: bool) -> None:
        rec.output = output
        rec.is_error = is_error
        self.session.push_tool_result(rec.tool_id, output, is_error=is_error)

    def _ask_approval(self, tool_name: str, tool_input: dict,
                      pre_messages: list) -> bool:
        if self.approval_callback is None:
            return False
        try:
            return bool(self.approval_callback(
                tool_name, tool_input, pre_messages
            ))
        except Exception:
            return False
