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

            # Provider 完結型 (claude_code): SDK 内 in-process MCP の handler が
            # _make_tool_executor 経由で execute 済み (push_session=False)。
            # ここで session に tool_result を一括 push + summary に詰めて break。
            if msg.tool_invocations:
                for inv in msg.tool_invocations:
                    rec = ToolInvocationRecord(
                        tool_id=inv["tool_id"],
                        tool_name=inv["tool_name"],
                        tool_input=inv["tool_input"],
                        output=inv["output"],
                        is_error=inv["is_error"],
                    )
                    summary.tool_invocations.append(rec)
                    self.session.push_tool_result(
                        rec.tool_id, rec.output, is_error=rec.is_error,
                    )
                summary.finish_reason = "completed_in_provider"
                break

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

    def run_turn_with_forced_tool(
        self, forced_tool_name: str,
        user_input: Optional[str] = None,
        forced_system_prompt: Optional[str] = None,
    ) -> TurnSummary:
        """controller が事前選定した tool を LLM に強制実行させる (X-guided モード)。

        通常の run_turn が LLM に tool を自由選択させるのに対し、本メソッドは
        provider の tool_choice 機能で特定 tool を強制する。LLM は選択判断を
        せず、args と 3 層 (tool_intent / tool_expected_outcome / message) の
        生成に専念する。Noetic 哲学「構造で選ぶ、LLM に選ばせない」と整合
        (memory/feedback_llm_as_brain.md)。

        実装方式 (2026-04-18 backend 非依存化):
          - OpenAI 互換: tools payload を forced_tool 1 個に絞り、
            tool_choice="required" (string) で強制。LM Studio 等の object
            tool_choice 未対応 backend に依存しない。
          - Anthropic: object tool_choice が正式対応のため従来通り。

        段階11-D Phase 8 hotfix ② (案 Y、ゆう原案 2026-04-26):
          forced_system_prompt が渡された場合、iteration 0 だけ system_prompt
          の [利用可能なツール] セクションを forced_tool 1 個に絞った prompt
          で LLM 呼出する (function calling spec の絞り込みと prompt 本文の
          絞り込みを同期させ、gemma 等が prompt 経路で別 tool を呼ぶ抜け道を
          塞ぐ)。iteration 1+ では元 system_prompt + filter 解除に戻し、chain
          micro_iter のハルシネーション多様性を確保 (= 「1 step 目: 身体の意思
          (controller) / 2 step 目以降: 脳の探索 (LLM)」の二重構造)。

        Args:
            forced_tool_name: controller が選定した tool 名。
            user_input: 任意の user テキスト (候補の reason など)。
            forced_system_prompt: iteration 0 用に override する system_prompt。
                None で従来挙動 (全 iteration で self.system_prompt 使用)。

        Returns:
            TurnSummary (通常の run_turn と同形式)
        """
        if user_input:
            self.session.push_user_text(user_input)

        summary = TurnSummary()
        tool_choice_forced = self._build_tool_choice(forced_tool_name)
        provider_name = getattr(self.provider, "name", "")
        filter_forced: Optional[set] = (
            None if provider_name == "anthropic" else {forced_tool_name}
        )

        for i in range(self.max_iterations):
            summary.iterations = i + 1

            if i == 0:
                # 身体の意思: controller が選んだ forced_tool 1 個に視界を絞る
                msg = self._call_llm(
                    tool_choice=tool_choice_forced,
                    filter_tool_names=filter_forced,
                    system_prompt_override=forced_system_prompt,
                )
            else:
                # 脳の探索: filter / system_prompt 共に元に戻し、ハルシネー
                # ション多様性を解放 (案 Y、ゆう原案)。tool_choice も外す。
                msg = self._call_llm()

            summary.assistant_messages.append(msg)
            summary.usage = msg.usage
            self.session.push_assistant_message(msg)

            # 段階11-D Step 4-2 hotfix v5: Provider 完結型 (claude_code) は
            # SDK 内 in-process MCP の handler が _make_tool_executor 経由で
            # 実行済みのため msg.tool_uses は常時空、msg.tool_invocations に
            # 記録される。run_turn (line 95-109) と同じパターンを適用して
            # forced_violation 誤判定を回避する。
            if msg.tool_invocations:
                for inv in msg.tool_invocations:
                    rec = ToolInvocationRecord(
                        tool_id=inv["tool_id"],
                        tool_name=inv["tool_name"],
                        tool_input=inv["tool_input"],
                        output=inv["output"],
                        is_error=inv["is_error"],
                    )
                    summary.tool_invocations.append(rec)
                    self.session.push_tool_result(
                        rec.tool_id, rec.output, is_error=rec.is_error,
                    )
                summary.finish_reason = "completed_in_provider"
                break

            if not msg.tool_uses:
                # 段階11-D Step 4-2 hotfix v4: iteration 0 (forced 段階) で tool
                # 不発火 = LLM① 選択 = controller 決定 を LLM が覆した design
                # contract 違反。memory/feedback_llm2_iter0_forced_contract.md 参照。
                # iteration 1+ で tool 不発火は LLM 自由探索 (案 Y 二段構造) の正常 end。
                # claude_code 完結型は上の msg.tool_invocations 分岐で先に処理済み。
                summary.finish_reason = (
                    "forced_violation" if i == 0 else "no_tool"
                )
                break

            for tu in msg.tool_uses:
                rec = self._execute_tool_use(tu.id, tu.name, tu.input)
                summary.tool_invocations.append(rec)
        else:
            summary.finish_reason = "max_iterations"

        summary.messages = [dict(m) for m in self.session.messages]
        return summary

    def _build_tool_choice(self, tool_name: str):
        """provider 固有の tool_choice 値を生成。

        Returns:
            - Anthropic / claude_code: {"type": "tool", "name": tool_name}
              (object, 正式対応形式)
            - OpenAI 互換: "required" (string)。tools 側で単一 tool に絞って強制。
              object 形式は LM Studio 等で未対応のため回避。
        """
        provider_name = getattr(self.provider, "name", "")
        if provider_name in ("anthropic", "claude_code"):
            return {"type": "tool", "name": tool_name}
        return "required"

    def _call_llm(
        self,
        tool_choice=None,
        filter_tool_names: Optional[set] = None,
        system_prompt_override: Optional[str] = None,
    ) -> AssistantMessage:
        tool_specs = self._build_tool_specs_for_provider(filter_tool_names)
        messages = self._serialize_messages()
        # 段階11-D Phase 8 hotfix ② (案 Y): forced_tool iteration 0 で
        # tool 視界を絞った system_prompt に切替可能、None で self.system_prompt 既定
        req = ApiRequest(
            system_prompt=system_prompt_override or self.system_prompt,
            messages=messages,
            tools=tool_specs,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tool_choice=tool_choice,
            tool_executor=self._make_tool_executor(),
        )
        return self.provider.stream(req)

    def _serialize_messages(self) -> list:
        name = getattr(self.provider, "name", "")
        # claude_code は stream-json input 経路で Anthropic native content block を
        # 受け付ける (実機確認済み 2026-04-27) ので anthropic と同じ serializer。
        if name in ("anthropic", "claude_code"):
            return self.session.serialize_for_anthropic()
        return self.session.serialize_for_openai()

    def _build_tool_specs_for_provider(
        self, filter_names: Optional[set] = None,
    ) -> list:
        specs = self.tool_registry.list()
        if filter_names is not None:
            specs = [s for s in specs if s.name in filter_names]
        name = getattr(self.provider, "name", "")
        # claude_code は in-process MCP に Anthropic native tool 形式 (name +
        # description + input_schema の flat) で渡す必要があるため anthropic と統一。
        if name in ("anthropic", "claude_code"):
            return [s.to_anthropic_format() for s in specs]
        return [s.to_openai_format() for s in specs]

    def _execute_tool_use(self, tool_id: str, tool_name: str,
                          tool_input: dict,
                          push_session: bool = True) -> ToolInvocationRecord:
        """tool 1 件の hook + permission + approval + execute を 1 連で行う。

        push_session=True (default): Anthropic / OpenAI 経路。実行結果を
            self.session.push_tool_result で session に積む。run_turn の
            for tu in msg.tool_uses ループからの呼び出し前提。
        push_session=False: claude_code provider 経路。session への push は
            run_turn の Provider 完結型分岐で一元管理する。
            (handler 内から _make_tool_executor 経由で呼ばれる場合)
        """
        rec = ToolInvocationRecord(
            tool_id=tool_id, tool_name=tool_name, tool_input=tool_input,
        )

        pre = self.hook_runner.run_pre_tool_use(tool_name, tool_input)
        rec.pre_hook_messages = list(pre.messages)
        current_input = pre.updated_input or tool_input

        if pre.denied:
            self._finalize(rec, "[REJECTED] denied by pre hook",
                           is_error=True, push_session=push_session)
            return rec
        if pre.failed:
            self._finalize(rec, "[REJECTED] pre hook failed",
                           is_error=True, push_session=push_session)
            return rec

        decision = self.permission_enforcer.check(tool_name, current_input)
        rec.permission_decision = decision.value

        if decision == PermissionDecision.DENY:
            self._finalize(rec, "[REJECTED] permission denied",
                           is_error=True, push_session=push_session)
            return rec

        if decision == PermissionDecision.ASK:
            approved = self._ask_approval(tool_name, current_input,
                                          rec.pre_hook_messages)
            if not approved:
                self._finalize(rec, "[REJECTED] approval denied",
                               is_error=True, push_session=push_session)
                return rec

        try:
            output = self.tool_registry.execute(tool_name, current_input)
        except Exception as e:
            err = f"tool execution error: {e}"
            self.hook_runner.run_post_tool_use_failure(
                tool_name, current_input, err
            )
            self._finalize(rec, err, is_error=True, push_session=push_session)
            return rec

        post = self.hook_runner.run_post_tool_use(
            tool_name, current_input, output
        )
        rec.post_hook_messages = list(post.messages)

        rec.tool_input = current_input
        self._finalize(rec, output, is_error=False,
                       push_session=push_session)
        return rec

    def _make_tool_executor(self):
        """claude_code provider 用の tool 実行 callback を生成。

        ApiRequest.tool_executor として provider に渡される。in-process MCP の
        handler 内から `(tool_id, tool_name, tool_input) -> (output, is_error)`
        の形で呼ばれて、ConversationRuntime の hook + permission + approval
        ロジックを完全経由する (= Anthropic / OpenAI 経路と同じ確認手順)。

        session への push は handler 内では行わない (push_session=False)。
        Provider 完結型 (claude_code) の run_turn 分岐側で一括 push する。
        """
        def executor(tool_id: str, tool_name: str, tool_input: dict):
            rec = self._execute_tool_use(
                tool_id, tool_name, tool_input, push_session=False,
            )
            return rec.output, rec.is_error
        return executor

    def _finalize(self, rec: ToolInvocationRecord,
                  output: str, is_error: bool,
                  push_session: bool = True) -> None:
        rec.output = output
        rec.is_error = is_error
        if push_session:
            self.session.push_tool_result(rec.tool_id, output,
                                           is_error=is_error)

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
