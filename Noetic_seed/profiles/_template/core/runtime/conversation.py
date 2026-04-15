"""ConversationRuntime — agent loop layer.

claw-code の rust/crates/runtime/src/conversation.rs:126-189 の Python port。

責務:
  - LLM の stream 呼出
  - ContentBlock から tool_use 抽出
  - pre_hook → execute → post_hook の駆動
  - session.messages への push
  - 停止判定 (tool_use が出なくなるまで / max_iterations)

Noetic 既存の main.py との関係:
  - main.py の pressure/entropy/発火判定は変更なし
  - 発火時に run_turn(user_input) を呼ぶだけ
  - 1 cycle = 1 tool を維持したい場合は max_iterations=1 で呼ぶ
  - 連続動作させたい場合は max_iterations を増やす

TODO: 別セッションで実装。現状は interface のみ。
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnSummary:
    """1 run_turn の結果。"""
    messages: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    usage: Optional[dict] = None
    iterations: int = 0
    finish_reason: str = "pending"  # "completed" | "max_iterations" | "aborted"


class ConversationRuntime:
    """エージェントループ本体。

    まだ未実装。インターフェースだけ。
    """

    def __init__(
        self,
        provider,       # providers.BaseProvider
        tool_registry,  # runtime.registry.ToolRegistry
        hook_runner,    # runtime.hooks.HookRunner
        permission_enforcer,  # runtime.permissions.PermissionEnforcer
        max_iterations: int = 1,  # 1 = Noetic 現状の細粒度息継ぎ型を維持
    ):
        self.provider = provider
        self.tool_registry = tool_registry
        self.hook_runner = hook_runner
        self.permission_enforcer = permission_enforcer
        self.max_iterations = max_iterations
        self.session_messages: list = []

    def run_turn(self, user_input: str) -> TurnSummary:
        """1 ターン実行する。

        pseudocode (未実装):
          session.push_user_text(user_input)
          for i in range(max_iterations):
              events = provider.stream(request)
              (assistant_msg, usage) = build_assistant_message(events)
              session.push(assistant_msg)
              pending_tool_uses = extract_tool_uses(assistant_msg)
              if not pending_tool_uses:
                  return TurnSummary(finish_reason="completed")
              for tool_use in pending_tool_uses:
                  pre = hook_runner.run_pre_tool_use(tool_use)
                  if pre.denied:
                      continue
                  output = tool_registry.execute(tool_use.name, tool_use.input)
                  hook_runner.run_post_tool_use(tool_use, output)
                  session.push_tool_result(tool_use.id, output)
          return TurnSummary(finish_reason="max_iterations")
        """
        raise NotImplementedError("ConversationRuntime.run_turn() not implemented yet")
