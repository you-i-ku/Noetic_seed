"""Hooks — PreToolUse / PostToolUse / PostToolUseFailure.

claw-code の rust/crates/runtime/src/hooks.rs:22-36 の Python port。

責務:
  - tool 実行の前後にフック処理を挟む
  - Noetic の E値評価を PostToolUse hook に埋め込む想定
  - Noetic の承認3層 (args + intent + message) を PreToolUse hook に埋め込む想定

TODO: 別セッションで実装。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class HookEvent(Enum):
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    POST_TOOL_USE_FAILURE = "post_tool_use_failure"


@dataclass
class HookRunResult:
    """hook の戻り値。"""
    denied: bool = False
    failed: bool = False
    permission_override: Optional[str] = None  # PermissionMode name
    updated_input: Optional[dict] = None
    messages: list = field(default_factory=list)


class HookRunner:
    """hook event を発火して、登録されたハンドラーを順次呼ぶ。

    まだ未実装。
    """

    def __init__(self):
        self._handlers: dict = {
            HookEvent.PRE_TOOL_USE: [],
            HookEvent.POST_TOOL_USE: [],
            HookEvent.POST_TOOL_USE_FAILURE: [],
        }

    def register(self, event: HookEvent, handler):
        """handler: callable(tool_name, tool_input, tool_output=None, is_error=False) -> HookRunResult"""
        self._handlers[event].append(handler)

    def run_pre_tool_use(self, tool_name: str, tool_input: dict) -> HookRunResult:
        """tool 実行前 hook. denied なら実行をスキップ。"""
        raise NotImplementedError

    def run_post_tool_use(self, tool_name: str, tool_input: dict, output: str) -> HookRunResult:
        """tool 実行後 hook (成功)。E値評価はここに登録する想定。"""
        raise NotImplementedError

    def run_post_tool_use_failure(self, tool_name: str, tool_input: dict, error: str) -> HookRunResult:
        """tool 実行後 hook (失敗)。"""
        raise NotImplementedError
