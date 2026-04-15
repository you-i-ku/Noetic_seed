"""Hooks — PreToolUse / PostToolUse / PostToolUseFailure.

claw-code の rust/crates/runtime/src/hooks.rs:22-36 の Python port。

厳密 claw-code 準拠。Noetic 固有の E値評価 / 承認3層は
handler として外部登録する形で接続する想定。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class HookEvent(Enum):
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    POST_TOOL_USE_FAILURE = "post_tool_use_failure"


@dataclass
class HookRunResult:
    """hook の戻り値。handler 複数分の集約結果。"""
    denied: bool = False
    failed: bool = False
    permission_override: Optional[str] = None
    updated_input: Optional[dict] = None
    messages: list = field(default_factory=list)

    @classmethod
    def allow(cls, messages: Optional[list] = None) -> "HookRunResult":
        return cls(denied=False, messages=messages or [])

    @classmethod
    def deny(cls, messages: Optional[list] = None) -> "HookRunResult":
        return cls(denied=True, messages=messages or [])

    def merge(self, other: "HookRunResult") -> "HookRunResult":
        return HookRunResult(
            denied=self.denied or other.denied,
            failed=self.failed or other.failed,
            permission_override=(other.permission_override
                                 or self.permission_override),
            updated_input=(other.updated_input
                           if other.updated_input is not None
                           else self.updated_input),
            messages=self.messages + other.messages,
        )


PreHandler = Callable[[str, dict], HookRunResult]
PostHandler = Callable[[str, dict, str], HookRunResult]
FailureHandler = Callable[[str, dict, str], HookRunResult]


class HookRunner:
    """hook event を発火して、登録された handler を順次呼ぶ。"""

    def __init__(self):
        self._pre: list = []
        self._post: list = []
        self._fail: list = []

    def register_pre(self, handler: PreHandler) -> None:
        self._pre.append(handler)

    def register_post(self, handler: PostHandler) -> None:
        self._post.append(handler)

    def register_failure(self, handler: FailureHandler) -> None:
        self._fail.append(handler)

    def run_pre_tool_use(self, tool_name: str,
                         tool_input: dict) -> HookRunResult:
        acc = HookRunResult.allow()
        current_input = tool_input
        for handler in self._pre:
            try:
                r = handler(tool_name, current_input)
            except Exception as e:
                return HookRunResult(
                    denied=False, failed=True,
                    messages=acc.messages + [f"pre_hook exception: {e}"],
                )
            acc = acc.merge(r)
            if r.updated_input is not None:
                current_input = r.updated_input
            if acc.denied:
                break
        if current_input is not tool_input:
            acc.updated_input = current_input
        return acc

    def run_post_tool_use(self, tool_name: str, tool_input: dict,
                          output: str) -> HookRunResult:
        acc = HookRunResult.allow()
        for handler in self._post:
            try:
                r = handler(tool_name, tool_input, output)
            except Exception as e:
                return HookRunResult(
                    failed=True,
                    messages=acc.messages + [f"post_hook exception: {e}"],
                )
            acc = acc.merge(r)
        return acc

    def run_post_tool_use_failure(self, tool_name: str, tool_input: dict,
                                  error: str) -> HookRunResult:
        acc = HookRunResult.allow()
        for handler in self._fail:
            try:
                r = handler(tool_name, tool_input, error)
            except Exception as e:
                return HookRunResult(
                    failed=True,
                    messages=acc.messages + [f"failure_hook exception: {e}"],
                )
            acc = acc.merge(r)
        return acc
