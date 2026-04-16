"""Hooks — PreToolUse / PostToolUse / PostToolUseFailure.

claw-code の rust/crates/runtime/src/hooks.rs:22-36 の Python port。

ファイル構成:
  - 上半分 (HookEvent / HookRunResult / HookRunner):
      claw-code 準拠の純粋インフラ。settings 非依存。
  - 下半分 (Noetic 固有 factory 群):
      承認 3 層チェッカー等、Phase 4 で追加される Noetic 固有 handler。
      settings 依存は factory 引数で受け取り、main.py 側で HookRunner に
      register する。
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


# ============================================================
# Noetic 固有 handler (Phase 4 追加)
# ------------------------------------------------------------
# 承認 3 層 (tool_intent / tool_expected_outcome / message) の欠損
# チェックを PreToolUse hook として登録するための factory。
# APPROVAL_PROMPT_SPEC.md §5 の仕様を実装する。
# ============================================================

_APPROVAL_FIELDS = ("tool_intent", "tool_expected_outcome", "message")
_VALID_MISSING_POLICIES = ("deny", "warn", "auto_fill")


def _auto_fill_field(tool_name: str, field_name: str) -> str:
    """tool_name と欠損フィールド名からプレースホルダ文字列を生成。

    Phase 4 Step A 時点の最小実装。将来 tool メタデータベースの推定に
    差し替え可能 (APPROVAL_PROMPT_SPEC.md §9 未決定事項)。
    """
    if field_name == "tool_intent":
        return f"[auto_fill] {tool_name} 実行"
    if field_name == "tool_expected_outcome":
        return f"[auto_fill] {tool_name} の結果取得"
    if field_name == "message":
        return f"[auto_fill] {tool_name} を実行します"
    return "[auto_fill]"


def make_pre_tool_use_approval_check(
    missing_field_policy: str = "deny",
) -> PreHandler:
    """承認 3 層欠損チェッカーを生成。

    tool_input が `tool_intent` / `tool_expected_outcome` / `message`
    を揃えているか検証する PreToolUse hook を返す。

    factory にしている理由: hooks.py を claw-code 準拠の純粋インフラに
    保つため、settings 依存は factory 引数で注入する。main.py 側が
    `settings.approval.missing_field_policy` を読んで register する。

    Args:
        missing_field_policy: 欠損時の動作。"deny" / "warn" / "auto_fill"

    Raises:
        ValueError: 未知の policy が渡された場合 (設定ミスは起動時に検出)
    """
    if missing_field_policy not in _VALID_MISSING_POLICIES:
        raise ValueError(
            f"unknown missing_field_policy={missing_field_policy!r}; "
            f"expected one of {_VALID_MISSING_POLICIES}"
        )

    def _check(tool_name: str, tool_input: dict) -> HookRunResult:
        missing = []
        for field_name in _APPROVAL_FIELDS:
            value = tool_input.get(field_name, "")
            if value is None or not str(value).strip():
                missing.append(field_name)

        if not missing:
            return HookRunResult.allow()

        if missing_field_policy == "deny":
            return HookRunResult.deny([
                f"[approval] tool_input 欠損: {', '.join(missing)}。"
                "tool_intent / tool_expected_outcome / message の 3 層を"
                "揃えて再生成してください。"
            ])

        if missing_field_policy == "warn":
            return HookRunResult(
                denied=False,
                messages=[
                    f"[approval] warning: 欠損フィールド {missing} "
                    "(missing_field_policy=warn のため実行継続)"
                ],
            )

        updated = dict(tool_input)
        for field_name in missing:
            updated[field_name] = _auto_fill_field(tool_name, field_name)
        return HookRunResult(
            denied=False,
            updated_input=updated,
            messages=[f"[approval] 自動補完: {missing}"],
        )

    return _check
