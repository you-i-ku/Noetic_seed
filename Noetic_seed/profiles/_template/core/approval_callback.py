"""approval_callback factory — ConversationRuntime の承認フック実装。

Phase 4 Step E-2c: Step A で「Step E に委譲」と記録した
`pause_on_await` 連動 + ws_server 3 層 UI 通知を実装する。

## 動作フロー (APPROVAL_PROMPT_SPEC §6.2)
```
PreToolUse hook (理由・予想と、確認時の note をチェック) 通過
  ↓
approval_callback 呼出 (本ファイル)
  - pause_on_await=True → ws_server.set_paused(True)  (pressure tick 停止)
  - 3 層 UI payload 整形 (args / intent&expected / note)
  - ws_server.request_approval() で WS 送信 + 応答待ち
  ↓
UI で OK/NG 判定 (WS 接続なければターミナル input)
  ↓
pause 解放 + bool 返却
```

## 哲学整合
- 承認者は「ユーザー」ではなく「端末前の協力者」(memory/feedback_no_user_assistant_frame.md)
- 3 層表示で what / why / to_human を明示 (APPROVAL_PROMPT_SPEC §2.2)
- pause_on_await で主観時間を停止 (Noetic の pressure 駆動と整合)
"""
from typing import Callable, Optional


ApprovalCallback = Callable[[str, dict, list], bool]


_APPROVAL_FIELDS = ("tool_intent", "tool_expected_outcome", "note")


def _format_preview(
    tool_name: str,
    tool_input: dict,
    pre_hook_messages: list,
) -> str:
    """APPROVAL_PROMPT_SPEC §6.3 の 3 層 UI payload を文字列化。

    構成:
      Tool: <tool_name>
      ---
      ① what (args): <tool 固有 args、承認 3 層は除外>
      ② why:
         intent:   <tool_intent>
         expected: <tool_expected_outcome>
      ③ to you (= 協力者):
         <note>
      ---
      [pre_hook messages (あれば)]
    """
    # tool 固有 args: 承認 3 層フィールドを除外
    args = {k: v for k, v in tool_input.items()
            if k not in _APPROVAL_FIELDS}

    # 空文字も '(空)' 扱い (PreToolUse hook で欠損 deny されるので通常は埋まる)
    lines: list = [
        f"Tool: {tool_name}",
        "---",
        f"① what (args): {args}",
        "② why:",
        f"   intent:   {tool_input.get('tool_intent') or '(空)'}",
        f"   expected: {tool_input.get('tool_expected_outcome') or '(空)'}",
        "③ to you (= 協力者):",
        f"   {tool_input.get('note') or '(空)'}",
    ]

    # 段階7 Step 5: memory_store で未登録タグ → 新タグ発明フラグ
    if tool_name == "memory_store":
        network = str(tool_input.get("network", "")).strip()
        if network:
            try:
                from core.tag_registry import is_tag_registered
                if not is_tag_registered(network):
                    lines.append("---")
                    lines.append(f"⚠️  新タグ発明: '{network}'")
                    rules = tool_input.get("rules")
                    if rules:
                        lines.append(f"   rules: {rules}")
                    else:
                        lines.append("   (rules 省略 → 全 False default で auto register)")
            except Exception:
                pass

    if pre_hook_messages:
        lines.append("---")
        lines.append("[pre_hook]")
        for m in pre_hook_messages:
            lines.append(f"  {m}")

    return "\n".join(lines)


PolicyFn = Callable[[str, dict], bool]


def make_approval_callback(
    pause_on_await: bool = True,
    timeout_sec: int = 300,
    request_approval_fn: Optional[Callable[[str, str, int], bool]] = None,
    set_paused_fn: Optional[Callable[[bool], None]] = None,
    auto_approve_all: bool = False,
    policy_fn: Optional[PolicyFn] = None,
) -> ApprovalCallback:
    """ConversationRuntime に渡す approval_callback を生成。

    Args:
        pause_on_await: True で承認待ち中に ws_server.set_paused(True)
            を発動 (settings.approval.pause_on_await)。
        timeout_sec: 承認応答のタイムアウト。既定 300 秒。
        request_approval_fn: 承認要求関数 (テスト注入用)。None で
            ws_server.request_approval を遅延 import。
        set_paused_fn: pause 操作関数 (テスト注入用)。None で
            ws_server.set_paused を遅延 import。
        auto_approve_all: True で全 tool 呼出を無条件 approve する
            smoke util (settings.approval.auto_approve_all)。pause / UI
            request は一切発動しない。実運用では False 維持。smoke 13
            では身体改変も含めて全 bypass で iku の自由度を最大化する設計。
        policy_fn: 段階13 Phase 6.2 で追加。signature
            `(tool_name: str, tool_input: dict) -> bool`、True なら承認
            必要 (UI 発火経路)、False なら auto bypass。LLM 由来の
            tool_input 任意 field を信頼せず、policy_fn が独立 source of
            truth として approval 要否を判定する (Codex review 3 周目 P1
            architecture level fix)。settings.json approval.rules から
            生成される (core/runtime/approval_rules.make_policy_fn)。
            None なら旧 HITL behavior (全 tool 承認、後方互換)。

    Returns:
        signature (tool_name, tool_input, pre_hook_messages) -> bool
    """
    def _resolve_request_approval() -> Callable[[str, str, int], bool]:
        if request_approval_fn is not None:
            return request_approval_fn
        from core.ws_server import request_approval as _ra
        return _ra

    def _resolve_set_paused() -> Callable[[bool], None]:
        if set_paused_fn is not None:
            return set_paused_fn
        from core.ws_server import set_paused as _sp
        return _sp

    def _callback(
        tool_name: str,
        tool_input: dict,
        pre_hook_messages: list,
    ) -> bool:
        # ★ 評価順序の design intent (Codex rescue 7 周目評価で確認):
        #   auto_approve_all (smoke 用 全 bypass) → policy_fn (本番 rules 評価)
        # の順で early return する。auto_approve_all=true の時は policy_fn が
        # 評価されないため、smoke 13 では身体改変も含めて全 bypass される
        # (ゆう 2026-05-04 spec literal、PLAN §5-4)。本番運用では
        # auto_approve_all=false で AgentSpec DSL rules が有効化される。
        if auto_approve_all:
            print(f"  [approval] auto-approve (smoke util): {tool_name}")
            return True

        # 段階13 Phase 6.2: AgentSpec DSL ベースの TOOL_APPROVAL_POLICY
        # (PLAN §5-4)。settings.json approval.rules から生成された
        # policy_fn が tool_name + path のみで approval 要否を判定。
        # LLM 由来の tool_input 任意 field の影響ゼロ (Codex review 3
        # 周目 P1 fix、architecture level)。
        if policy_fn is not None and not policy_fn(tool_name, tool_input):
            return True  # auto bypass

        preview = _format_preview(tool_name, tool_input,
                                  pre_hook_messages or [])

        request_approval = _resolve_request_approval()
        set_paused = _resolve_set_paused()

        if pause_on_await:
            set_paused(True)
        try:
            return bool(request_approval(tool_name, preview, timeout_sec))
        finally:
            if pause_on_await:
                set_paused(False)

    return _callback
