"""Approval rules engine — AgentSpec DSL pattern (PLAN §5-4)。

段階13 Phase 6.2 で導入。`settings.json` の `approval.rules` を順に評価して
approval 要否 (True=承認必要 / False=auto) を判定する policy 関数を生成する。

## 設計方針 (ゆう要件「動的・抽象・フロート」+ W 案「既存 helper 共有」)

Web 調査 (2026 ベストプラクティス) で identified された AgentSpec DSL pattern
(arxiv 2503.18666 / ICSE 2026) の Python port + Codex review 1-5 周目で
発覚した「path 正規化二重実装の root cause」への (W) 解決案。

- **動的**: rules は settings.json から runtime に読込、再起動で反映
- **抽象**: tool 名 hard-code じゃなく pattern (tool 集合 + body_modify boolean) で集約
- **フロート**: 同じ tool でも tool_input.path 等で別判定 (write_file は
  body_modify path で approve、それ以外で auto)
- **path 評価は path_classifier helper に委譲** (W 案、root cause fix):
  rule 内に独自 regex を書かず、`body_modify: true` boolean で
  `is_body_modify_path` helper を呼ぶ。git_auto_stash / pending_body_modify
  hook と path 判定が完全一致、Windows case / traversal / absolute / backslash
  の path quirk を 1 か所で吸収。

完全動的 (ProbGuard 系 behavioral 学習) は段階14 Pure FEP / Phase 7 smoke 後
の reserved。本実装は実用 latency (<1ms) + Noetic settings 体系整合の現実解。

## rule schema

各 rule は以下の field を持つ dict:

```json
{
  "tools": ["bash", "reboot"],     // tool 名 set (必須)
  "body_modify": true,             // optional boolean (true なら
                                   //   path_classifier.is_body_modify_path
                                   //   が True を返す path のみ rule hit)
  "action": "approve" | "auto"      // 必須
}
```

評価順序:
1. rules を順次評価 (settings 配列順)
2. tool_name が rule["tools"] に含まれる かつ
   body_modify が False または body_modify=True かつ
   `is_body_modify_path(tool_input["path"], workspace_root)` が True
   なら rule の action を採用、即返却
3. どの rule にも hit しなければ default_action を採用

## LLM injection 耐性

LLM が tool_input に `_skip_approval=True` / `_safety_class=...` 等の任意
field を仕込んでも policy_fn は **tool_name と (規定された) path 引数のみ**
評価するため影響ゼロ。Codex review 3 周目 P1 (model-supplied skip flag による
approval bypass) を architecture level で完全防御。

## path quirk 一網打尽 (Codex review 2/4/5 周目 root cause fix、W 案)

path 評価は `core/runtime/path_classifier.is_body_modify_path` に委譲、
git_auto_stash hook / pending_body_modify hook と **完全同一の path 正規化
logic** を共有する。

- 表現差 (Windows case / traversal / absolute / backslash) は helper 1 か所
  で対応、approval_rules.py / hooks.py の二重実装を撤廃
- 新 path quirk 発見時は path_classifier 1 か所修正で 3 hook 全反映
- Codex 連続 BLOCKER detection (Z 案 → Q 案 path_pattern → W 案 helper 共有)
  の architecture-level root cause fix

## たとえ

policy_fn = 「**rules engine** が **DSL で書かれた pattern 集合** を順に
チェックする鍵業者」。鍵業者は **path 専門業者** (path_classifier helper) に
身体改変判定を **完全委託**、自分で path ラベルを読まない。新 tool / 新 path
は settings.json に 1 行追加するだけで対応、tool 内部に metadata 書かなくて
済む。default は safe-by-default approval (= 知らない扉は人に聞く)、HITL
fallback で安全。
"""
from pathlib import Path
from typing import Callable, Optional


_VALID_ACTIONS = ("approve", "auto")


def make_policy_fn(
    rules: list,
    default_action: str = "approve",
    workspace_root=None,
) -> Callable[[str, dict], bool]:
    """rules + default から policy 関数を生成。

    Args:
        rules: rule dict のリスト (上記 schema)
        default_action: rules 漏れ時の action ("approve" or "auto")。
            default = "approve" は safe-by-default HITL fallback、新 tool /
            未知 tool で承認 UI 発火 → ゆう判断機会確保。
        workspace_root: profile root (pathlib.Path or str)。`body_modify: true`
            rule の path 判定で `is_body_modify_path` helper に渡される。
            None なら helper は raw path で判定 (test 用 backwards compat)。

    Returns:
        signature: (tool_name: str, tool_input: dict) -> bool
            True = 承認必要 / False = auto bypass

    Raises:
        ValueError: default_action が "approve" / "auto" 以外、または
            rule["action"] が同様に invalid な場合 (起動時 fail-fast)。
    """
    from core.runtime.path_resolver import is_body_modify_path

    if default_action not in _VALID_ACTIONS:
        raise ValueError(
            f"default_action must be one of {_VALID_ACTIONS}, "
            f"got {default_action!r}"
        )
    default_requires_approval = (default_action == "approve")

    root: Optional[Path] = None
    if workspace_root is not None:
        try:
            root = Path(workspace_root).resolve()
        except Exception as e:
            raise ValueError(
                f"workspace_root cannot be resolved: {workspace_root!r} ({e})"
            )

    compiled: list = []
    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError(f"rules[{idx}] must be dict, got {type(rule).__name__}")
        tools = rule.get("tools")
        if not isinstance(tools, list) or not tools:
            raise ValueError(
                f"rules[{idx}].tools must be non-empty list, got {tools!r}"
            )
        action = rule.get("action")
        if action not in _VALID_ACTIONS:
            raise ValueError(
                f"rules[{idx}].action must be one of {_VALID_ACTIONS}, "
                f"got {action!r}"
            )
        body_modify = bool(rule.get("body_modify", False))
        compiled.append({
            "tools": set(tools),
            "body_modify": body_modify,
            "action": action,
        })

    def policy_fn(tool_name: str, tool_input: dict) -> bool:
        """rules を順次評価、hit した rule の action を採用。

        Returns:
            True: 承認必要 (approval UI 発火経路)
            False: auto bypass
        """
        for rule in compiled:
            if tool_name not in rule["tools"]:
                continue
            if rule["body_modify"]:
                raw_path = str(tool_input.get("path") or "")
                if not is_body_modify_path(raw_path, root):
                    continue
            return rule["action"] == "approve"
        return default_requires_approval

    return policy_fn
