"""Permissions — 5 mode + allow/ask/deny rules.

claw-code の rust/crates/runtime/src/permissions.rs + permission_enforcer.rs の Python port。

責務:
  - PermissionMode 判定 (ReadOnly / WorkspaceWrite / DangerFullAccess / Prompt / Allow)
  - tool ごとの required_permission マッピング
  - allow/ask/deny ルール適用
  - Noetic の tool_level 0-6 と統合する

TODO: 別セッションで実装。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PermissionMode(Enum):
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    DANGER_FULL_ACCESS = "danger_full_access"
    PROMPT = "prompt"
    ALLOW = "allow"


class PermissionDecision(Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class PermissionRules:
    """settings.json の permissions.rules 相当。"""
    allow: list = field(default_factory=list)  # tool name or prefix
    ask: list = field(default_factory=list)
    deny: list = field(default_factory=list)


class PermissionEnforcer:
    """permission チェックの中核。

    まだ未実装。
    """

    def __init__(self, mode: PermissionMode, rules: Optional[PermissionRules] = None):
        self.mode = mode
        self.rules = rules or PermissionRules()

    def check(self, tool_name: str, tool_input: dict) -> PermissionDecision:
        """tool の実行可否を判定。"""
        raise NotImplementedError

    def required_mode_for(self, tool_name: str) -> PermissionMode:
        """tool ごとの必要権限を返す。"""
        raise NotImplementedError
