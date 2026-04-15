"""Permissions — 5 mode + allow/ask/deny rules.

claw-code の rust/crates/runtime/src/permissions.rs + permission_enforcer.rs の Python port。

責務:
  - PermissionMode 判定 (ReadOnly / WorkspaceWrite / DangerFullAccess / Prompt / Allow)
  - tool ごとの required_permission マッピング
  - allow/ask/deny ルール適用

参照:
  - CLAWCODE_CAPABILITY_INVENTORY.md §5 Permissions
  - claw-code/rust/crates/runtime/src/permission_enforcer.rs:1-340

厳密 claw-code 準拠。Noetic 固有 tool は含めない。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PermissionMode(Enum):
    """全体モード。settings.json の permissions.mode で指定。"""
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    DANGER_FULL_ACCESS = "danger_full_access"
    PROMPT = "prompt"
    ALLOW = "allow"


class PermissionDecision(Enum):
    """check() の結果。"""
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass
class PermissionRules:
    """settings.json の permissions.rules 相当。

    name または prefix で指定可能。
      - "bash" → 完全一致
      - "Worker*" → startswith("Worker")
    """
    allow: list = field(default_factory=list)
    ask: list = field(default_factory=list)
    deny: list = field(default_factory=list)


# claw-code の tool → required_permission マッピング (CLAWCODE_CAPABILITY_INVENTORY.md §5.2)。
# Noetic 固有 tool は含めない。必要なら register_tool_permission() で後付け。
_REQUIRED_PERMISSION: dict = {
    # === read-only ===
    "read_file": PermissionMode.READ_ONLY,
    "glob_search": PermissionMode.READ_ONLY,
    "grep_search": PermissionMode.READ_ONLY,
    "WebFetch": PermissionMode.READ_ONLY,
    "WebSearch": PermissionMode.READ_ONLY,
    "Skill": PermissionMode.READ_ONLY,
    "ToolSearch": PermissionMode.READ_ONLY,
    "LSP": PermissionMode.READ_ONLY,
    "ListMcpResources": PermissionMode.READ_ONLY,
    "ReadMcpResource": PermissionMode.READ_ONLY,
    "StructuredOutput": PermissionMode.READ_ONLY,
    "AskUserQuestion": PermissionMode.READ_ONLY,
    "SendUserMessage": PermissionMode.READ_ONLY,
    "Sleep": PermissionMode.READ_ONLY,
    "TodoWrite": PermissionMode.READ_ONLY,
    "TestingPermission": PermissionMode.READ_ONLY,
    "TaskGet": PermissionMode.READ_ONLY,
    "TaskList": PermissionMode.READ_ONLY,
    "TaskOutput": PermissionMode.READ_ONLY,
    "CronList": PermissionMode.READ_ONLY,
    "WorkerGet": PermissionMode.READ_ONLY,
    "WorkerAwaitReady": PermissionMode.READ_ONLY,
    "WorkerObserveCompletion": PermissionMode.READ_ONLY,

    # === workspace-write ===
    "write_file": PermissionMode.WORKSPACE_WRITE,
    "edit_file": PermissionMode.WORKSPACE_WRITE,
    "Config": PermissionMode.WORKSPACE_WRITE,
    "EnterPlanMode": PermissionMode.WORKSPACE_WRITE,
    "ExitPlanMode": PermissionMode.WORKSPACE_WRITE,
    "NotebookEdit": PermissionMode.WORKSPACE_WRITE,

    # === danger-full-access ===
    "bash": PermissionMode.DANGER_FULL_ACCESS,
    "PowerShell": PermissionMode.DANGER_FULL_ACCESS,
    "REPL": PermissionMode.DANGER_FULL_ACCESS,
    "Agent": PermissionMode.DANGER_FULL_ACCESS,
    "TaskCreate": PermissionMode.DANGER_FULL_ACCESS,
    "TaskStop": PermissionMode.DANGER_FULL_ACCESS,
    "TaskUpdate": PermissionMode.DANGER_FULL_ACCESS,
    "TeamCreate": PermissionMode.DANGER_FULL_ACCESS,
    "TeamDelete": PermissionMode.DANGER_FULL_ACCESS,
    "CronCreate": PermissionMode.DANGER_FULL_ACCESS,
    "CronDelete": PermissionMode.DANGER_FULL_ACCESS,
    "MCP": PermissionMode.DANGER_FULL_ACCESS,
    "McpAuth": PermissionMode.DANGER_FULL_ACCESS,
    "RemoteTrigger": PermissionMode.DANGER_FULL_ACCESS,
    "RunTaskPacket": PermissionMode.DANGER_FULL_ACCESS,
    "WorkerCreate": PermissionMode.DANGER_FULL_ACCESS,
    "WorkerObserve": PermissionMode.DANGER_FULL_ACCESS,
    "WorkerResolveTrust": PermissionMode.DANGER_FULL_ACCESS,
    "WorkerSendPrompt": PermissionMode.DANGER_FULL_ACCESS,
    "WorkerRestart": PermissionMode.DANGER_FULL_ACCESS,
    "WorkerTerminate": PermissionMode.DANGER_FULL_ACCESS,
}


# PermissionMode の強度順 (数値が大きいほど緩い)。
_MODE_LEVEL: dict = {
    PermissionMode.READ_ONLY: 1,
    PermissionMode.WORKSPACE_WRITE: 2,
    PermissionMode.DANGER_FULL_ACCESS: 3,
}


def _matches_pattern(name: str, pattern: str) -> bool:
    """name が pattern にマッチするか判定。
    完全一致 or 末尾 '*' で prefix マッチ。"""
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    return name == pattern


def _matches_any(name: str, patterns: list) -> bool:
    return any(_matches_pattern(name, p) for p in patterns)


class PermissionEnforcer:
    """permission チェックの中核。

    判定順序:
      1. deny rules に一致 → DENY
      2. allow rules に一致 → ALLOW
      3. ask rules に一致 → ASK
      4. mode が ALLOW → ALLOW (全許可)
      5. mode が PROMPT → ASK (常に確認)
      6. mode が DANGER_FULL_ACCESS → ALLOW
      7. tool_required <= mode → ALLOW
      8. tool_required > mode → ASK
    """

    def __init__(self, mode: PermissionMode,
                 rules: Optional[PermissionRules] = None):
        self.mode = mode
        self.rules = rules or PermissionRules()

    def check(self, tool_name: str,
              tool_input: Optional[dict] = None) -> PermissionDecision:
        """tool の実行可否を判定。"""
        if _matches_any(tool_name, self.rules.deny):
            return PermissionDecision.DENY
        if _matches_any(tool_name, self.rules.allow):
            return PermissionDecision.ALLOW
        if _matches_any(tool_name, self.rules.ask):
            return PermissionDecision.ASK
        if self.mode == PermissionMode.ALLOW:
            return PermissionDecision.ALLOW
        if self.mode == PermissionMode.PROMPT:
            return PermissionDecision.ASK
        if self.mode == PermissionMode.DANGER_FULL_ACCESS:
            return PermissionDecision.ALLOW
        required = self.required_mode_for(tool_name)
        if required is None:
            return PermissionDecision.ASK
        mode_lvl = _MODE_LEVEL.get(self.mode, 0)
        req_lvl = _MODE_LEVEL.get(required, 99)
        if req_lvl <= mode_lvl:
            return PermissionDecision.ALLOW
        return PermissionDecision.ASK

    def required_mode_for(self, tool_name: str) -> Optional[PermissionMode]:
        """tool ごとの必要権限を返す。未登録なら None。"""
        return _REQUIRED_PERMISSION.get(tool_name)

    def register_tool_permission(self, tool_name: str,
                                 required: PermissionMode) -> None:
        """tool の required_permission を登録 (MCP tool 等の動的追加用)。"""
        _REQUIRED_PERMISSION[tool_name] = required
