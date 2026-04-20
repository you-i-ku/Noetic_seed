"""Bash Validation — claw-code 準拠の安全性検査。

claw-code 参照: rust/crates/runtime/src/bash_validation.rs (feature branch)

責務: bash command 文字列を解析して、モード/危険度に応じて拒否/警告を返す。
実 bash 実行の pre_tool_use hook として組み込む想定。
"""
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ValidationSeverity(Enum):
    OK = "ok"
    WARN = "warn"
    DENY = "deny"


@dataclass
class ValidationResult:
    severity: ValidationSeverity = ValidationSeverity.OK
    reasons: list = field(default_factory=list)

    @property
    def denied(self) -> bool:
        return self.severity == ValidationSeverity.DENY


# 明確に破壊的なパターン (DENY)
_DESTRUCTIVE_PATTERNS = [
    (r"\brm\s+-rf\s+/\s*(?:$|\s|;|&)", "rm -rf /"),
    (r"\brm\s+-rf\s+/\*", "rm -rf /*"),
    (r"\bdd\s+.*of=/dev/(sd[a-z]|nvme|xvd)", "dd to raw device"),
    (r":\(\)\s*\{.*:\|:\&.*\}\s*;?", "fork bomb"),
    (r"\bmkfs\.(ext|xfs|ntfs|fat)", "filesystem format"),
    (r">\s*/dev/sd[a-z]", "write to raw disk"),
]

# 注意を促すパターン (WARN)
_WARN_PATTERNS = [
    (r"\brm\s+-rf\b", "rm -rf usage"),
    (r"\bsudo\b", "sudo usage"),
    (r"\bchmod\s+777\b", "chmod 777"),
    (r"\b(curl|wget)\b.*\|\s*(bash|sh|zsh)", "piping download to shell"),
    (r"\beval\s+", "eval usage"),
    (r"\bforce\b.*push|\bpush\b.*--force", "force push"),
]

# read-only mode で許可されるコマンド頭 (ホワイトリスト)
_READ_ONLY_WHITELIST = {
    "ls", "cat", "head", "tail", "wc", "grep", "find", "echo",
    "pwd", "whoami", "date", "env", "printenv", "which",
    "git", "stat", "file", "du", "df", "ps", "top", "uname",
}


def _first_token(command: str) -> str:
    try:
        parts = shlex.split(command, posix=True)
    except ValueError:
        return command.split()[0] if command else ""
    return parts[0] if parts else ""


def validate_bash(command: str,
                  read_only_mode: bool = False) -> ValidationResult:
    """bash command を検証する。

    read_only_mode=True のとき、whitelist に無いコマンドは DENY。
    """
    result = ValidationResult()
    if not command or not command.strip():
        result.severity = ValidationSeverity.DENY
        result.reasons.append("empty command")
        return result

    # DESTRUCTIVE 系
    for pat, label in _DESTRUCTIVE_PATTERNS:
        if re.search(pat, command):
            result.severity = ValidationSeverity.DENY
            result.reasons.append(f"destructive: {label}")
            return result

    # READ ONLY
    if read_only_mode:
        head = _first_token(command).split("/")[-1]  # "bash" でも "/bin/bash"
        # サブシェル含むかもチェック (単純に ; | && を含めば複数コマンド)
        if re.search(r"[;&|`]|\$\(|>", command):
            result.severity = ValidationSeverity.DENY
            result.reasons.append(
                "read-only mode disallows compound commands"
            )
            return result
        if head not in _READ_ONLY_WHITELIST:
            result.severity = ValidationSeverity.DENY
            result.reasons.append(
                f"read-only mode disallows '{head}'"
            )
            return result

    # WARN 系
    for pat, label in _WARN_PATTERNS:
        if re.search(pat, command):
            result.severity = ValidationSeverity.WARN
            result.reasons.append(f"caution: {label}")

    return result
