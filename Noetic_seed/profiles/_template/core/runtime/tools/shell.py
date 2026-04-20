"""Shell — bash / PowerShell / REPL.

claw-code 参照: rust/crates/runtime/src/bash.rs:1-283
claw-code 厳密準拠を基本としつつ、Noetic 固有補完として Windows での
Git Bash フォールバックを追加 (claw 本家は Unix 前提のため)。
sandbox (Linux unshare) は将来実装。
"""
import os
import shutil
import subprocess
import uuid
from typing import Optional


# ============================================================
# Noetic 固有補完: Windows での bash 実行ファイル探索
# ============================================================
# claw 本家 (bash.rs) は `which bash` 相当のみで探索し、Windows に
# bash 実行ファイルが PATH 上にない環境では常に failure する。
# Noetic は Windows 常駐前提なので、Git Bash の典型的なインストール先も
# fallback として探す。PATH 最優先で、見つからない場合のみ候補探索。
# ============================================================

_WINDOWS_BASH_CANDIDATES = (
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files\Git\usr\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
)


def _find_bash_executable() -> Optional[str]:
    """bash 実行ファイルのフルパスを返す。

    探索順:
      1. shutil.which("bash")  — PATH (Unix / Git Bash が PATH にある Windows)
      2. Windows のみ: Git Bash の典型的インストール先を候補探索
      3. 見つからなければ None

    Returns:
        bash 実行ファイルのフルパス、または None。
    """
    path = shutil.which("bash")
    if path:
        return path
    if os.name == "nt":
        for cand in _WINDOWS_BASH_CANDIDATES:
            if os.path.isfile(cand):
                return cand
    return None

from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


# In-memory background task registry
_bg_tasks: dict = {}


# ============================================================
# bash
# ============================================================

def bash(inp: dict) -> str:
    command = (inp.get("command") or "").strip()
    if not command:
        return "Error: command is required"

    timeout = int(inp.get("timeout", 120))
    timeout = max(1, min(timeout, 600))
    run_bg = bool(inp.get("run_in_background", False))

    bash_path = _find_bash_executable()
    if not bash_path:
        return (
            "Error: bash not found. "
            "Windows の場合は Git for Windows (https://git-scm.com/) を "
            "インストールしてください (C:\\Program Files\\Git\\bin\\bash.exe 等を自動検出)。"
        )

    if run_bg:
        return _spawn_background([bash_path, "-c", command])

    try:
        result = subprocess.run(
            [bash_path, "-c", command],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: timeout after {timeout}s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    return _format_result(result.stdout, result.stderr, result.returncode)


# ============================================================
# PowerShell
# ============================================================

def powershell(inp: dict) -> str:
    command = (inp.get("command") or "").strip()
    if not command:
        return "Error: command is required"

    timeout = int(inp.get("timeout", 120))
    timeout = max(1, min(timeout, 600))
    run_bg = bool(inp.get("run_in_background", False))

    pwsh_path = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh_path:
        return "Error: powershell not found in PATH"

    args = [pwsh_path, "-NoProfile", "-Command", command]
    if run_bg:
        return _spawn_background(args)

    try:
        result = subprocess.run(
            args,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: timeout after {timeout}s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    return _format_result(result.stdout, result.stderr, result.returncode)


# ============================================================
# REPL
# ============================================================

import sys as _sys


# claw-code が想定する language → interpreter コマンドのマップ
# Windows では python3 が Microsoft Store stub のことがあるので python 優先。
_REPL_INTERPRETERS = {
    "python": [[_sys.executable, "-c"], ["python", "-c"], ["python3", "-c"]],
    "node":   [["node", "-e"]],
    "ruby":   [["ruby", "-e"]],
    "bash":   [["bash", "-c"]],
    "sh":     [["sh", "-c"]],
}


def repl(inp: dict) -> str:
    code = inp.get("code") or ""
    language = (inp.get("language") or "").strip().lower()
    timeout_ms = int(inp.get("timeout_ms", 30000))
    timeout = max(1, timeout_ms // 1000)

    if not code:
        return "Error: code is required"
    if not language:
        return "Error: language is required"

    interp_options = _REPL_INTERPRETERS.get(language)
    if not interp_options:
        return (f"Error: unsupported language '{language}'. "
                f"Supported: {list(_REPL_INTERPRETERS.keys())}")

    interp = None
    for option in interp_options:
        if shutil.which(option[0]):
            interp = option
            break
    if interp is None:
        return f"Error: no interpreter found for '{language}'"

    try:
        result = subprocess.run(
            interp + [code],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: timeout after {timeout}s"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    return _format_result(result.stdout, result.stderr, result.returncode)


# ============================================================
# 共通
# ============================================================

def _spawn_background(args: list) -> str:
    task_id = f"bg_{uuid.uuid4().hex[:8]}"
    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8", errors="replace",
        )
    except Exception as e:
        return f"Error: spawn failed: {e}"
    _bg_tasks[task_id] = {"proc": proc, "args": args}
    return f"Started in background: backgroundTaskId={task_id} pid={proc.pid}"


def _format_result(stdout: str, stderr: str, returncode: int) -> str:
    parts: list = []
    if stdout:
        parts.append(f"[stdout]\n{stdout.rstrip()}")
    if stderr:
        parts.append(f"[stderr]\n{stderr.rstrip()}")
    if not parts:
        parts.append("(no output)")
    if returncode != 0:
        parts.append(f"[exit code: {returncode}]")
    return "\n".join(parts)


def get_background_task(task_id: str) -> Optional[dict]:
    return _bg_tasks.get(task_id)


# ============================================================
# register
# ============================================================

def register(registry: ToolRegistry) -> None:
    specs = [
        ToolSpec(
            name="bash",
            description=("Execute a shell command via bash. "
                         "Use timeout (seconds) or run_in_background for long tasks."),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1,
                                "maximum": 600,
                                "description": "Seconds (default 120, max 600)"},
                    "run_in_background": {"type": "boolean"},
                    "description": {"type": "string",
                                    "description": "What this command does (display only)"},
                },
                "required": ["command"],
            },
            required_permission=PermissionMode.DANGER_FULL_ACCESS,
            handler=bash,
        ),
        ToolSpec(
            name="PowerShell",
            description="Execute a PowerShell command (Windows).",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
                    "run_in_background": {"type": "boolean"},
                },
                "required": ["command"],
            },
            required_permission=PermissionMode.DANGER_FULL_ACCESS,
            handler=powershell,
        ),
        ToolSpec(
            name="REPL",
            description=("Execute a small code snippet in the specified language. "
                         "Supported: python, node, ruby, bash, sh."),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "language": {"type": "string",
                                 "enum": ["python", "node", "ruby", "bash", "sh"]},
                    "timeout_ms": {"type": "integer", "minimum": 100,
                                   "maximum": 600000},
                },
                "required": ["code", "language"],
            },
            required_permission=PermissionMode.DANGER_FULL_ACCESS,
            handler=repl,
        ),
    ]
    for s in specs:
        registry.register(s)
