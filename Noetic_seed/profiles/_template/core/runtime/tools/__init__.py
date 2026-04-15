"""v0.5 Tools (claw-code 厳密準拠)。

Noetic 固有機能 (recall / modify_self / camera_stream 等) は含めない。
それらは別モジュール (将来的に _noetic_ext/) に隔離する。

構成:
  file_ops.py  — read_file / write_file / edit_file / glob_search / grep_search
  shell.py     — bash / PowerShell / REPL
  web.py       — WebFetch / WebSearch / RemoteTrigger
  task.py      — TaskCreate/Get/List/Stop/Update/Output
  worker.py    — WorkerCreate/Get/Observe/... (9 tool)
  team_cron.py — TeamCreate/Delete, CronCreate/Delete/List
  lsp.py       — LSP
  mcp.py       — MCP / ListMcpResources / ReadMcpResource / McpAuth
  ui.py        — AskUserQuestion / SendUserMessage / StructuredOutput / Config
  plan.py      — EnterPlanMode / ExitPlanMode
  skill.py     — Skill / Agent / ToolSearch
  util.py      — Sleep / TodoWrite / NotebookEdit / TestingPermission / RunTaskPacket

register_all(registry, workspace_root) で全 tool を一括登録する。
"""
from pathlib import Path

from core.runtime.registry import ToolRegistry


def register_all(registry: ToolRegistry, workspace_root: Path) -> None:
    """claw-code の全 tool を registry に登録。

    workspace_root: file_ops の境界として使う (symlink escape 防止)。
    """
    from core.runtime.tools import file_ops, shell, web
    file_ops.register(registry, workspace_root)
    shell.register(registry)
    web.register(registry)
