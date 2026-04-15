"""v0.5 Tools (claw-code 厳密準拠)。

Noetic 固有機能 (recall / modify_self / camera_stream 等) は含めない。
それらは別モジュール (将来的に _noetic_ext/) に隔離する。

構成 (12 module / 50 tool):
  file_ops.py   (5)  read_file / write_file / edit_file / glob_search / grep_search
  shell.py      (3)  bash / PowerShell / REPL
  web.py        (3)  WebFetch / WebSearch / RemoteTrigger
  task.py       (6)  TaskCreate/Get/List/Stop/Update/Output
  worker.py     (9)  WorkerCreate/Get/Observe/ResolveTrust/AwaitReady/
                     SendPrompt/Restart/Terminate/ObserveCompletion
  team_cron.py  (5)  TeamCreate/Delete, CronCreate/List/Delete
  lsp.py        (1)  LSP
  mcp.py        (4)  MCP / ListMcpResources / ReadMcpResource / McpAuth
  ui.py         (4)  AskUserQuestion / SendUserMessage /
                     StructuredOutput / Config
  plan.py       (2)  EnterPlanMode / ExitPlanMode
  skill.py      (3)  Skill / Agent / ToolSearch
  util.py       (5)  Sleep / TodoWrite / NotebookEdit /
                     TestingPermission / RunTaskPacket

register_all(registry, workspace_root, settings_path, skill_dirs) で
全 tool を一括登録する。
"""
from pathlib import Path
from typing import Optional

from core.runtime.registry import ToolRegistry


def register_all(
    registry: ToolRegistry,
    workspace_root: Path,
    settings_path: Optional[Path] = None,
    skill_dirs: Optional[list] = None,
) -> None:
    """claw-code の全 tool (50) を registry に登録。

    workspace_root: file_ops の境界
    settings_path:  Config tool の対象 (デフォルト workspace_root/settings.json)
    skill_dirs:     Skill tool の検索パス群
    """
    from core.runtime.tools import (
        file_ops, shell, web, task, worker, team_cron,
        lsp, mcp, ui, plan, skill, util,
    )

    if settings_path is None:
        settings_path = workspace_root / "settings.json"
    if skill_dirs is None:
        skill_dirs = [str(workspace_root / ".claude" / "skills"),
                      str(workspace_root / "skills")]

    file_ops.register(registry, workspace_root)
    shell.register(registry)
    web.register(registry)
    task.register(registry)
    worker.register(registry)
    team_cron.register(registry)
    lsp.register(registry)
    mcp.register(registry)
    ui.register(registry, settings_path)
    plan.register(registry)
    skill.register(registry, skill_dirs)
    util.register(registry, workspace_root)
