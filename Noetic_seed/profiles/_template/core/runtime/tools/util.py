"""Utility — Sleep / TodoWrite / NotebookEdit / TestingPermission / RunTaskPacket.

claw-code 参照: rust/crates/tools/src/lib.rs (各 tool)

シンプルな utility 群。NotebookEdit は Jupyter .ipynb JSON を直接編集。
"""
import json
import time
from pathlib import Path
from typing import Callable

from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


# ============================================================
# Sleep
# ============================================================

def sleep(inp: dict) -> str:
    duration_ms = int(inp.get("duration_ms", 0))
    if duration_ms < 0:
        return "Error: duration_ms must be non-negative"
    if duration_ms > 60000:
        return "Error: duration_ms too large (max 60000)"
    time.sleep(duration_ms / 1000.0)
    return f"Slept {duration_ms} ms"


# ============================================================
# TodoWrite — in-memory todo list
# ============================================================

_todos: list = []


def todo_write(inp: dict) -> str:
    items = inp.get("todos", [])
    if not isinstance(items, list):
        return "Error: todos must be a list"
    global _todos
    _todos = list(items)
    return f"Todo list updated ({len(_todos)} items)"


def get_todos() -> list:
    return list(_todos)


# ============================================================
# NotebookEdit — Jupyter notebook cell edit
# ============================================================

def _make_notebook_edit(workspace_root: Path) -> Callable:
    def notebook_edit(inp: dict) -> str:
        path = (inp.get("path") or "").strip()
        cell_index = inp.get("cell_index")
        new_source = inp.get("new_source", "")
        cell_type = inp.get("cell_type")
        action = (inp.get("action") or "replace").lower()

        if not path:
            return "Error: path is required"
        if cell_index is None:
            return "Error: cell_index is required"
        try:
            cell_index = int(cell_index)
        except (TypeError, ValueError):
            return "Error: cell_index must be an integer"

        root = workspace_root.resolve()
        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return f"Error: path '{path}' is outside workspace"
        if not target.exists():
            return f"Error: notebook not found: {path}"

        try:
            nb = json.loads(target.read_text(encoding="utf-8"))
        except Exception as e:
            return f"Error: invalid notebook JSON: {e}"

        cells = nb.get("cells", [])

        if action == "insert":
            new_cell = {
                "cell_type": cell_type or "code",
                "metadata": {},
                "source": new_source if isinstance(new_source, list) else [new_source],
            }
            if new_cell["cell_type"] == "code":
                new_cell["outputs"] = []
                new_cell["execution_count"] = None
            cells.insert(min(cell_index, len(cells)), new_cell)
        elif action == "delete":
            if cell_index < 0 or cell_index >= len(cells):
                return f"Error: cell_index {cell_index} out of range"
            cells.pop(cell_index)
        else:  # replace
            if cell_index < 0 or cell_index >= len(cells):
                return f"Error: cell_index {cell_index} out of range"
            cells[cell_index]["source"] = (
                new_source if isinstance(new_source, list) else [new_source]
            )
            if cell_type:
                cells[cell_index]["cell_type"] = cell_type

        nb["cells"] = cells
        try:
            target.write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                              encoding="utf-8")
        except Exception as e:
            return f"Error: write failed: {e}"
        return f"Notebook {action} at cell {cell_index} complete"

    return notebook_edit


# ============================================================
# TestingPermission — permission 動作確認用ダミー
# ============================================================

def testing_permission(inp: dict) -> str:
    return f"TestingPermission invoked with: {json.dumps(inp, ensure_ascii=False)}"


# ============================================================
# RunTaskPacket
# ============================================================

def run_task_packet(inp: dict) -> str:
    """structured task packet を実行する (現状はメタ情報の要約のみ返す)。

    packet の各フィールド:
      objective, scope, repo, branch_policy, acceptance_tests,
      commit_policy, reporting_contract, escalation_policy
    """
    packet = inp.get("packet")
    if not isinstance(packet, dict):
        return "Error: packet must be an object"
    obj = packet.get("objective", "")
    scope = packet.get("scope", "")
    if not obj:
        return "Error: packet.objective is required"
    summary = [
        f"objective: {obj[:200]}",
        f"scope: {scope[:200]}" if scope else "",
        f"repo: {packet.get('repo', '')}",
        f"branch_policy: {packet.get('branch_policy', '')}",
        f"acceptance_tests: {len(packet.get('acceptance_tests') or [])} items",
    ]
    return "TaskPacket accepted:\n" + "\n".join(l for l in summary if l)


# ============================================================
# register
# ============================================================

def register(registry: ToolRegistry, workspace_root: Path) -> None:
    specs = [
        ToolSpec(
            name="Sleep",
            description="Sleep for the specified duration in milliseconds.",
            input_schema={
                "type": "object",
                "properties": {
                    "duration_ms": {"type": "integer", "minimum": 0,
                                    "maximum": 60000},
                },
                "required": ["duration_ms"],
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=sleep,
        ),
        ToolSpec(
            name="TodoWrite",
            description="Update the in-memory todo list.",
            input_schema={
                "type": "object",
                "properties": {
                    "todos": {"type": "array", "items": {}},
                },
                "required": ["todos"],
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=todo_write,
        ),
        ToolSpec(
            name="NotebookEdit",
            description="Edit a Jupyter notebook cell (replace/insert/delete).",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "cell_index": {"type": "integer"},
                    "new_source": {},
                    "cell_type": {"type": "string",
                                  "enum": ["code", "markdown", "raw"]},
                    "action": {"type": "string",
                               "enum": ["replace", "insert", "delete"]},
                },
                "required": ["path", "cell_index"],
            },
            required_permission=PermissionMode.WORKSPACE_WRITE,
            handler=_make_notebook_edit(workspace_root),
        ),
        ToolSpec(
            name="TestingPermission",
            description="Test permission behaviour (no-op echo tool).",
            input_schema={"type": "object"},
            required_permission=PermissionMode.READ_ONLY,
            handler=testing_permission,
        ),
        ToolSpec(
            name="RunTaskPacket",
            description="Execute a structured task packet (objective/scope/acceptance_tests/...).",
            input_schema={
                "type": "object",
                "properties": {
                    "packet": {"type": "object"},
                },
                "required": ["packet"],
            },
            required_permission=PermissionMode.DANGER_FULL_ACCESS,
            handler=run_task_packet,
        ),
    ]
    for s in specs:
        registry.register(s)
