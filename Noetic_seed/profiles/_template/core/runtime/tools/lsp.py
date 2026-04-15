"""LSP — Language Server Protocol tool.

claw-code 参照: rust/crates/runtime/src/lsp_client.rs:1-438

Phase 2 では LSP client の本実装はせず、interface と dispatch 層のみ提供。
外部から set_lsp_backend() で実 LSP client を差し込む設計。
"""
from typing import Callable, Optional

from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


_backend: dict = {"client": None}


def set_lsp_backend(client_fn: Callable) -> None:
    """LSP 実 backend を注入。callable(action, **kwargs) -> str。"""
    _backend["client"] = client_fn


_VALID_ACTIONS = {"symbols", "references", "diagnostics",
                  "definition", "hover"}


def lsp_dispatch(inp: dict) -> str:
    action = (inp.get("action") or "").strip().lower()
    if not action:
        return "Error: action is required"
    if action not in _VALID_ACTIONS:
        return f"Error: unknown action '{action}'. Valid: {sorted(_VALID_ACTIONS)}"

    backend = _backend.get("client")
    if backend is None:
        return (f"[LSP pending — backend not configured]\n"
                f"action: {action}\n"
                f"path: {inp.get('path', '')}\n"
                f"line: {inp.get('line', '')}\n"
                f"character: {inp.get('character', '')}\n"
                f"query: {inp.get('query', '')}")

    try:
        return backend(action, **{k: v for k, v in inp.items()
                                   if k != "action"})
    except Exception as e:
        return f"Error: LSP backend failed: {e}"


def register(registry: ToolRegistry) -> None:
    registry.register(ToolSpec(
        name="LSP",
        description=("Query Language Server Protocol. "
                     "Actions: symbols, references, diagnostics, definition, hover."),
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string",
                           "enum": list(sorted(_VALID_ACTIONS))},
                "path": {"type": "string"},
                "line": {"type": "integer"},
                "character": {"type": "integer"},
                "query": {"type": "string"},
            },
            "required": ["action"],
        },
        required_permission=PermissionMode.READ_ONLY,
        handler=lsp_dispatch,
    ))
