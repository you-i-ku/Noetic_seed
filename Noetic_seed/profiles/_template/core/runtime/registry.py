"""Tool Registry — 名前 → ToolSpec のマップ。

claw-code の global_tool_registry (rust/crates/tools/src/lib.rs:1-80) +
mcp_tool_bridge (rust/crates/runtime/src/mcp_tool_bridge.rs) の Python port。

厳密 claw-code 準拠。Noetic 固有機能は含めない。
"""
from typing import Optional
from dataclasses import dataclass
import json
import math
import re
from urllib.parse import quote, unquote

from core.runtime.tool_schema import ToolSpec


class ToolError(Exception):
    """The requested operation failed; detail contains facts supplied by the tool."""

    def __init__(self, message, detail=None):
        super().__init__(message)
        self.detail = detail


@dataclass
class ToolResult:
    """Optional detail for successful operations; ordinary string returns still work."""
    message: str
    detail: object = None


DETAIL_MAX_BYTES = 8192
DETAIL_MAX_DEPTH = 8  # root is depth 0; containers at depth 8 become null
DETAIL_MAX_ITEMS = 256  # data children; at most 2 additional marker/wrapper fields
_SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
    r"key|token|secret|password|private[_-]?key|credential)(?:$|[_-])", re.I)


def result_redactor():
    """Snapshot configured credentials once per invocation, never log their values."""
    from core.auth import _load_secrets
    values = set()
    todo = [(_load_secrets(), False)]
    while todo:
        value, sensitive = todo.pop()
        if isinstance(value, dict):
            todo.extend((v, sensitive or bool(_SECRET_KEY.search(str(k)))
                         or str(k).lower() in ("authorization", "cookie"))
                        for k, v in value.items())
        elif isinstance(value, list):
            todo.extend((v, sensitive) for v in value)
        elif sensitive and isinstance(value, str) and value:
            values.update((value, quote(value, safe="")))
    secrets = sorted(values, key=len, reverse=True)

    def redact(text):
        text = str(text)
        for secret in secrets:
            text = text.replace(secret, "[REDACTED]")
        # Header values may contain spaces and multiple cookies. Remove the whole value.
        text = re.sub(r'''(?im)(\b(?:authorization|proxy-authorization|cookie|set-cookie)["']?\s*[:=]\s*)[^\r\n]+''',
                      r"\1[REDACTED]", text)
        text = re.sub(r"(?i)(https?://)[^\s/@]+@", r"\1[REDACTED]@", text)
        text = re.sub(r"([?&])([^\s=&#]+)=([^\s&#]*)",
                      lambda m: (m[1] + m[2] + "=[REDACTED]"
                                 if _SECRET_KEY.search(unquote(m[2])) else m[0]), text)
        return text
    return redact


def normalize_detail(detail, redact):
    """Validate/redact the entire value BEFORE bounding it. No repr of bad values.

    Iterative traversal handles very deep JSON and detects cycles even in omitted
    branches. Invalid data becomes null plus a fixed, non-sensitive reason.
    """
    if detail is None:
        return None, None
    root = [None]
    active = set()
    stack = [(detail, root, 0, False)]
    try:
        while stack:
            value, parent, key, leaving = stack.pop()
            if leaving:
                active.remove(id(value))
                continue
            if isinstance(value, (dict, list, tuple)):
                if id(value) in active:
                    return None, "non_json_detail"
                active.add(id(value))
                stack.append((value, None, None, True))
                target = {} if isinstance(value, dict) else [None] * len(value)
                parent[key] = target
                if isinstance(value, dict):
                    for k, v in value.items():
                        if not isinstance(k, str):
                            if k is not None and type(k) not in (bool, int, float):
                                return None, "non_json_detail"
                            if type(k) is float and not math.isfinite(k):
                                return None, "non_json_detail"
                            k = next(iter(json.loads(json.dumps({k: None}, allow_nan=False))))
                        clean_key = redact(k)
                        if _SECRET_KEY.search(k) or k.lower() in ("authorization", "cookie", "set-cookie", "proxy-authorization"):
                            # Still validate the original value, including invalid hidden branches.
                            stack.append((v, [None], 0, False))
                            target[clean_key] = "[REDACTED]"
                        else:
                            stack.append((v, target, clean_key, False))
                else:
                    stack.extend((v, target, i, False) for i, v in enumerate(value))
            elif isinstance(value, str):
                parent[key] = redact(value)
            elif value is None or type(value) in (bool, int):
                parent[key] = value
            elif type(value) is float and math.isfinite(value):
                parent[key] = value
            else:
                return None, "non_json_detail"

        remaining = DETAIL_MAX_ITEMS
        truncated = False
        def bound(value, depth=0):
            nonlocal remaining, truncated
            if depth >= DETAIL_MAX_DEPTH and isinstance(value, (dict, list)):
                truncated = True
                return None
            if isinstance(value, (dict, list)):
                out = {} if isinstance(value, dict) else []
                entries = value.items() if isinstance(value, dict) else enumerate(value)
                for k, v in entries:
                    if remaining == 0:
                        truncated = True
                        break
                    remaining -= 1
                    child = bound(v, depth + 1)
                    if isinstance(out, dict):
                        out[k] = child
                    else:
                        out.append(child)
                return out
            return value
        # Reserve one level for a possible marker envelope around a non-object.
        result = bound(root[0], 0 if isinstance(root[0], dict) else 1)
        if truncated:
            result = ({**result, "_truncated": True} if isinstance(result, dict)
                      else {"value": result, "_truncated": True})
        def size(value):
            return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        if size(result) > DETAIL_MAX_BYTES:
            # Remove whole values; never slice serialized JSON or leak partial secrets.
            result = ({**result, "_truncated": True} if isinstance(result, dict)
                      else {"value": result, "_truncated": True})
            for key in list(result):
                if size(result) <= DETAIL_MAX_BYTES:
                    break
                if key != "_truncated":
                    del result[key]
        return result, None
    except Exception:
        return None, "detail_normalization_failed"


def exception_detail(exc):
    """Only known factual attributes; never collect arbitrary __dict__ contents."""
    detail = {"type": type(exc).__name__, "message": str(exc)}
    for name in ("returncode", "errno", "filename", "filename2", "timeout", "cmd",
                 "stdout", "stderr", "output"):
        try:
            value = getattr(exc, name, None)
        except Exception:
            continue
        if value is not None:
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            detail[name] = value  # Redaction/bounds occur later, before any branch.
    return detail


def tool_result_body(message, is_error=False, detail=None, detail_error=None):
    """Serialize exactly at a provider boundary; internal records remain structured."""
    if not is_error and detail is None and detail_error is None:
        return message
    body = {"is_error": is_error, "message": message, "detail": detail}
    if detail_error:
        body["detail_error"] = detail_error
    return json.dumps(body, ensure_ascii=False, allow_nan=False)


class ToolRegistry:
    """tool の集中レジストリ。"""

    def __init__(self):
        self._tools: dict = {}

    def register(self, spec: ToolSpec) -> None:
        """tool を登録。同名は上書き。"""
        self._tools[spec.name] = spec

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def all_names(self) -> list:
        return list(self._tools.keys())

    def list(
        self,
        max_permission=None,  # PermissionMode
        allowlist: Optional[list] = None,
        denylist: Optional[list] = None,
    ) -> list:
        """フィルタ付き tool 一覧を返す。

        max_permission:  この permission 以下で動く tool のみ
        allowlist:       含まれる tool 名のみ
        denylist:        除外する tool 名
        """
        from core.runtime.permissions import _MODE_LEVEL

        max_level = _MODE_LEVEL.get(max_permission, 99) if max_permission else 99
        out = []
        for spec in self._tools.values():
            if allowlist is not None and spec.name not in allowlist:
                continue
            if denylist is not None and spec.name in denylist:
                continue
            spec_level = _MODE_LEVEL.get(spec.required_permission, 99)
            if spec_level > max_level:
                continue
            out.append(spec)
        return out

    def execute(self, name: str, tool_input: dict) -> str | ToolResult:
        """tool を実行して str または注釈付き ToolResult を返す。未登録なら ValueError。"""
        spec = self.get(name)
        if spec is None:
            raise ValueError(f"Tool not registered: {name}")
        return spec.handler(tool_input)

    # ---- MCP tool 用ヘルパ ----

    @staticmethod
    def mcp_tool_name(server_name: str, tool_name: str) -> str:
        """MCP tool の prefix 付き正規化名。
        claw-code/rust/crates/runtime/src/mcp.rs:26-37 に準拠。"""
        def normalize(s: str) -> str:
            return "".join(c if c.isalnum() or c == "_" else "_" for c in s)

        return f"mcp__{normalize(server_name)}__{normalize(tool_name)}"

    def is_mcp_tool(self, name: str) -> bool:
        return name.startswith("mcp__")
