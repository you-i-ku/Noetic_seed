"""File Operations — read_file / write_file / edit_file / glob_search / grep_search.

claw-code 参照: rust/crates/runtime/src/file_ops.rs:1-744
厳密 claw-code 準拠。Noetic 固有セキュリティは一切含めない。

workspace_root を constructor 的に渡すことで、path traversal を防ぐ。
バイナリ判定 (NUL byte)、最大サイズ (10 MB)、行指定 offset/limit をサポート。
"""
import re
from pathlib import Path
from typing import Callable, Optional

from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry, ToolError, ToolResult
from core.runtime.tool_schema import ToolSpec


# claw-code の制限値
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
BINARY_CHECK_BYTES = 8192         # 先頭 8KB で NUL byte 検査


# ============================================================
# 内部ヘルパ
# ============================================================

def _resolve_and_check(workspace_root: Path, path_str: str) -> Optional[Path]:
    """workspace_root 配下に正規化。境界外なら None。"""
    root = workspace_root.resolve()
    target = (root / path_str).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    if target.is_relative_to((root / "memory" / "state_generations").resolve()):
        return None
    return target


def _is_binary(data: bytes) -> bool:
    """先頭 N byte に NUL があれば binary 判定。"""
    return b"\x00" in data[:BINARY_CHECK_BYTES]


# ============================================================
# read_file
# ============================================================

def _make_read_file(workspace_root: Path) -> Callable:
    def read_file(inp: dict) -> str:
        path = (inp.get("path") or "").strip()
        if not path:
            raise ToolError("Error: path is required")

        try:
            offset = int(inp.get("offset", 0))
            limit = inp.get("limit")
            if limit is not None:
                limit = int(limit)
        except (TypeError, ValueError) as e:
            raise ToolError(f"Error: invalid offset/limit: {e}") from e
        if offset < 0 or (limit is not None and limit < 1):
            raise ToolError("Error: offset must be non-negative and limit must be positive")

        target = _resolve_and_check(workspace_root, path)
        if target is None:
            raise ToolError(f"Error: path '{path}' is outside workspace")
        if not target.exists():
            raise ToolError(f"Error: file not found: {path}")
        if not target.is_file():
            raise ToolError(f"Error: not a file: {path}")

        try:
            size = target.stat().st_size
            if size > MAX_FILE_SIZE:
                raise ToolError(f"Error: file too large ({size} bytes, max {MAX_FILE_SIZE})")
            raw = target.read_bytes()
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Error: read failed: {e}")

        if _is_binary(raw):
            raise ToolError(f"Error: binary file not supported: {path}")

        try:
            text = raw.decode("utf-8", errors="replace")
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Error: decode failed: {e}")

        lines = text.splitlines()
        total = len(lines)
        sliced = lines[offset:] if limit is None else lines[offset:offset + limit]
        header = f"[{path} | lines {offset+1}-{offset+len(sliced)}/{total}]\n"
        return header + "\n".join(sliced)

    return read_file


# ============================================================
# write_file
# ============================================================

def _make_write_file(workspace_root: Path) -> Callable:
    def write_file(inp: dict) -> str:
        path = (inp.get("path") or "").strip()
        content = inp.get("content", "")
        if not path:
            raise ToolError("Error: path is required")
        if not isinstance(content, str):
            raise ToolError("Error: content must be a string")

        target = _resolve_and_check(workspace_root, path)
        if target is None:
            raise ToolError(f"Error: path '{path}' is outside workspace")

        if len(content.encode("utf-8")) > MAX_FILE_SIZE:
            raise ToolError(f"Error: content too large (max {MAX_FILE_SIZE} bytes)")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Wrote {target.relative_to(workspace_root).as_posix()} ({len(content)} chars)"
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Error: write failed: {e}")

    return write_file


# ============================================================
# edit_file
# ============================================================

def _make_edit_file(workspace_root: Path) -> Callable:
    def edit_file(inp: dict) -> str:
        path = (inp.get("path") or "").strip()
        old_string = inp.get("old_string", "")
        new_string = inp.get("new_string", "")
        replace_all = bool(inp.get("replace_all", False))

        if not path:
            raise ToolError("Error: path is required")
        if old_string == "":
            raise ToolError("Error: old_string is required")
        if not isinstance(old_string, str) or not isinstance(new_string, str):
            raise ToolError("Error: old_string and new_string must be strings")

        target = _resolve_and_check(workspace_root, path)
        if target is None:
            raise ToolError(f"Error: path '{path}' is outside workspace")
        if not target.exists():
            raise ToolError(f"Error: file not found: {path}")

        try:
            original = target.read_text(encoding="utf-8")
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Error: read failed: {e}")

        count = original.count(old_string)
        if count == 0:
            raise ToolError(f"Error: old_string not found in {path}")
        if count > 1 and not replace_all:
            raise ToolError((f"Error: old_string matches {count} times. "
                    f"Use replace_all=true or make old_string more specific."))

        new_content = (original.replace(old_string, new_string)
                       if replace_all
                       else original.replace(old_string, new_string, 1))

        try:
            target.write_text(new_content, encoding="utf-8")
            delta = len(new_content) - len(original)
            return (f"Edited {target.relative_to(workspace_root).as_posix()} "
                    f"({count} replacement(s), {delta:+d} chars)")
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Error: write failed: {e}")

    return edit_file


# ============================================================
# glob_search
# ============================================================

def _make_glob_search(workspace_root: Path) -> Callable:
    def glob_search(inp: dict) -> str:
        pattern = (inp.get("pattern") or "").strip()
        if not pattern:
            raise ToolError("Error: pattern is required")

        start_raw = inp.get("path") or ""
        if start_raw:
            start = _resolve_and_check(workspace_root, start_raw)
            if start is None:
                raise ToolError(f"Error: path '{start_raw}' is outside workspace")
        else:
            start = workspace_root.resolve()

        if not start.exists():
            raise ToolError(f"Error: path not found: {start_raw}")
        if not start.is_dir():
            raise ToolError(f"Error: not a directory: {start_raw}")

        try:
            matches = sorted(start.glob(pattern))
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Error: glob failed: {e}")

        results: list = []
        for f in matches:
            if _resolve_and_check(workspace_root, str(f)) is None:
                continue
            if not f.is_file():
                continue
            try:
                rel = f.relative_to(workspace_root).as_posix()
                results.append(rel)
            except ValueError:
                continue

        if not results:
            return f"No matches for pattern: {pattern}"

        shown = results[:100]
        more = len(results) - len(shown)
        lines = [f"Found {len(results)} file(s) matching '{pattern}':"]
        lines.extend(f"  {p}" for p in shown)
        if more > 0:
            lines.append(f"  ... ({more} more not shown)")
        return "\n".join(lines)

    return glob_search


# ============================================================
# grep_search
# ============================================================

def _make_grep_search(workspace_root: Path) -> Callable:
    def grep_search(inp: dict) -> str:
        pattern = inp.get("pattern") or ""
        if not pattern:
            raise ToolError("Error: pattern is required")

        flags = re.MULTILINE
        if inp.get("-i") or inp.get("ignore_case"):
            flags |= re.IGNORECASE
        if inp.get("multiline"):
            flags |= re.DOTALL

        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            raise ToolError(f"Error: invalid regex: {e}")

        try:
            head_limit = int(inp.get("head_limit", 100))
        except (TypeError, ValueError) as e:
            raise ToolError(f"Error: invalid head_limit: {e}") from e
        if head_limit < 1:
            head_limit = 100

        glob_pat = inp.get("glob") or "**/*"

        start_raw = inp.get("path") or ""
        if start_raw:
            start = _resolve_and_check(workspace_root, start_raw)
            if start is None:
                raise ToolError(f"Error: path '{start_raw}' is outside workspace")
        else:
            start = workspace_root.resolve()

        if not start.exists():
            raise ToolError(f"Error: path not found: {start_raw}")
        if not start.is_dir():
            raise ToolError(f"Error: not a directory: {start_raw}")

        results: list = []
        total = 0
        readable = unreadable = 0
        try:
            candidates = sorted(start.glob(glob_pat))
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Error: glob failed: {e}")

        for f in candidates:
            if total >= head_limit:
                break
            if _resolve_and_check(workspace_root, str(f)) is None:
                continue
            if not f.is_file():
                continue
            try:
                raw = f.read_bytes()
                if len(raw) > MAX_FILE_SIZE:
                    continue
                if _is_binary(raw):
                    continue
                text = raw.decode("utf-8", errors="replace")
            except ToolError:
                raise
            except Exception:
                unreadable += 1
                continue
            readable += 1
            try:
                rel = f.relative_to(workspace_root).as_posix()
            except ValueError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append(f"{rel}:{i}: {line}")
                    total += 1
                    if total >= head_limit:
                        break

        message = (f"Found {total} match(es) for '{pattern}':\n" + "\n".join(results)
                   if results else f"No matches for pattern: {pattern}")
        if unreadable:
            detail = {"unreadable_files": unreadable, "readable_files": readable}
            if not readable:
                raise ToolError(message, detail=detail)
            return ToolResult(message, detail=detail)
        return message

    return grep_search


# ============================================================
# register
# ============================================================

def register(registry: ToolRegistry, workspace_root: Path) -> None:
    """5 tool を registry に登録。"""
    specs = [
        ToolSpec(
            name="read_file",
            description="Read the contents of a file. Supports offset/limit for partial reads.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "Path relative to workspace root"},
                    "offset": {"type": "integer", "minimum": 0,
                               "description": "Starting line (0-indexed)"},
                    "limit": {"type": "integer", "minimum": 1,
                              "description": "Number of lines to read"},
                },
                "required": ["path"],
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=_make_read_file(workspace_root),
        ),
        ToolSpec(
            name="write_file",
            description="Write content to a file (creates or overwrites).",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            required_permission=PermissionMode.WORKSPACE_WRITE,
            handler=_make_write_file(workspace_root),
        ),
        ToolSpec(
            name="edit_file",
            description=("Edit a file by replacing old_string with new_string. "
                         "Set replace_all=true to replace all occurrences."),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            required_permission=PermissionMode.WORKSPACE_WRITE,
            handler=_make_edit_file(workspace_root),
        ),
        ToolSpec(
            name="glob_search",
            description="Find files matching a glob pattern (e.g. '**/*.py').",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string",
                             "description": "Search root (default: workspace root)"},
                },
                "required": ["pattern"],
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=_make_glob_search(workspace_root),
        ),
        ToolSpec(
            name="grep_search",
            description="Search file contents using a regular expression.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string",
                             "description": "File name glob (e.g. '*.py')"},
                    "-i": {"type": "boolean",
                           "description": "Case insensitive"},
                    "head_limit": {"type": "integer", "minimum": 1,
                                   "maximum": 1000},
                    "multiline": {"type": "boolean",
                                  "description": "Let . match newlines"},
                },
                "required": ["pattern"],
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=_make_grep_search(workspace_root),
        ),
    ]
    for s in specs:
        registry.register(s)
