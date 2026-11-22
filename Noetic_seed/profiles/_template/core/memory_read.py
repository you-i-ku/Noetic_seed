"""JSONL memory reads with opt-in factual diagnostics for tool callers."""
import json
from dataclasses import dataclass

from core.runtime.registry import ToolError, ToolResult


@dataclass
class MemoryReadReport:
    records_read: int = 0
    rows_unreadable: int = 0
    files_unreadable: int = 0
    empty_files: int = 0

    def result(self, message):
        if not (self.rows_unreadable or self.files_unreadable):
            return message
        detail = dict(vars(self))
        if not (self.records_read or self.empty_files):
            raise ToolError(message, detail=detail)
        return ToolResult(message, detail=detail)


def read_memory_jsonl(path, *, report=None, reverse=False, limit=None):
    """Missing/blank sources are empty; malformed nonblank rows are counted.

    Only consumed rows count: a caller's limit is an intentional stop, not a
    read failure. Unexpected exceptions (including ToolError) propagate.
    """
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        if report is not None:
            report.files_unreadable += 1
        return []
    if not any(line.strip() for line in lines) and report is not None:
        report.empty_files += 1
    entries = []
    for line in reversed(lines) if reverse else lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if not isinstance(entry, dict):
                raise ValueError("memory row is not an object")
        except ValueError:
            if report is not None:
                report.rows_unreadable += 1
            continue
        entries.append(entry)
        if report is not None:
            report.records_read += 1
        if limit is not None and len(entries) >= limit:
            break
    return entries
