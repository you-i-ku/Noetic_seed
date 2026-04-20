"""Task — TaskCreate / TaskGet / TaskList / TaskStop / TaskUpdate / TaskOutput.

claw-code 参照: rust/crates/runtime/src/task_registry.rs:1-336
in-memory thread-safe registry での task ライフサイクル管理。
"""
import threading
import time
import uuid
from dataclasses import dataclass, field

from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


@dataclass
class TaskRecord:
    id: str
    description: str
    status: str = "created"  # created | running | stopped | completed | failed
    output: list = field(default_factory=list)  # 文字列の append-only ログ
    messages: list = field(default_factory=list)  # update で送られた指示
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class _TaskRegistry:
    def __init__(self):
        self._lock = threading.RLock()
        self._tasks: dict = {}

    def create(self, description: str) -> TaskRecord:
        with self._lock:
            tid = f"task_{uuid.uuid4().hex[:8]}"
            rec = TaskRecord(id=tid, description=description, status="created")
            self._tasks[tid] = rec
            return rec

    def get(self, tid: str):
        with self._lock:
            return self._tasks.get(tid)

    def list_all(self) -> list:
        with self._lock:
            return list(self._tasks.values())

    def stop(self, tid: str) -> bool:
        with self._lock:
            rec = self._tasks.get(tid)
            if not rec:
                return False
            rec.status = "stopped"
            rec.updated_at = time.time()
            return True

    def update(self, tid: str, message: str) -> bool:
        with self._lock:
            rec = self._tasks.get(tid)
            if not rec:
                return False
            rec.messages.append(message)
            rec.updated_at = time.time()
            return True

    def append_output(self, tid: str, line: str) -> bool:
        with self._lock:
            rec = self._tasks.get(tid)
            if not rec:
                return False
            rec.output.append(line)
            rec.updated_at = time.time()
            return True


_registry = _TaskRegistry()


def get_task_registry() -> _TaskRegistry:
    """外部から task を操作するためのアクセサ。"""
    return _registry


# ============================================================
# Tools
# ============================================================

def task_create(inp: dict) -> str:
    description = (inp.get("description") or "").strip()
    if not description:
        return "Error: description is required"
    rec = _registry.create(description)
    return f"Task created: id={rec.id} description={rec.description[:100]}"


def task_get(inp: dict) -> str:
    tid = (inp.get("task_id") or "").strip()
    if not tid:
        return "Error: task_id is required"
    rec = _registry.get(tid)
    if not rec:
        return f"Error: task '{tid}' not found"
    return (f"Task {rec.id}\n"
            f"  description: {rec.description}\n"
            f"  status: {rec.status}\n"
            f"  output_lines: {len(rec.output)}\n"
            f"  messages: {len(rec.messages)}")


def task_list(inp: dict) -> str:
    tasks = _registry.list_all()
    if not tasks:
        return "No tasks."
    lines = [f"Tasks ({len(tasks)}):"]
    for t in tasks:
        lines.append(f"  {t.id}  [{t.status}]  {t.description[:80]}")
    return "\n".join(lines)


def task_stop(inp: dict) -> str:
    tid = (inp.get("task_id") or "").strip()
    if not tid:
        return "Error: task_id is required"
    if not _registry.stop(tid):
        return f"Error: task '{tid}' not found"
    return f"Task {tid} stopped"


def task_update(inp: dict) -> str:
    tid = (inp.get("task_id") or "").strip()
    message = inp.get("message") or ""
    if not tid:
        return "Error: task_id is required"
    if not message:
        return "Error: message is required"
    if not _registry.update(tid, message):
        return f"Error: task '{tid}' not found"
    return f"Sent message to task {tid}"


def task_output(inp: dict) -> str:
    tid = (inp.get("task_id") or "").strip()
    if not tid:
        return "Error: task_id is required"
    rec = _registry.get(tid)
    if not rec:
        return f"Error: task '{tid}' not found"
    if not rec.output:
        return f"Task {tid}: (no output yet)"
    return f"Task {tid} output:\n" + "\n".join(rec.output[-100:])


# ============================================================
# register
# ============================================================

def register(registry: ToolRegistry) -> None:
    specs = [
        ToolSpec(
            name="TaskCreate",
            description="Create a background task.",
            input_schema={
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
            },
            required_permission=PermissionMode.DANGER_FULL_ACCESS,
            handler=task_create,
        ),
        ToolSpec(
            name="TaskGet",
            description="Get task details by ID.",
            input_schema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=task_get,
        ),
        ToolSpec(
            name="TaskList",
            description="List all tasks.",
            input_schema={"type": "object", "properties": {}},
            required_permission=PermissionMode.READ_ONLY,
            handler=task_list,
        ),
        ToolSpec(
            name="TaskStop",
            description="Stop a running task.",
            input_schema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
            required_permission=PermissionMode.DANGER_FULL_ACCESS,
            handler=task_stop,
        ),
        ToolSpec(
            name="TaskUpdate",
            description="Send a message/update to a running task.",
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["task_id", "message"],
            },
            required_permission=PermissionMode.DANGER_FULL_ACCESS,
            handler=task_update,
        ),
        ToolSpec(
            name="TaskOutput",
            description="Retrieve a task's output log.",
            input_schema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=task_output,
        ),
    ]
    for s in specs:
        registry.register(s)
