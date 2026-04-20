"""Team & Cron — TeamCreate / TeamDelete / CronCreate / CronList / CronDelete.

claw-code 参照: rust/crates/runtime/src/team_cron_registry.rs:1-363
in-memory registry。実際の並列実行・cron スケジューラは将来実装。
Phase 2 では state 管理のみ。
"""
import re
import threading
import time
import uuid
from dataclasses import dataclass, field

from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


# ============================================================
# Team
# ============================================================

@dataclass
class TeamRecord:
    id: str
    name: str
    members: list = field(default_factory=list)  # list of agent specs
    tasks: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class _TeamRegistry:
    def __init__(self):
        self._lock = threading.RLock()
        self._teams: dict = {}

    def create(self, name: str, members: list, tasks: list) -> TeamRecord:
        with self._lock:
            tid = f"team_{uuid.uuid4().hex[:8]}"
            rec = TeamRecord(id=tid, name=name,
                             members=list(members), tasks=list(tasks))
            self._teams[tid] = rec
            return rec

    def delete(self, tid: str) -> bool:
        with self._lock:
            return self._teams.pop(tid, None) is not None

    def list_all(self) -> list:
        with self._lock:
            return list(self._teams.values())


_team_registry = _TeamRegistry()


def get_team_registry() -> _TeamRegistry:
    return _team_registry


def team_create(inp: dict) -> str:
    name = (inp.get("name") or "").strip()
    members = inp.get("members") or []
    tasks = inp.get("tasks") or []
    if not name:
        return "Error: name is required"
    if not isinstance(members, list) or not members:
        return "Error: members (non-empty list) is required"
    rec = _team_registry.create(name, members, tasks)
    return (f"Team created: id={rec.id} name={rec.name} "
            f"members={len(rec.members)} tasks={len(rec.tasks)}")


def team_delete(inp: dict) -> str:
    tid = (inp.get("team_id") or "").strip()
    if not tid:
        return "Error: team_id is required"
    if not _team_registry.delete(tid):
        return f"Error: team '{tid}' not found"
    return f"Team {tid} deleted"


# ============================================================
# Cron
# ============================================================

@dataclass
class CronRecord:
    id: str
    schedule: str  # 5-field cron format
    prompt: str
    description: str = ""
    created_at: float = field(default_factory=time.time)


_CRON_FIELD_RE = re.compile(r"^[\d\*/,\-]+$")


def _validate_cron(schedule: str) -> bool:
    """5-field cron format の簡易検証 (* / , - / 数字のみ)。"""
    fields = schedule.split()
    if len(fields) != 5:
        return False
    return all(_CRON_FIELD_RE.match(f) for f in fields)


class _CronRegistry:
    def __init__(self):
        self._lock = threading.RLock()
        self._jobs: dict = {}

    def create(self, schedule: str, prompt: str,
               description: str = "") -> CronRecord:
        with self._lock:
            cid = f"cron_{uuid.uuid4().hex[:8]}"
            rec = CronRecord(id=cid, schedule=schedule, prompt=prompt,
                             description=description)
            self._jobs[cid] = rec
            return rec

    def delete(self, cid: str) -> bool:
        with self._lock:
            return self._jobs.pop(cid, None) is not None

    def list_all(self) -> list:
        with self._lock:
            return list(self._jobs.values())


_cron_registry = _CronRegistry()


def get_cron_registry() -> _CronRegistry:
    return _cron_registry


def cron_create(inp: dict) -> str:
    schedule = (inp.get("schedule") or "").strip()
    prompt = inp.get("prompt") or ""
    description = inp.get("description") or ""
    if not schedule:
        return "Error: schedule is required"
    if not prompt:
        return "Error: prompt is required"
    if not _validate_cron(schedule):
        return f"Error: invalid cron schedule '{schedule}' (need 5 fields)"
    rec = _cron_registry.create(schedule, prompt, description)
    return f"Cron created: id={rec.id} schedule={rec.schedule}"


def cron_list(inp: dict) -> str:
    jobs = _cron_registry.list_all()
    if not jobs:
        return "No cron jobs."
    lines = [f"Cron jobs ({len(jobs)}):"]
    for j in jobs:
        lines.append(f"  {j.id}  [{j.schedule}]  {j.prompt[:60]}"
                     + (f"  ({j.description})" if j.description else ""))
    return "\n".join(lines)


def cron_delete(inp: dict) -> str:
    cid = (inp.get("cron_id") or "").strip()
    if not cid:
        return "Error: cron_id is required"
    if not _cron_registry.delete(cid):
        return f"Error: cron '{cid}' not found"
    return f"Cron {cid} deleted"


# ============================================================
# register
# ============================================================

def register(registry: ToolRegistry) -> None:
    danger = PermissionMode.DANGER_FULL_ACCESS
    ro = PermissionMode.READ_ONLY
    specs = [
        ToolSpec(name="TeamCreate",
                 description="Create a sub-agent team for parallel execution.",
                 input_schema={
                     "type": "object",
                     "properties": {
                         "name": {"type": "string"},
                         "members": {"type": "array"},
                         "tasks": {"type": "array"},
                     },
                     "required": ["name", "members"],
                 },
                 required_permission=danger, handler=team_create),
        ToolSpec(name="TeamDelete",
                 description="Delete a team and stop its tasks.",
                 input_schema={"type": "object",
                               "properties": {"team_id": {"type": "string"}},
                               "required": ["team_id"]},
                 required_permission=danger, handler=team_delete),
        ToolSpec(name="CronCreate",
                 description="Create a scheduled cron job (5-field format).",
                 input_schema={
                     "type": "object",
                     "properties": {
                         "schedule": {"type": "string"},
                         "prompt": {"type": "string"},
                         "description": {"type": "string"},
                     },
                     "required": ["schedule", "prompt"],
                 },
                 required_permission=danger, handler=cron_create),
        ToolSpec(name="CronList",
                 description="List all cron jobs.",
                 input_schema={"type": "object", "properties": {}},
                 required_permission=ro, handler=cron_list),
        ToolSpec(name="CronDelete",
                 description="Delete a cron job.",
                 input_schema={"type": "object",
                               "properties": {"cron_id": {"type": "string"}},
                               "required": ["cron_id"]},
                 required_permission=danger, handler=cron_delete),
    ]
    for s in specs:
        registry.register(s)
