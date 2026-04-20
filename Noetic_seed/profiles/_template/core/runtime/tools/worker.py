"""Worker — Bootstrap protocol (9 tools).

claw-code 参照: rust/crates/runtime/src/worker_boot.rs

State machine:
  Created → TrustGate → Trusted → AwaitingReady → Ready →
  PromptSent → Executing → Completed (Finished/Failed)

in-memory registry。実際の worker process spawn は将来実装。
Phase 2 では state 遷移と handshake インターフェースのみ。
"""
import threading
import time
import uuid
from dataclasses import dataclass, field

from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


# State enum (string で扱う)
S_CREATED = "created"
S_TRUST_GATE = "trust_gate"
S_TRUSTED = "trusted"
S_AWAITING_READY = "awaiting_ready"
S_READY = "ready"
S_PROMPT_SENT = "prompt_sent"
S_EXECUTING = "executing"
S_FINISHED = "finished"
S_FAILED = "failed"


@dataclass
class WorkerRecord:
    id: str
    cwd: str
    trusted_roots: list = field(default_factory=list)
    state: str = S_CREATED
    trust_prompt: str = ""
    last_snapshot: str = ""
    prompt: str = ""
    finish_reason: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class _WorkerRegistry:
    def __init__(self):
        self._lock = threading.RLock()
        self._workers: dict = {}

    def create(self, cwd: str, trusted_roots: list) -> WorkerRecord:
        with self._lock:
            wid = f"worker_{uuid.uuid4().hex[:8]}"
            rec = WorkerRecord(id=wid, cwd=cwd,
                               trusted_roots=list(trusted_roots),
                               state=S_TRUST_GATE,
                               trust_prompt=f"Trust workspace at {cwd}?")
            self._workers[wid] = rec
            return rec

    def get(self, wid: str):
        with self._lock:
            return self._workers.get(wid)

    def remove(self, wid: str) -> bool:
        with self._lock:
            return self._workers.pop(wid, None) is not None

    def transition(self, wid: str, new_state: str) -> bool:
        with self._lock:
            rec = self._workers.get(wid)
            if not rec:
                return False
            rec.state = new_state
            rec.updated_at = time.time()
            return True


_registry = _WorkerRegistry()


def get_worker_registry() -> _WorkerRegistry:
    return _registry


# ============================================================
# Tool handlers
# ============================================================

def worker_create(inp: dict) -> str:
    cwd = (inp.get("cwd") or "").strip()
    trusted_roots = inp.get("trusted_roots") or []
    if not cwd:
        return "Error: cwd is required"
    rec = _registry.create(cwd, trusted_roots)
    return f"Worker created: id={rec.id} state={rec.state} cwd={rec.cwd}"


def worker_get(inp: dict) -> str:
    wid = (inp.get("worker_id") or "").strip()
    if not wid:
        return "Error: worker_id is required"
    rec = _registry.get(wid)
    if not rec:
        return f"Error: worker '{wid}' not found"
    return (f"Worker {rec.id}\n"
            f"  state: {rec.state}\n"
            f"  cwd: {rec.cwd}\n"
            f"  trusted_roots: {rec.trusted_roots}\n"
            f"  trust_prompt: {rec.trust_prompt}\n"
            f"  finish_reason: {rec.finish_reason or '(n/a)'}")


def worker_observe(inp: dict) -> str:
    wid = (inp.get("worker_id") or "").strip()
    snapshot = inp.get("snapshot") or ""
    if not wid:
        return "Error: worker_id is required"
    rec = _registry.get(wid)
    if not rec:
        return f"Error: worker '{wid}' not found"
    rec.last_snapshot = str(snapshot)
    rec.updated_at = time.time()
    # 単純なヒューリスティック: snapshot に "ready" 含めば awaiting_ready 扱い
    txt = str(snapshot).lower()
    if rec.state == S_TRUSTED and "ready" in txt:
        rec.state = S_AWAITING_READY
    return f"Observation recorded for {wid} (state={rec.state})"


def worker_resolve_trust(inp: dict) -> str:
    wid = (inp.get("worker_id") or "").strip()
    decision = (inp.get("decision") or "").strip().lower()
    if not wid:
        return "Error: worker_id is required"
    if decision not in ("trust", "deny"):
        return "Error: decision must be 'trust' or 'deny'"
    rec = _registry.get(wid)
    if not rec:
        return f"Error: worker '{wid}' not found"
    if rec.state != S_TRUST_GATE:
        return f"Error: worker not in trust_gate state (current: {rec.state})"
    if decision == "trust":
        rec.state = S_TRUSTED
        return f"Worker {wid} trusted"
    rec.state = S_FAILED
    rec.finish_reason = "trust denied"
    return f"Worker {wid} trust denied"


def worker_await_ready(inp: dict) -> str:
    wid = (inp.get("worker_id") or "").strip()
    if not wid:
        return "Error: worker_id is required"
    rec = _registry.get(wid)
    if not rec:
        return f"Error: worker '{wid}' not found"
    if rec.state in (S_AWAITING_READY, S_READY):
        rec.state = S_READY
        return f"Worker {wid} is ready"
    return f"Worker {wid} not yet ready (state={rec.state})"


def worker_send_prompt(inp: dict) -> str:
    wid = (inp.get("worker_id") or "").strip()
    prompt = inp.get("prompt") or ""
    if not wid:
        return "Error: worker_id is required"
    if not prompt:
        return "Error: prompt is required"
    rec = _registry.get(wid)
    if not rec:
        return f"Error: worker '{wid}' not found"
    if rec.state != S_READY:
        return f"Error: worker not ready (state={rec.state})"
    rec.prompt = prompt
    rec.state = S_PROMPT_SENT
    return f"Prompt sent to {wid} ({len(prompt)} chars)"


def worker_restart(inp: dict) -> str:
    wid = (inp.get("worker_id") or "").strip()
    if not wid:
        return "Error: worker_id is required"
    rec = _registry.get(wid)
    if not rec:
        return f"Error: worker '{wid}' not found"
    rec.state = S_TRUST_GATE
    rec.last_snapshot = ""
    rec.prompt = ""
    rec.finish_reason = ""
    rec.updated_at = time.time()
    return f"Worker {wid} restarted (state=trust_gate)"


def worker_terminate(inp: dict) -> str:
    wid = (inp.get("worker_id") or "").strip()
    if not wid:
        return "Error: worker_id is required"
    if not _registry.remove(wid):
        return f"Error: worker '{wid}' not found"
    return f"Worker {wid} terminated"


def worker_observe_completion(inp: dict) -> str:
    wid = (inp.get("worker_id") or "").strip()
    finish_reason = (inp.get("finish_reason") or "").strip()
    if not wid:
        return "Error: worker_id is required"
    if finish_reason not in ("Finished", "Failed"):
        return "Error: finish_reason must be 'Finished' or 'Failed'"
    rec = _registry.get(wid)
    if not rec:
        return f"Error: worker '{wid}' not found"
    rec.state = S_FINISHED if finish_reason == "Finished" else S_FAILED
    rec.finish_reason = finish_reason
    return f"Worker {wid} completed: {finish_reason}"


# ============================================================
# register
# ============================================================

def register(registry: ToolRegistry) -> None:
    danger = PermissionMode.DANGER_FULL_ACCESS
    ro = PermissionMode.READ_ONLY
    specs = [
        ToolSpec(name="WorkerCreate", description="Create a coding worker with trust-gate.",
                 input_schema={"type": "object",
                               "properties": {"cwd": {"type": "string"},
                                              "trusted_roots": {"type": "array",
                                                                "items": {"type": "string"}}},
                               "required": ["cwd"]},
                 required_permission=danger, handler=worker_create),
        ToolSpec(name="WorkerGet", description="Get worker boot state.",
                 input_schema={"type": "object",
                               "properties": {"worker_id": {"type": "string"}},
                               "required": ["worker_id"]},
                 required_permission=ro, handler=worker_get),
        ToolSpec(name="WorkerObserve",
                 description="Feed a terminal snapshot to advance worker state.",
                 input_schema={"type": "object",
                               "properties": {"worker_id": {"type": "string"},
                                              "snapshot": {"type": "string"}},
                               "required": ["worker_id", "snapshot"]},
                 required_permission=danger, handler=worker_observe),
        ToolSpec(name="WorkerResolveTrust",
                 description="Resolve trust prompt (trust/deny).",
                 input_schema={"type": "object",
                               "properties": {"worker_id": {"type": "string"},
                                              "decision": {"type": "string",
                                                           "enum": ["trust", "deny"]}},
                               "required": ["worker_id", "decision"]},
                 required_permission=danger, handler=worker_resolve_trust),
        ToolSpec(name="WorkerAwaitReady",
                 description="Wait/poll until worker is ready.",
                 input_schema={"type": "object",
                               "properties": {"worker_id": {"type": "string"}},
                               "required": ["worker_id"]},
                 required_permission=ro, handler=worker_await_ready),
        ToolSpec(name="WorkerSendPrompt",
                 description="Send a task prompt to a ready worker.",
                 input_schema={"type": "object",
                               "properties": {"worker_id": {"type": "string"},
                                              "prompt": {"type": "string"}},
                               "required": ["worker_id", "prompt"]},
                 required_permission=danger, handler=worker_send_prompt),
        ToolSpec(name="WorkerRestart",
                 description="Reset worker boot state.",
                 input_schema={"type": "object",
                               "properties": {"worker_id": {"type": "string"}},
                               "required": ["worker_id"]},
                 required_permission=danger, handler=worker_restart),
        ToolSpec(name="WorkerTerminate",
                 description="Terminate worker and release lane.",
                 input_schema={"type": "object",
                               "properties": {"worker_id": {"type": "string"}},
                               "required": ["worker_id"]},
                 required_permission=danger, handler=worker_terminate),
        ToolSpec(name="WorkerObserveCompletion",
                 description="Report session completion (Finished/Failed).",
                 input_schema={"type": "object",
                               "properties": {"worker_id": {"type": "string"},
                                              "finish_reason": {"type": "string",
                                                                "enum": ["Finished", "Failed"]}},
                               "required": ["worker_id", "finish_reason"]},
                 required_permission=ro, handler=worker_observe_completion),
    ]
    for s in specs:
        registry.register(s)
