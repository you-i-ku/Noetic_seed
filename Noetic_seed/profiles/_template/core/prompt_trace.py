"""M0: 呼出前の入力と応答の対応を残す。記録失敗は呼出元に伝えない。"""
import copy
import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime


TRACE_FILE_NAME = "prompt_trace.jsonl"
_active = ContextVar("prompt_trace", default=None)


def _safe(action):
    try:
        return action()
    except Exception as e:
        try:
            print(f"  [prompt_trace] observation skip: {e}")
        except Exception:
            pass


def emit(record):
    """UTF-8 JSONL を追記。未完の末尾は改行で隔離し、次の行を守る。"""
    from core.config import MEMORY_DIR
    data = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    with (MEMORY_DIR / TRACE_FILE_NAME).open("a+b") as f:
        if f.tell():
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write(data)


@contextmanager
def trace_fire(event):
    """0b の ID を共有し、正常終了・途中終了とも fire のコンテキストを解除。"""
    context = _safe(lambda: {
        "run_id": event["run_id"], "attempt_id": event["attempt_id"],
        "cycle_id": event["cycle_id"], "chain_position": None, "call_index": 0,
    })
    token = _active.set(context)
    try:
        yield
    finally:
        _active.reset(token)


def set_chain_position(position, cycle_id):
    context = _active.get()
    if context is not None:
        context.update(chain_position=position, cycle_id=cycle_id)


def _begin(role, cycle_id=None):
    context = _active.get()
    if context is None:
        return None
    index = context["call_index"]
    context["call_index"] += 1  # コピーや書出しに失敗しても再利用しない。
    return {
        "trace_id": f"{context['attempt_id']}:{index}",
        "run_id": context["run_id"], "attempt_id": context["attempt_id"],
        "cycle_id": context["cycle_id"] if cycle_id is None else cycle_id,
        "role": role, "chain_position": context["chain_position"] if role == "llm2" else None,
        "call_index": index,
    }


def _record(ids, kind, payload):
    if ids is not None:
        _safe(lambda: emit({**ids, "kind": kind, "time": datetime.now().isoformat(),
                            **copy.deepcopy(payload())}))


def _invoke(ids, request_payload, invoke, response_payload):
    _record(ids, "request", request_payload)
    try:
        response = invoke()
    except BaseException as e:
        _record(ids, "error", lambda: {"exception_type": type(e).__name__, "exception_message": str(e)})
        raise
    _record(ids, "response", lambda: response_payload(response))
    return response


def trace_llm1(call, prompt, *, cycle_id, max_tokens, temperature, image_paths):
    """call_llm を一度だけ呼ぶ。応答は debug log と同じ文字列の長さ・SHA-256。"""
    ids = _safe(lambda: _begin("llm1", cycle_id))

    def provider_metadata():
        from core.llm import _get_active_provider_config
        provider, _, _, model = _get_active_provider_config()
        return {"provider": provider, "model": model}

    def payload():
        metadata = _safe(provider_metadata)
        if metadata is None:
            metadata = {"provider": None, "model": None, "provider_config_unavailable": True}
        return {**metadata, "observation_scope": "call_llm_input",
                "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                "image_paths": image_paths}

    return _invoke(ids, payload,
                   lambda: call(prompt, max_tokens=max_tokens, temperature=temperature,
                                image_paths=image_paths),
                   lambda response: {"response_length": len(response),
                                     "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest()})


class TracingProvider:
    """ApiRequest 境界だけを観察。同一 request / response をそのまま受け渡す。"""

    def __init__(self, provider):
        self._provider = provider

    @property
    def name(self):
        return self._provider.name

    @property
    def model(self):
        return self._provider.model

    def supports_tool_use(self):
        return self._provider.supports_tool_use()

    def supports_vision(self):
        return self._provider.supports_vision()

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def stream(self, request):
        ids = _safe(lambda: _begin("llm2"))

        def payload():
            return {"provider": self.name, "model": self.model,
                    "observation_scope": "provider_api_request_before_conversion",
                    "sdk_internal_inputs": "unobserved" if self.name == "claude_code" else "not_applicable",
                    **{key: getattr(request, key) for key in (
                        "system_prompt", "messages", "tools", "tool_choice",
                        "max_tokens", "temperature", "image_paths")},
                    "messages": request.observation_messages if request.observation_messages is not None else request.messages}

        def response_payload(response):
            return {"tool_uses": [{"id": t.id, "name": t.name} for t in response.tool_uses],
                    "tool_invocations": [{"tool_id": t["tool_id"], "tool_name": t["tool_name"]}
                                         for t in response.tool_invocations]}

        return _invoke(ids, payload, lambda: self._provider.stream(request), response_payload)
