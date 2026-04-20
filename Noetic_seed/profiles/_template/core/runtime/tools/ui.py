"""UI — AskUserQuestion / SendUserMessage / StructuredOutput / Config.

claw-code 参照: rust/crates/runtime/src/user_interaction.rs, config_tools.rs

AskUserQuestion / SendUserMessage は runtime 外の UI レイヤー (ws_server 等)
への橋渡しだけ。callback がなければ「pending」を返す。
Config は settings.json の get/set。
"""
import json
from pathlib import Path
from typing import Callable, Optional

from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


# UI レイヤーへの注入点 (runtime 外から set_ui_bridge で差替)
_ui_bridge: dict = {
    "ask_user": None,    # callable(question, options) -> str (answer)
    "send_user": None,   # callable(message, attachments, status) -> None
}


def set_ui_bridge(ask_user: Optional[Callable] = None,
                  send_user: Optional[Callable] = None) -> None:
    if ask_user is not None:
        _ui_bridge["ask_user"] = ask_user
    if send_user is not None:
        _ui_bridge["send_user"] = send_user


# ============================================================
# AskUserQuestion
# ============================================================

def ask_user_question(inp: dict) -> str:
    question = (inp.get("question") or "").strip()
    options = inp.get("options") or []
    if not question:
        return "Error: question is required"

    fn = _ui_bridge.get("ask_user")
    if fn is None:
        # UI レイヤー未接続時は質問を保留として返す
        return ("[AskUserQuestion pending — UI bridge not configured]\n"
                f"Question: {question}\n"
                f"Options: {options if options else '(free-form)'}")
    try:
        answer = fn(question, options)
    except Exception as e:
        return f"Error: ask_user callback failed: {e}"
    return f"User answered: {answer}"


# ============================================================
# SendUserMessage
# ============================================================

def send_user_message(inp: dict) -> str:
    message = inp.get("message") or ""
    attachments = inp.get("attachments") or []
    status = (inp.get("status") or "normal").lower()
    if status not in ("normal", "proactive"):
        return f"Error: invalid status '{status}'"
    if not message:
        return "Error: message is required"

    fn = _ui_bridge.get("send_user")
    if fn is None:
        return (f"[SendUserMessage pending — UI bridge not configured]\n"
                f"Status: {status}\n"
                f"Message: {message[:500]}")
    try:
        fn(message, attachments, status)
    except Exception as e:
        return f"Error: send_user callback failed: {e}"
    # 段階10 Step 4 付帯 D: Fix 5 精神で sent message truncation 撤去
    return f"Sent ({status}): {message}"


# ============================================================
# StructuredOutput
# ============================================================

def structured_output(inp: dict) -> str:
    """入力をそのまま JSON 整形して返すだけ (agent が構造化結果を返すための通路)。"""
    try:
        return json.dumps(inp, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: serialize failed: {e}"


# ============================================================
# Config
# ============================================================

def _make_config(settings_path: Path) -> Callable:
    def config(inp: dict) -> str:
        setting = (inp.get("setting") or "").strip()
        value = inp.get("value")  # None = get, else = set

        if not setting:
            return "Error: setting is required"

        if not settings_path.exists():
            try:
                settings_path.write_text("{}", encoding="utf-8")
            except Exception as e:
                return f"Error: cannot create settings file: {e}"

        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception as e:
            return f"Error: invalid settings JSON: {e}"

        # dot 記法対応 (e.g. "provider.model")
        keys = setting.split(".")

        if value is None:
            # get
            node = data
            for k in keys:
                if not isinstance(node, dict) or k not in node:
                    return f"Error: setting '{setting}' not found"
                node = node[k]
            return json.dumps({setting: node}, ensure_ascii=False)

        # set
        node = data
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
        try:
            settings_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            return f"Error: write failed: {e}"
        return f"Set {setting} = {json.dumps(value, ensure_ascii=False)}"

    return config


# ============================================================
# register
# ============================================================

def register(registry: ToolRegistry, settings_path: Path) -> None:
    specs = [
        ToolSpec(
            name="AskUserQuestion",
            description="Ask the user a question and wait for their answer.",
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array",
                                "items": {"type": "string"}},
                },
                "required": ["question"],
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=ask_user_question,
        ),
        ToolSpec(
            name="SendUserMessage",
            description="Send a message to the user (proactive or normal).",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "attachments": {"type": "array"},
                    "status": {"type": "string",
                               "enum": ["normal", "proactive"]},
                },
                "required": ["message"],
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=send_user_message,
        ),
        ToolSpec(
            name="StructuredOutput",
            description="Return a structured JSON output.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=structured_output,
        ),
        ToolSpec(
            name="Config",
            description="Get or set a settings value (dot-separated path supported).",
            input_schema={
                "type": "object",
                "properties": {
                    "setting": {"type": "string"},
                    "value": {},
                },
                "required": ["setting"],
            },
            required_permission=PermissionMode.WORKSPACE_WRITE,
            handler=_make_config(settings_path),
        ),
    ]
    for s in specs:
        registry.register(s)
