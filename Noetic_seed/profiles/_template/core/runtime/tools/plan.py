"""Plan — EnterPlanMode / ExitPlanMode.

claw-code 参照: rust/crates/runtime/src/plan_mode.rs

plan mode は runtime に override フラグを立てるだけのシンプルな機構。
在 memory (module global) で状態保持。
"""
from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


_plan_mode_active: bool = False
_plan_content: str = ""


def is_plan_mode_active() -> bool:
    return _plan_mode_active


def get_plan_content() -> str:
    return _plan_content


def enter_plan_mode(inp: dict) -> str:
    global _plan_mode_active, _plan_content
    plan = inp.get("plan", "")
    _plan_mode_active = True
    _plan_content = str(plan) if plan else ""
    # 段階10 Step 4 付帯 D: Fix 5 精神で plan content truncation 撤去
    return f"Entered plan mode. Plan: {_plan_content or '(empty)'}"


def exit_plan_mode(inp: dict) -> str:
    global _plan_mode_active, _plan_content
    _plan_mode_active = False
    saved = _plan_content
    _plan_content = ""
    # 段階10 Step 4 付帯 D: Fix 5 精神で plan content truncation 撤去
    return f"Exited plan mode. Previous plan: {saved or '(empty)'}"


def register(registry: ToolRegistry) -> None:
    specs = [
        ToolSpec(
            name="EnterPlanMode",
            description="Enter planning mode. Tools will not be executed until ExitPlanMode.",
            input_schema={
                "type": "object",
                "properties": {"plan": {"type": "string"}},
            },
            required_permission=PermissionMode.WORKSPACE_WRITE,
            handler=enter_plan_mode,
        ),
        ToolSpec(
            name="ExitPlanMode",
            description="Exit planning mode.",
            input_schema={"type": "object", "properties": {}},
            required_permission=PermissionMode.WORKSPACE_WRITE,
            handler=exit_plan_mode,
        ),
    ]
    for s in specs:
        registry.register(s)
