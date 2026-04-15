"""Skill — Skill / Agent / ToolSearch.

claw-code 参照: rust/crates/runtime/src/skill_registry.rs, agent_dispatch.rs,
                 tool_search.rs

Skill: YAML frontmatter + Markdown の skill 定義ファイル読込
Agent: 専門 agent に task を投げる (Phase 2 では stub 的に応答)
ToolSearch: 登録済 tool から description 類似で検索
"""
import re
from pathlib import Path
from typing import Callable, Optional

from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


# ============================================================
# Skill
# ============================================================

def _parse_skill_file(path: Path) -> dict:
    """YAML frontmatter + Markdown body をパース。

    フォーマット:
      ---
      name: skill-name
      description: ...
      ---
      body...
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}

    frontmatter = {}
    body = text
    if text.startswith("---"):
        end = text.find("---", 3)
        if end > 0:
            fm_text = text[3:end].strip()
            body = text[end + 3:].lstrip("\n")
            for line in fm_text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    frontmatter[k.strip()] = v.strip().strip('"').strip("'")
    return {"frontmatter": frontmatter, "body": body}


def _make_skill(skill_dirs: list) -> Callable:
    def skill(inp: dict) -> str:
        name = (inp.get("name") or "").strip()
        if not name:
            return "Error: name is required"

        for d in skill_dirs:
            if not d or not Path(d).exists():
                continue
            for fname in (f"{name}.md", f"{name}/SKILL.md",
                          f"{name}/skill.md"):
                p = Path(d) / fname
                if p.exists():
                    parsed = _parse_skill_file(p)
                    fm = parsed.get("frontmatter", {})
                    body = parsed.get("body", "")
                    return (
                        f"[Skill: {fm.get('name', name)}]\n"
                        f"{fm.get('description', '')}\n\n"
                        f"{body}"
                    )
        return f"Error: skill '{name}' not found"

    return skill


# ============================================================
# Agent
# ============================================================

_agent_bridge: dict = {"dispatch": None}


def set_agent_dispatcher(dispatch_fn: Callable) -> None:
    """agent 実行を外部に委譲する bridge を登録。"""
    _agent_bridge["dispatch"] = dispatch_fn


def agent(inp: dict) -> str:
    agent_type = (inp.get("agent_type") or "").strip()
    task = (inp.get("task") or "").strip()
    if not agent_type:
        return "Error: agent_type is required"
    if not task:
        return "Error: task is required"

    fn = _agent_bridge.get("dispatch")
    if fn is None:
        return (f"[Agent pending — dispatcher not configured]\n"
                f"agent_type: {agent_type}\n"
                f"task: {task[:500]}")
    try:
        return fn(agent_type, task, inp)
    except Exception as e:
        return f"Error: agent dispatch failed: {e}"


# ============================================================
# ToolSearch
# ============================================================

def _make_tool_search(registry: ToolRegistry) -> Callable:
    def tool_search(inp: dict) -> str:
        query = (inp.get("query") or "").strip().lower()
        if not query:
            return "Error: query is required"

        query_tokens = set(re.findall(r"\w+", query))
        scored: list = []
        for spec in registry.list():
            text = f"{spec.name} {spec.description}".lower()
            tokens = set(re.findall(r"\w+", text))
            overlap = len(query_tokens & tokens)
            if overlap > 0:
                scored.append((overlap, spec.name, spec.description))

        if not scored:
            return f"No tools match: {query}"

        scored.sort(key=lambda x: -x[0])
        lines = [f"Tools matching '{query}':"]
        for _, name, desc in scored[:10]:
            lines.append(f"  - {name}: {desc[:100]}")
        return "\n".join(lines)

    return tool_search


# ============================================================
# register
# ============================================================

def register(registry: ToolRegistry, skill_dirs: list) -> None:
    """Skill の検索対象ディレクトリ群を渡す。"""
    specs = [
        ToolSpec(
            name="Skill",
            description=("Load and return a skill definition "
                         "(YAML frontmatter + Markdown body)."),
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=_make_skill(skill_dirs),
        ),
        ToolSpec(
            name="Agent",
            description="Dispatch a task to a specialized agent.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_type": {"type": "string"},
                    "task": {"type": "string"},
                },
                "required": ["agent_type", "task"],
            },
            required_permission=PermissionMode.DANGER_FULL_ACCESS,
            handler=agent,
        ),
        ToolSpec(
            name="ToolSearch",
            description="Search registered tools by name/description.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=_make_tool_search(registry),
        ),
    ]
    for s in specs:
        registry.register(s)
