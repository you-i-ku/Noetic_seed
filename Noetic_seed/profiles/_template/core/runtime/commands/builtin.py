"""Builtin Slash Commands (claw-code 準拠、主要 15 個)。

claw-code 参照: commands_snapshot.json (1036 行)

ctx (runtime context) の想定キー:
  "version"      : str
  "model"        : str
  "provider"     : str
  "session"      : Session
  "runtime"      : ConversationRuntime (optional)
  "tool_registry": ToolRegistry
  "settings_path": Path
  "workspace_root": Path
  "skill_dirs"   : list[str]

各 handler は (args, ctx) -> CommandResult。
"""
import json
import subprocess
from pathlib import Path
from typing import Callable, Optional

from core.runtime.commands.dispatcher import (
    CommandDispatcher,
    CommandResult,
    CommandSpec,
)


# ============================================================
# /doctor  — setup + preflight diagnostic
# ============================================================

def _cmd_doctor(args, ctx) -> CommandResult:
    import platform
    import sys as _sys
    lines = ["=== claw-code doctor ==="]
    lines.append(f"python: {_sys.version.split()[0]} ({platform.platform()})")
    lines.append(f"provider: {ctx.get('provider', '(unset)')}")
    lines.append(f"model: {ctx.get('model', '(unset)')}")
    ws = ctx.get("workspace_root")
    lines.append(f"workspace_root: {ws}")
    sp = ctx.get("settings_path")
    lines.append(f"settings_path: {sp} "
                 f"({'exists' if sp and Path(sp).exists() else 'missing'})")
    reg = ctx.get("tool_registry")
    if reg is not None:
        lines.append(f"tools registered: {len(reg.all_names())}")
    # optional deps
    for pkg in ("httpx", "lmstudio"):
        try:
            __import__(pkg)
            lines.append(f"  ✓ {pkg}")
        except ImportError:
            lines.append(f"  ✗ {pkg} (not installed)")
    return CommandResult(text="\n".join(lines))


# ============================================================
# /status
# ============================================================

def _cmd_status(args, ctx) -> CommandResult:
    session = ctx.get("session")
    runtime = ctx.get("runtime")
    reg = ctx.get("tool_registry")
    lines = ["=== session status ==="]
    lines.append(f"provider: {ctx.get('provider', '-')}")
    lines.append(f"model: {ctx.get('model', '-')}")
    if session is not None:
        lines.append(f"messages: {len(session.messages)}")
    if runtime is not None:
        lines.append(f"max_iterations: {runtime.max_iterations}")
    if reg is not None:
        lines.append(f"tools: {len(reg.all_names())}")
    return CommandResult(text="\n".join(lines))


# ============================================================
# /version
# ============================================================

def _cmd_version(args, ctx) -> CommandResult:
    return CommandResult(text=f"Noetic Seed v{ctx.get('version', '0.5-dev')}")


# ============================================================
# /exit / /quit
# ============================================================

def _cmd_exit(args, ctx) -> CommandResult:
    return CommandResult(text="Goodbye.", action="exit")


# ============================================================
# /help
# ============================================================

def _make_help(dispatcher: CommandDispatcher) -> Callable:
    def _cmd_help(args, ctx):
        if args:
            name = args[0].lstrip("/")
            spec = dispatcher.get(name)
            if spec is None:
                return CommandResult(text=f"Unknown: /{name}",
                                     is_error=True)
            out = [f"/{spec.name} — {spec.description}"]
            if spec.usage:
                out.append(f"Usage: {spec.usage}")
            return CommandResult(text="\n".join(out))
        lines = ["Available commands:"]
        for name in dispatcher.all_names():
            sp = dispatcher.get(name)
            lines.append(f"  /{name:12s}  {sp.description}")
        return CommandResult(text="\n".join(lines))
    return _cmd_help


# ============================================================
# /config
# ============================================================

def _cmd_config(args, ctx) -> CommandResult:
    sp = ctx.get("settings_path")
    if not sp:
        return CommandResult(text="Error: settings_path not configured",
                             is_error=True)
    path = Path(sp)
    if not args:
        # show
        if not path.exists():
            return CommandResult(text="(settings file not found)")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return CommandResult(text=f"Parse error: {e}", is_error=True)
        return CommandResult(text=json.dumps(data, ensure_ascii=False,
                                             indent=2))
    if len(args) == 1:
        # get single key
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        keys = args[0].split(".")
        node = data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return CommandResult(text=f"Not set: {args[0]}",
                                     is_error=True)
            node = node[k]
        return CommandResult(text=f"{args[0]} = {json.dumps(node, ensure_ascii=False)}")
    if len(args) >= 2:
        # set
        key = args[0]
        value_str = " ".join(args[1:])
        try:
            value = json.loads(value_str)
        except Exception:
            value = value_str
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            data = {}
        keys = key.split(".")
        node = data
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return CommandResult(text=f"Set {key} = {json.dumps(value, ensure_ascii=False)}")
    return CommandResult(text="Usage: /config [key] [value]")


# ============================================================
# /memory — show recent session messages
# ============================================================

def _cmd_memory(args, ctx) -> CommandResult:
    session = ctx.get("session")
    if session is None:
        return CommandResult(text="(no session)")
    n = 10
    if args:
        try:
            n = max(1, min(int(args[0]), 100))
        except ValueError:
            pass
    msgs = session.messages[-n:]
    if not msgs:
        return CommandResult(text="(empty)")
    lines = [f"Last {len(msgs)} messages:"]
    for i, m in enumerate(msgs, start=1):
        role = m.get("role", "?")
        content = m.get("content", [])
        if isinstance(content, list):
            preview_parts = []
            for b in content:
                t = b.get("type", "")
                if t == "text":
                    preview_parts.append(b.get("text", "")[:100])
                elif t == "tool_use":
                    preview_parts.append(f"[tool_use:{b.get('name','')}]")
                elif t == "tool_result":
                    preview_parts.append(f"[tool_result:{str(b.get('content',''))[:60]}]")
            preview = " ".join(preview_parts)
        else:
            preview = str(content)[:120]
        lines.append(f"  {i:3d}. [{role}] {preview}")
    return CommandResult(text="\n".join(lines))


# ============================================================
# /clear — clear session messages
# ============================================================

def _cmd_clear(args, ctx) -> CommandResult:
    session = ctx.get("session")
    if session is None:
        return CommandResult(text="(no session)", is_error=True)
    before = len(session.messages)
    session.clear()
    return CommandResult(text=f"Cleared {before} messages.")


# ============================================================
# /brief — response shortening toggle (flag in ctx)
# ============================================================

def _cmd_brief(args, ctx) -> CommandResult:
    current = ctx.get("brief_mode", False)
    new = not current if not args else args[0].lower() in ("on", "true", "1")
    ctx["brief_mode"] = new
    return CommandResult(text=f"brief_mode = {new}")


# ============================================================
# /plan — enter plan mode
# ============================================================

def _cmd_plan(args, ctx) -> CommandResult:
    from core.runtime.tools import plan as _plan_mod
    if args and args[0].lower() == "exit":
        _plan_mod.exit_plan_mode({})
        return CommandResult(text="Exited plan mode.")
    plan_text = " ".join(args) if args else ""
    _plan_mod.enter_plan_mode({"plan": plan_text})
    return CommandResult(text=f"Entered plan mode: {plan_text[:100] or '(empty)'}")


# ============================================================
# /mcp — MCP server management (list/auth)
# ============================================================

def _cmd_mcp(args, ctx) -> CommandResult:
    settings_path = ctx.get("settings_path")
    sub = args[0] if args else "list"
    if sub == "list":
        if not settings_path or not Path(settings_path).exists():
            return CommandResult(text="(no settings)")
        try:
            data = json.loads(Path(settings_path).read_text(encoding="utf-8"))
        except Exception as e:
            return CommandResult(text=f"Parse error: {e}", is_error=True)
        servers = data.get("mcp_servers", {})
        if not servers:
            return CommandResult(text="(no MCP servers configured)")
        lines = [f"MCP servers ({len(servers)}):"]
        for name, cfg in servers.items():
            stype = cfg.get("type", "?")
            lines.append(f"  {name}  [{stype}]")
        return CommandResult(text="\n".join(lines))
    if sub == "auth" and len(args) >= 2:
        server = args[1]
        from core.runtime.tools import mcp as _mcp
        out = _mcp.mcp_auth({"server": server})
        return CommandResult(text=out)
    return CommandResult(text="Usage: /mcp [list | auth <server>]")


# ============================================================
# Git 関連: /branch /commit /diff /pr
# ============================================================

def _run_git(args: list, cwd: Optional[Path] = None) -> str:
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=str(cwd) if cwd else None,
            timeout=30,
        )
    except FileNotFoundError:
        return "Error: git not in PATH"
    except Exception as e:
        return f"Error: {e}"
    out = result.stdout.rstrip()
    err = result.stderr.rstrip()
    if result.returncode != 0:
        return f"[git {' '.join(args)} rc={result.returncode}]\n{err}\n{out}"
    return out or err or "(no output)"


def _cmd_branch(args, ctx) -> CommandResult:
    cwd = ctx.get("workspace_root")
    return CommandResult(text=_run_git(["status", "-b", "--short"],
                                       cwd=Path(cwd) if cwd else None))


def _cmd_commit(args, ctx) -> CommandResult:
    cwd = ctx.get("workspace_root")
    # /commit -m "msg"  または  /commit "msg"
    if not args:
        return CommandResult(text="Usage: /commit -m <message>")
    git_args = ["commit"]
    if args[0] == "-m" and len(args) > 1:
        git_args.extend(["-m", " ".join(args[1:])])
    else:
        git_args.extend(["-m", " ".join(args)])
    return CommandResult(text=_run_git(git_args, cwd=Path(cwd) if cwd else None))


def _cmd_diff(args, ctx) -> CommandResult:
    cwd = ctx.get("workspace_root")
    return CommandResult(text=_run_git(["diff"] + list(args),
                                       cwd=Path(cwd) if cwd else None))


def _cmd_pr(args, ctx) -> CommandResult:
    # gh CLI に委譲
    cwd = ctx.get("workspace_root")
    try:
        result = subprocess.run(
            ["gh", "pr"] + list(args),
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=str(cwd) if cwd else None,
            timeout=60,
        )
    except FileNotFoundError:
        return CommandResult(text="Error: gh not in PATH", is_error=True)
    except Exception as e:
        return CommandResult(text=f"Error: {e}", is_error=True)
    out = (result.stdout or result.stderr).rstrip()
    return CommandResult(text=out or "(no output)",
                         is_error=result.returncode != 0)


# ============================================================
# /skill — list/show
# ============================================================

def _cmd_skill(args, ctx) -> CommandResult:
    skill_dirs = ctx.get("skill_dirs") or []
    if not args or args[0] == "list":
        found: list = []
        for d in skill_dirs:
            p = Path(d)
            if not p.exists():
                continue
            for md in p.rglob("*.md"):
                rel = md.relative_to(p).as_posix()
                found.append(f"  {rel}  (in {d})")
        if not found:
            return CommandResult(text="(no skills found)")
        return CommandResult(text="Skills:\n" + "\n".join(found))
    # show <name>
    if args[0] == "show" and len(args) > 1:
        from core.runtime.tools import skill as _skill
        loader = _skill._make_skill(skill_dirs)
        return CommandResult(text=loader({"name": args[1]}))
    return CommandResult(text="Usage: /skill [list | show <name>]")


# ============================================================
# /plugin — show registered plugins (placeholder)
# ============================================================

def _cmd_plugin(args, ctx) -> CommandResult:
    return CommandResult(text="Plugin system not yet implemented "
                              "(see CLAWCODE_CAPABILITY_INVENTORY §10.5)")


# ============================================================
# register
# ============================================================

def register_default_commands(dispatcher: CommandDispatcher) -> None:
    specs = [
        CommandSpec(name="doctor",
                    description="setup + preflight diagnostic",
                    handler=_cmd_doctor),
        CommandSpec(name="status",
                    description="session status",
                    handler=_cmd_status),
        CommandSpec(name="version",
                    description="version info",
                    handler=_cmd_version),
        CommandSpec(name="exit",
                    description="session terminate",
                    handler=_cmd_exit),
        CommandSpec(name="quit",
                    description="alias for /exit",
                    handler=_cmd_exit),
        CommandSpec(name="config",
                    description="settings get/set",
                    handler=_cmd_config,
                    usage="/config [key [value]]"),
        CommandSpec(name="memory",
                    description="show last N session messages",
                    handler=_cmd_memory,
                    usage="/memory [n=10]"),
        CommandSpec(name="clear",
                    description="clear session messages",
                    handler=_cmd_clear),
        CommandSpec(name="brief",
                    description="toggle brief response mode",
                    handler=_cmd_brief),
        CommandSpec(name="plan",
                    description="enter/exit plan mode",
                    handler=_cmd_plan,
                    usage="/plan [text | exit]"),
        CommandSpec(name="mcp",
                    description="MCP server management",
                    handler=_cmd_mcp,
                    usage="/mcp [list | auth <server>]"),
        CommandSpec(name="branch",
                    description="git branch status",
                    handler=_cmd_branch),
        CommandSpec(name="commit",
                    description="git commit",
                    handler=_cmd_commit,
                    usage="/commit -m <message>"),
        CommandSpec(name="diff",
                    description="git diff",
                    handler=_cmd_diff),
        CommandSpec(name="pr",
                    description="gh pr operations",
                    handler=_cmd_pr),
        CommandSpec(name="skill",
                    description="list/show skills",
                    handler=_cmd_skill,
                    usage="/skill [list | show <name>]"),
        CommandSpec(name="plugin",
                    description="plugin management (placeholder)",
                    handler=_cmd_plugin),
    ]
    for s in specs:
        dispatcher.register(s)
    # /help は dispatcher 自身を参照するので register 後に差し込む
    dispatcher.register(CommandSpec(
        name="help",
        description="list commands or show command usage",
        handler=_make_help(dispatcher),
        usage="/help [command]",
    ))
