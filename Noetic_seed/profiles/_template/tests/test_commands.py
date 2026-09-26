"""Slash Commands の動作確認テスト。"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.commands.dispatcher import CommandDispatcher
from core.runtime.commands.builtin import register_default_commands
from core.runtime.registry import ToolRegistry
from core.runtime.session import Session


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


TMPDIR: Path = None


def _setup():
    global TMPDIR
    TMPDIR = Path(tempfile.mkdtemp(prefix="clawcode_cmd_"))
    (TMPDIR / "settings.json").write_text(
        json.dumps({"provider": "anthropic", "model": "claude-sonnet-4-6"}),
        encoding="utf-8",
    )
    sk = TMPDIR / "skills"
    sk.mkdir()
    (sk / "hello.md").write_text(
        "---\nname: hello\ndescription: greet\n---\nHello body.",
        encoding="utf-8",
    )


def _teardown():
    if TMPDIR and TMPDIR.exists():
        shutil.rmtree(TMPDIR)


def _ctx():
    sess = Session()
    sess.push_user_text("first message")
    reg = ToolRegistry()
    return {
        "version": "0.5-test",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "session": sess,
        "tool_registry": reg,
        "settings_path": TMPDIR / "settings.json",
        "workspace_root": TMPDIR,
        "skill_dirs": [str(TMPDIR / "skills")],
    }


def _dispatcher():
    d = CommandDispatcher()
    register_default_commands(d)
    return d


# ============================================================
# dispatcher basic
# ============================================================

def test_dispatch_unknown():
    print("== dispatcher: unknown command ==")
    d = _dispatcher()
    r = d.dispatch("/nonexistent", _ctx())
    return all([
        _assert(r.is_error, "is_error"),
        _assert("Unknown command" in r.text, "メッセージ"),
    ])


def test_dispatch_not_a_command():
    print("== dispatcher: not a command ==")
    d = _dispatcher()
    r = d.dispatch("plain text", _ctx())
    return _assert(r.is_error and "Not a command" in r.text, "拒否")


def test_dispatch_empty():
    print("== dispatcher: empty line ==")
    d = _dispatcher()
    r = d.dispatch("", _ctx())
    return _assert(r.is_error, "空文字拒否")


# ============================================================
# basic commands
# ============================================================

def test_help():
    print("== /help ==")
    d = _dispatcher()
    r = d.dispatch("/help", _ctx())
    return all([
        _assert(not r.is_error, "not error"),
        _assert("/status" in r.text, "/status 含む"),
        _assert("/help" in r.text, "/help 自身"),
        _assert("/commit" in r.text, "/commit 含む"),
    ])


def test_help_specific():
    print("== /help status ==")
    d = _dispatcher()
    r = d.dispatch("/help status", _ctx())
    return all([
        _assert("session status" in r.text, "description"),
    ])


def test_version():
    print("== /version ==")
    d = _dispatcher()
    r = d.dispatch("/version", _ctx())
    return _assert("0.5-test" in r.text, "version 表示")


def test_status():
    print("== /status ==")
    d = _dispatcher()
    r = d.dispatch("/status", _ctx())
    return all([
        _assert("provider" in r.text, "provider"),
        _assert("messages" in r.text, "messages count"),
    ])


def test_doctor():
    print("== /doctor ==")
    d = _dispatcher()
    r = d.dispatch("/doctor", _ctx())
    return all([
        _assert("python:" in r.text, "python"),
        _assert("workspace_root" in r.text, "workspace"),
    ])


def test_exit():
    print("== /exit ==")
    d = _dispatcher()
    r = d.dispatch("/exit", _ctx())
    return _assert(r.action == "exit", "action=exit")


def test_quit_alias():
    print("== /quit (alias of /exit) ==")
    d = _dispatcher()
    r = d.dispatch("/quit", _ctx())
    return _assert(r.action == "exit", "action=exit")


# ============================================================
# /config
# ============================================================

def test_config_show():
    print("== /config show all ==")
    d = _dispatcher()
    r = d.dispatch("/config", _ctx())
    return _assert("anthropic" in r.text, "provider 含む")


def test_config_get():
    print("== /config <key> ==")
    d = _dispatcher()
    r = d.dispatch("/config provider", _ctx())
    return _assert('"anthropic"' in r.text, "value 取得")


def test_config_set():
    print("== /config <key> <value> ==")
    d = _dispatcher()
    ctx = _ctx()
    r = d.dispatch('/config temperature 0.5', ctx)
    # verify
    r2 = d.dispatch("/config temperature", ctx)
    return all([
        _assert("Set" in r.text, "set メッセージ"),
        _assert("0.5" in r2.text, "get で取得"),
    ])


def test_config_dotted():
    print("== /config dot.path set ==")
    d = _dispatcher()
    ctx = _ctx()
    d.dispatch("/config nested.deep.key 123", ctx)
    r = d.dispatch("/config nested.deep.key", ctx)
    return _assert("123" in r.text, "dot 記法保存")


# ============================================================
# /memory /clear
# ============================================================

def test_memory():
    print("== /memory ==")
    d = _dispatcher()
    ctx = _ctx()
    r = d.dispatch("/memory", ctx)
    return all([
        _assert("[user]" in r.text, "user role 表示"),
        _assert("first message" in r.text, "content 表示"),
    ])


def test_clear():
    print("== /clear ==")
    d = _dispatcher()
    ctx = _ctx()
    r = d.dispatch("/clear", ctx)
    return all([
        _assert("Cleared" in r.text, "message"),
        _assert(len(ctx["session"].messages) == 0, "実際に空"),
    ])


# ============================================================
# /brief
# ============================================================

def test_brief_toggle():
    print("== /brief toggle ==")
    d = _dispatcher()
    ctx = _ctx()
    d.dispatch("/brief", ctx)
    r = d.dispatch("/brief", ctx)
    return _assert("brief_mode" in r.text, "brief_mode 表示")


# ============================================================
# /plan
# ============================================================

def test_plan_enter_exit():
    print("== /plan enter/exit ==")
    d = _dispatcher()
    r1 = d.dispatch('/plan "step 1"', _ctx())
    r2 = d.dispatch("/plan exit", _ctx())
    return all([
        _assert("Entered" in r1.text, "enter"),
        _assert("Exited" in r2.text, "exit"),
    ])


# ============================================================
# /mcp
# ============================================================

def test_mcp_list_empty():
    print("== /mcp list (empty) ==")
    d = _dispatcher()
    r = d.dispatch("/mcp list", _ctx())
    return _assert("no MCP" in r.text, "空リスト")


def test_mcp_list_with_servers():
    print("== /mcp list (with servers) ==")
    d = _dispatcher()
    ctx = _ctx()
    # write mcp_servers into settings
    data = {"mcp_servers": {
        "slack": {"type": "stdio", "command": "mcp-slack"},
        "github": {"type": "http", "url": "https://..."},
    }}
    (TMPDIR / "settings.json").write_text(json.dumps(data),
                                          encoding="utf-8")
    r = d.dispatch("/mcp list", ctx)
    return all([
        _assert("slack" in r.text, "slack"),
        _assert("github" in r.text, "github"),
    ])


# ============================================================
# /skill
# ============================================================

def test_skill_list():
    print("== /skill list ==")
    d = _dispatcher()
    r = d.dispatch("/skill list", _ctx())
    return _assert("hello.md" in r.text, "skill 発見")


def test_skill_show():
    print("== /skill show hello ==")
    d = _dispatcher()
    r = d.dispatch("/skill show hello", _ctx())
    return _assert("Hello body" in r.text, "本文表示")


# ============================================================
# /plugin (placeholder)
# ============================================================

def test_plugin_placeholder():
    print("== /plugin placeholder ==")
    d = _dispatcher()
    r = d.dispatch("/plugin", _ctx())
    return _assert("not yet" in r.text, "placeholder")


# ============================================================
# Git 系 (git コマンドあれば検証)
# ============================================================

def test_branch():
    """No git process: verify the slash adapter preserves failure facts."""
    from unittest.mock import patch
    from types import SimpleNamespace
    with patch("core.runtime.commands.builtin.subprocess.run", return_value=SimpleNamespace(
        returncode=128, stdout="", stderr="not a repository")) as run:
        result = _dispatcher().dispatch("/branch", _ctx())
    assert result.is_error
    assert result.text == "[git status -b --short rc=128]\nnot a repository\n"
    assert result.detail["returncode"] == 128
    assert run.call_args.args[0] == ["git", "status", "-b", "--short"]
    return True


def test_commit_usage():
    print("== /commit usage ==")
    d = _dispatcher()
    r = d.dispatch("/commit", _ctx())
    return _assert("Usage" in r.text, "usage 表示")


# ============================================================
# handler 例外のキャッチ
# ============================================================

def test_handler_exception():
    print("== handler 例外 → CommandResult.is_error ==")
    from core.runtime.commands.dispatcher import (
        CommandDispatcher, CommandSpec,
    )
    d = CommandDispatcher()
    def broken(args, ctx):
        raise RuntimeError("boom")
    d.register(CommandSpec(name="break", description="", handler=broken))
    r = d.dispatch("/break", {})
    return all([
        _assert(r.is_error, "is_error"),
        _assert("boom" in r.text, "例外文字列"),
    ])


# ============================================================
# 引数 parsing (shlex)
# ============================================================

def test_shlex_quoted_args():
    print("== dispatcher: shlex クォート対応 ==")
    from core.runtime.commands.dispatcher import (
        CommandDispatcher, CommandSpec, CommandResult,
    )
    d = CommandDispatcher()
    captured = {}
    def h(args, ctx):
        captured["args"] = args
        return CommandResult(text="ok")
    d.register(CommandSpec(name="x", description="", handler=h))
    d.dispatch('/x "hello world" foo', {})
    return _assert(captured["args"] == ["hello world", "foo"],
                   "クォート対応")


def main():
    _setup()
    try:
        tests = [
            test_dispatch_unknown, test_dispatch_not_a_command,
            test_dispatch_empty,
            test_help, test_help_specific, test_version,
            test_status, test_doctor, test_exit, test_quit_alias,
            test_config_show, test_config_get, test_config_set,
            test_config_dotted,
            test_memory, test_clear, test_brief_toggle,
            test_plan_enter_exit,
            test_mcp_list_empty, test_mcp_list_with_servers,
            test_skill_list, test_skill_show, test_plugin_placeholder,
            test_branch, test_commit_usage,
            test_handler_exception, test_shlex_quoted_args,
        ]
        print(f"Running {len(tests)} test groups...\n")
        passed = 0
        for t in tests:
            if t():
                passed += 1
            print()
        print(f"=== Result: {passed}/{len(tests)} test groups passed ===")
        return 0 if passed == len(tests) else 1
    finally:
        _teardown()


if __name__ == "__main__":
    sys.exit(main())


import pytest
from unittest.mock import Mock
from types import SimpleNamespace
from core.runtime.registry import ToolError, ToolResult
from core.runtime.commands import builtin as commands


@pytest.fixture(autouse=True)
def command_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "TMPDIR", tmp_path)
    (tmp_path / "settings.json").write_text(json.dumps({"provider": "anthropic", "model": "claude-sonnet-4-6"}), encoding="utf-8")
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "hello.md").write_text("Hello body.", encoding="utf-8")


@pytest.mark.parametrize("line", ["/branch", "/commit -m test", "/diff", "/pr view"])
@pytest.mark.parametrize("rc", [0, 1])
def test_command_process_exit_status(monkeypatch, line, rc):
    run = Mock(return_value=SimpleNamespace(returncode=rc, stdout="Error is normal content", stderr="stderr"))
    monkeypatch.setattr(commands.subprocess, "run", run)
    result = _dispatcher().dispatch(line, _ctx())
    assert result.is_error is (rc != 0)
    assert "Error is normal content" in result.text
    if rc:
        assert result.detail["returncode"] == rc
        assert result.detail["command"] == run.call_args.args[0]
    else:
        assert result.detail is None
    assert run.call_count == 1


@pytest.mark.parametrize("line", ["/mcp auth missing", "/skill show missing"])
def test_command_direct_tool_failure_has_original_text(monkeypatch, line):
    from core.runtime.tools import mcp
    monkeypatch.setitem(mcp._bridge, "auth", None)
    result = _dispatcher().dispatch(line, _ctx())
    assert result.is_error
    assert "Command /" not in result.text
    assert "pending" in result.text if line.startswith("/mcp") else result.text == "Error: skill 'missing' not found"


def test_command_direct_tool_detail_preserved(monkeypatch):
    from core.runtime.tools import mcp
    monkeypatch.setattr(mcp, "mcp_auth", Mock(side_effect=ToolError("exact", {"status": 401})))
    result = commands._cmd_mcp(["auth", "s"], {})
    assert result.text == "exact" and result.is_error
    assert result.detail == {"status": 401}
    monkeypatch.setattr(mcp, "mcp_auth", lambda args: ToolResult("partial", {"count": 2}))
    result = commands._cmd_mcp(["auth", "s"], {})
    assert result.text == "partial" and not result.is_error
    assert result.detail == {"count": 2}


@pytest.mark.parametrize("args", [["key"], ["key", "new"]])
def test_command_corrupt_settings_not_overwritten(tmp_path, args):
    path = tmp_path / "settings.json"
    path.write_text("{bad", encoding="utf-8")
    result = commands._cmd_config(args, {"settings_path": path})
    assert result.is_error and result.text.startswith("Parse error:")
    assert path.read_text(encoding="utf-8") == "{bad"


def test_dispatcher_catches_toolerror_without_prefix():
    from core.runtime.commands.dispatcher import CommandSpec
    dispatcher = CommandDispatcher()
    dispatcher.register(CommandSpec("probe", "", Mock(side_effect=ToolError("exact", {"fact": 1}))))
    result = dispatcher.dispatch("/probe")
    assert result.is_error and result.text == "exact" and result.detail == {"fact": 1}
