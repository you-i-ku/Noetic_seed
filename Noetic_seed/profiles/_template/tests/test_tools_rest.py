"""残りの claw-code tool 群の統合テスト。

対象: plan / util / ui / skill / task / worker / team_cron / lsp / mcp
+ register_all で 50 tool 全登録の確認。

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_tools_rest.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.registry import ToolRegistry
from core.runtime.tools import (
    plan, util, ui, skill, task, worker, team_cron, lsp, mcp,
)
from core.runtime.tools import register_all


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


TMPDIR: Path = None


def _setup():
    global TMPDIR
    TMPDIR = Path(tempfile.mkdtemp(prefix="clawcode_rest_"))
    (TMPDIR / "settings.json").write_text("{}", encoding="utf-8")
    sk = TMPDIR / "skills"
    sk.mkdir()
    (sk / "hello.md").write_text(
        "---\nname: hello\ndescription: greet\n---\nBody of hello skill.",
        encoding="utf-8",
    )
    (TMPDIR / "nb.ipynb").write_text(json.dumps({
        "cells": [
            {"cell_type": "code", "metadata": {}, "source": ["x=1\n"],
             "outputs": [], "execution_count": None},
            {"cell_type": "markdown", "metadata": {}, "source": ["# title"]},
        ],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
    }), encoding="utf-8")


def _teardown():
    if TMPDIR and TMPDIR.exists():
        shutil.rmtree(TMPDIR)


# ============================================================
# plan
# ============================================================

def test_plan():
    print("== plan: enter/exit ==")
    reg = ToolRegistry()
    plan.register(reg)
    r1 = reg.execute("EnterPlanMode", {"plan": "step 1"})
    active = plan.is_plan_mode_active()
    r2 = reg.execute("ExitPlanMode", {})
    return all([
        _assert("Entered" in r1, "enter"),
        _assert(active, "active=True"),
        _assert("Exited" in r2, "exit"),
        _assert(not plan.is_plan_mode_active(), "active=False"),
    ])


# ============================================================
# util
# ============================================================

def test_util_sleep():
    print("== util: Sleep ==")
    reg = ToolRegistry()
    util.register(reg, TMPDIR)
    import time
    t0 = time.time()
    out = reg.execute("Sleep", {"duration_ms": 100})
    elapsed = time.time() - t0
    return all([
        _assert("Slept" in out, "message"),
        _assert(elapsed >= 0.09, f"elapsed {elapsed:.3f}s"),
    ])


def test_util_sleep_too_large():
    print("== util: Sleep too large ==")
    reg = ToolRegistry()
    util.register(reg, TMPDIR)
    return _assert("too large" in reg.execute("Sleep", {"duration_ms": 999999}),
                   "拒否")


def test_util_todowrite():
    print("== util: TodoWrite ==")
    reg = ToolRegistry()
    util.register(reg, TMPDIR)
    out = reg.execute("TodoWrite",
                      {"todos": [{"task": "a"}, {"task": "b"}]})
    return all([
        _assert("2 items" in out, "count"),
        _assert(len(util.get_todos()) == 2, "保持"),
    ])


def test_util_notebook_edit():
    print("== util: NotebookEdit replace ==")
    reg = ToolRegistry()
    util.register(reg, TMPDIR)
    out = reg.execute("NotebookEdit", {
        "path": "nb.ipynb", "cell_index": 0,
        "new_source": "x=42\n", "action": "replace",
    })
    nb = json.loads((TMPDIR / "nb.ipynb").read_text(encoding="utf-8"))
    return all([
        _assert("complete" in out, "成功"),
        _assert("x=42" in "".join(nb["cells"][0]["source"]),
                "source 更新"),
    ])


def test_util_notebook_insert_delete():
    print("== util: NotebookEdit insert + delete ==")
    reg = ToolRegistry()
    util.register(reg, TMPDIR)
    reg.execute("NotebookEdit", {
        "path": "nb.ipynb", "cell_index": 1,
        "new_source": "y=2\n", "action": "insert",
        "cell_type": "code",
    })
    nb = json.loads((TMPDIR / "nb.ipynb").read_text(encoding="utf-8"))
    count_after_insert = len(nb["cells"])
    reg.execute("NotebookEdit", {
        "path": "nb.ipynb", "cell_index": 0, "action": "delete",
    })
    nb2 = json.loads((TMPDIR / "nb.ipynb").read_text(encoding="utf-8"))
    count_after_delete = len(nb2["cells"])
    return all([
        _assert(count_after_insert == 3, "insert で 3 cells"),
        _assert(count_after_delete == 2, "delete で 2 cells"),
    ])


def test_util_testing_permission():
    print("== util: TestingPermission echo ==")
    reg = ToolRegistry()
    util.register(reg, TMPDIR)
    out = reg.execute("TestingPermission", {"x": 1})
    return _assert('"x": 1' in out, "echo")


def test_util_run_task_packet():
    print("== util: RunTaskPacket ==")
    reg = ToolRegistry()
    util.register(reg, TMPDIR)
    out = reg.execute("RunTaskPacket", {"packet": {
        "objective": "test", "scope": "unit",
        "acceptance_tests": ["t1", "t2"],
    }})
    return all([
        _assert("objective: test" in out, "objective"),
        _assert("2 items" in out, "test count"),
    ])


# ============================================================
# ui
# ============================================================

def test_ui_config_get_set():
    print("== ui: Config get/set ==")
    reg = ToolRegistry()
    ui.register(reg, TMPDIR / "settings.json")
    r1 = reg.execute("Config", {"setting": "foo.bar", "value": 42})
    r2 = reg.execute("Config", {"setting": "foo.bar"})
    return all([
        _assert("Set" in r1, "set"),
        _assert('"foo.bar": 42' in r2, "get"),
    ])


def test_ui_ask_user_no_bridge():
    print("== ui: AskUserQuestion (bridge 未設定) ==")
    reg = ToolRegistry()
    ui.register(reg, TMPDIR / "settings.json")
    out = reg.execute("AskUserQuestion", {"question": "どう?"})
    return _assert("pending" in out, "pending 返却")


def test_ui_ask_user_with_bridge():
    print("== ui: AskUserQuestion (bridge 設定) ==")
    reg = ToolRegistry()
    ui.register(reg, TMPDIR / "settings.json")
    ui.set_ui_bridge(ask_user=lambda q, opts: f"ok:{q}")
    out = reg.execute("AskUserQuestion", {"question": "A?"})
    ui.set_ui_bridge(ask_user=lambda q, opts: None)  # reset
    return _assert("ok:A?" in out, "callback 応答")


def test_ui_structured_output():
    print("== ui: StructuredOutput ==")
    reg = ToolRegistry()
    ui.register(reg, TMPDIR / "settings.json")
    out = reg.execute("StructuredOutput", {"result": [1, 2, 3]})
    import json as _json
    try:
        parsed = _json.loads(out)
        ok = parsed.get("result") == [1, 2, 3]
    except Exception:
        ok = False
    return _assert(ok, "JSON 出力 (parse 可能)")


# ============================================================
# skill
# ============================================================

def test_skill_load():
    print("== skill: Skill load ==")
    reg = ToolRegistry()
    skill.register(reg, [str(TMPDIR / "skills")])
    out = reg.execute("Skill", {"name": "hello"})
    return all([
        _assert("hello" in out, "name"),
        _assert("Body of hello" in out, "body"),
    ])


def test_skill_not_found():
    print("== skill: not found ==")
    reg = ToolRegistry()
    skill.register(reg, [str(TMPDIR / "skills")])
    out = reg.execute("Skill", {"name": "nonexistent"})
    return _assert("not found" in out, "エラー")


def test_skill_tool_search():
    print("== skill: ToolSearch ==")
    reg = ToolRegistry()
    # Register some tools to search
    from core.runtime.tools import file_ops, shell as _sh
    file_ops.register(reg, TMPDIR)
    _sh.register(reg)
    skill.register(reg, [str(TMPDIR / "skills")])
    out = reg.execute("ToolSearch", {"query": "file"})
    return all([
        _assert("read_file" in out or "write_file" in out, "file 系ヒット"),
    ])


def test_skill_agent_no_dispatcher():
    print("== skill: Agent (dispatcher 未設定) ==")
    reg = ToolRegistry()
    skill.register(reg, [str(TMPDIR / "skills")])
    out = reg.execute("Agent", {"agent_type": "dev", "task": "x"})
    return _assert("pending" in out, "pending")


# ============================================================
# task
# ============================================================

def test_task_lifecycle():
    print("== task: lifecycle (create/get/list/update/output/stop) ==")
    reg = ToolRegistry()
    task.register(reg)
    r_c = reg.execute("TaskCreate", {"description": "my task"})
    # extract id
    import re
    m = re.search(r"id=(task_\w+)", r_c)
    tid = m.group(1) if m else ""
    r_g = reg.execute("TaskGet", {"task_id": tid})
    r_l = reg.execute("TaskList", {})
    r_u = reg.execute("TaskUpdate", {"task_id": tid, "message": "go"})
    task.get_task_registry().append_output(tid, "line1")
    r_o = reg.execute("TaskOutput", {"task_id": tid})
    r_s = reg.execute("TaskStop", {"task_id": tid})
    return all([
        _assert(tid.startswith("task_"), "id 取得"),
        _assert("my task" in r_g, "get"),
        _assert(tid in r_l, "list に含む"),
        _assert("Sent" in r_u, "update"),
        _assert("line1" in r_o, "output"),
        _assert("stopped" in r_s, "stop"),
    ])


def test_task_not_found():
    print("== task: not found ==")
    reg = ToolRegistry()
    task.register(reg)
    return _assert("not found" in reg.execute("TaskGet", {"task_id": "missing"}),
                   "error")


# ============================================================
# worker
# ============================================================

def test_worker_trust_flow():
    print("== worker: create → trust → await → send prompt → complete ==")
    reg = ToolRegistry()
    worker.register(reg)
    r_c = reg.execute("WorkerCreate", {"cwd": "/tmp/w",
                                       "trusted_roots": ["/tmp"]})
    import re
    m = re.search(r"id=(worker_\w+)", r_c)
    wid = m.group(1) if m else ""
    r_trust = reg.execute("WorkerResolveTrust",
                          {"worker_id": wid, "decision": "trust"})
    r_obs = reg.execute("WorkerObserve",
                        {"worker_id": wid, "snapshot": "system ready"})
    r_ready = reg.execute("WorkerAwaitReady", {"worker_id": wid})
    r_prompt = reg.execute("WorkerSendPrompt",
                           {"worker_id": wid, "prompt": "do X"})
    r_done = reg.execute("WorkerObserveCompletion",
                         {"worker_id": wid, "finish_reason": "Finished"})
    return all([
        _assert(wid.startswith("worker_"), "id"),
        _assert("trusted" in r_trust, "trust"),
        _assert("state=awaiting_ready" in r_obs, "observe awaiting"),
        _assert("ready" in r_ready, "ready"),
        _assert("Prompt sent" in r_prompt, "prompt sent"),
        _assert("Finished" in r_done, "complete"),
    ])


def test_worker_trust_deny():
    print("== worker: trust deny ==")
    reg = ToolRegistry()
    worker.register(reg)
    r_c = reg.execute("WorkerCreate", {"cwd": "/tmp"})
    import re
    wid = re.search(r"id=(worker_\w+)", r_c).group(1)
    r = reg.execute("WorkerResolveTrust",
                    {"worker_id": wid, "decision": "deny"})
    return _assert("denied" in r, "denied")


def test_worker_restart_terminate():
    print("== worker: restart / terminate ==")
    reg = ToolRegistry()
    worker.register(reg)
    r_c = reg.execute("WorkerCreate", {"cwd": "/tmp"})
    import re
    wid = re.search(r"id=(worker_\w+)", r_c).group(1)
    r_r = reg.execute("WorkerRestart", {"worker_id": wid})
    r_t = reg.execute("WorkerTerminate", {"worker_id": wid})
    r_nf = reg.execute("WorkerGet", {"worker_id": wid})
    return all([
        _assert("restarted" in r_r, "restart"),
        _assert("terminated" in r_t, "terminate"),
        _assert("not found" in r_nf, "消えた"),
    ])


# ============================================================
# team_cron
# ============================================================

def test_team_create_delete():
    print("== team: create/delete ==")
    reg = ToolRegistry()
    team_cron.register(reg)
    r_c = reg.execute("TeamCreate", {"name": "squad",
                                     "members": ["a", "b"], "tasks": ["t1"]})
    import re
    tid = re.search(r"id=(team_\w+)", r_c).group(1)
    r_d = reg.execute("TeamDelete", {"team_id": tid})
    return all([
        _assert(tid.startswith("team_"), "id"),
        _assert("deleted" in r_d, "delete"),
    ])


def test_team_requires_members():
    print("== team: members required ==")
    reg = ToolRegistry()
    team_cron.register(reg)
    out = reg.execute("TeamCreate", {"name": "x", "members": []})
    return _assert("required" in out, "members 必須")


def test_cron_create_list_delete():
    print("== cron: create/list/delete ==")
    reg = ToolRegistry()
    team_cron.register(reg)
    r_c = reg.execute("CronCreate", {"schedule": "0 * * * *",
                                     "prompt": "hourly"})
    import re
    cid = re.search(r"id=(cron_\w+)", r_c).group(1)
    r_l = reg.execute("CronList", {})
    r_d = reg.execute("CronDelete", {"cron_id": cid})
    return all([
        _assert(cid.startswith("cron_"), "id"),
        _assert(cid in r_l, "list"),
        _assert("deleted" in r_d, "delete"),
    ])


def test_cron_invalid_schedule():
    print("== cron: invalid schedule ==")
    reg = ToolRegistry()
    team_cron.register(reg)
    out = reg.execute("CronCreate", {"schedule": "not cron",
                                     "prompt": "x"})
    return _assert("invalid cron" in out, "拒否")


# ============================================================
# lsp
# ============================================================

def test_lsp_no_backend():
    print("== lsp: backend 未設定 ==")
    reg = ToolRegistry()
    lsp.register(reg)
    out = reg.execute("LSP", {"action": "symbols", "path": "a.py"})
    return _assert("pending" in out, "pending")


def test_lsp_invalid_action():
    print("== lsp: invalid action ==")
    reg = ToolRegistry()
    lsp.register(reg)
    out = reg.execute("LSP", {"action": "eval"})
    return _assert("unknown action" in out, "拒否")


def test_lsp_with_backend():
    print("== lsp: backend 注入 ==")
    reg = ToolRegistry()
    lsp.register(reg)
    lsp.set_lsp_backend(lambda action, **kw: f"{action}:{kw.get('path', '')}")
    out = reg.execute("LSP", {"action": "hover", "path": "a.py"})
    lsp.set_lsp_backend(lambda *a, **kw: None)  # reset
    return _assert("hover:a.py" in out, "backend 応答")


# ============================================================
# mcp
# ============================================================

def test_mcp_no_bridge():
    print("== mcp: bridge 未設定 ==")
    reg = ToolRegistry()
    mcp.register(reg)
    out = reg.execute("MCP", {"server": "slack", "tool": "post",
                              "arguments": {"text": "hi"}})
    return _assert("pending" in out, "pending")


def test_mcp_with_bridge():
    print("== mcp: bridge 注入 ==")
    reg = ToolRegistry()
    mcp.register(reg)
    mcp.set_mcp_bridge(
        call_tool=lambda s, t, a: f"{s}:{t}:{a}",
        list_resources=lambda s: [{"uri": "foo://1", "name": "A"}],
        read_resource=lambda s, u: f"read {s} {u}",
        auth=lambda s: f"auth {s} OK",
    )
    r_call = reg.execute("MCP", {"server": "slack", "tool": "post",
                                  "arguments": {"x": 1}})
    r_list = reg.execute("ListMcpResources", {"server": "slack"})
    r_read = reg.execute("ReadMcpResource",
                         {"server": "slack", "uri": "foo://1"})
    r_auth = reg.execute("McpAuth", {"server": "slack"})
    # reset
    mcp.set_mcp_bridge(call_tool=lambda *a, **kw: None,
                        list_resources=lambda *a, **kw: None,
                        read_resource=lambda *a, **kw: None,
                        auth=lambda *a, **kw: None)
    return all([
        _assert("slack:post" in r_call, "call"),
        _assert("foo://1" in r_list, "list"),
        _assert("read slack foo://1" in r_read, "read"),
        _assert("auth slack OK" in r_auth, "auth"),
    ])


def test_mcp_tool_name_helper():
    print("== mcp: tool_name 正規化 ==")
    name = mcp.mcp_tool_name("slack-bot", "post-msg")
    return _assert(name == "mcp__slack_bot__post_msg", "正規化")


# ============================================================
# register_all
# ============================================================

def test_register_all():
    print("== register_all: 50 tool 全登録 ==")
    reg = ToolRegistry()
    register_all(reg, TMPDIR,
                 settings_path=TMPDIR / "settings.json",
                 skill_dirs=[str(TMPDIR / "skills")])
    names = set(reg.all_names())
    expected = {
        # file_ops (5)
        "read_file", "write_file", "edit_file", "glob_search", "grep_search",
        # shell (3)
        "bash", "PowerShell", "REPL",
        # web (3)
        "WebFetch", "WebSearch", "RemoteTrigger",
        # task (6)
        "TaskCreate", "TaskGet", "TaskList", "TaskStop",
        "TaskUpdate", "TaskOutput",
        # worker (9)
        "WorkerCreate", "WorkerGet", "WorkerObserve", "WorkerResolveTrust",
        "WorkerAwaitReady", "WorkerSendPrompt", "WorkerRestart",
        "WorkerTerminate", "WorkerObserveCompletion",
        # team_cron (5)
        "TeamCreate", "TeamDelete", "CronCreate", "CronList", "CronDelete",
        # lsp (1)
        "LSP",
        # mcp (4)
        "MCP", "ListMcpResources", "ReadMcpResource", "McpAuth",
        # ui (4)
        "AskUserQuestion", "SendUserMessage", "StructuredOutput", "Config",
        # plan (2)
        "EnterPlanMode", "ExitPlanMode",
        # skill (3)
        "Skill", "Agent", "ToolSearch",
        # util (5)
        "Sleep", "TodoWrite", "NotebookEdit", "TestingPermission",
        "RunTaskPacket",
    }
    missing = expected - names
    return all([
        _assert(not missing, f"missing: {missing}"),
        _assert(len(expected) == 50, f"期待 50 tool (got {len(expected)})"),
    ])


def main():
    _setup()
    try:
        tests = [
            test_plan,
            test_util_sleep, test_util_sleep_too_large,
            test_util_todowrite, test_util_notebook_edit,
            test_util_notebook_insert_delete, test_util_testing_permission,
            test_util_run_task_packet,
            test_ui_config_get_set, test_ui_ask_user_no_bridge,
            test_ui_ask_user_with_bridge, test_ui_structured_output,
            test_skill_load, test_skill_not_found,
            test_skill_tool_search, test_skill_agent_no_dispatcher,
            test_task_lifecycle, test_task_not_found,
            test_worker_trust_flow, test_worker_trust_deny,
            test_worker_restart_terminate,
            test_team_create_delete, test_team_requires_members,
            test_cron_create_list_delete, test_cron_invalid_schedule,
            test_lsp_no_backend, test_lsp_invalid_action, test_lsp_with_backend,
            test_mcp_no_bridge, test_mcp_with_bridge, test_mcp_tool_name_helper,
            test_register_all,
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
