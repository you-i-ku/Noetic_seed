"""config / compaction / git_context / sandbox / bash_validation / usage /
session_store の統合テスト。"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.config import (
    RuntimeConfig, McpServerConfig, OAuthConfig,
    deep_merge, load_runtime_config,
)
from core.runtime.compaction import (
    estimate_session_tokens, should_compact, compact_session,
    get_compact_continuation_message,
    DEFAULT_AUTO_COMPACT_THRESHOLD,
)
from core.runtime.git_context import GitContext
from core.runtime.sandbox import (
    SandboxConfig, FilesystemMode, wrap_command,
    detect_container, unshare_available,
)
from core.runtime.bash_validation import (
    validate_bash, ValidationSeverity,
)
from core.runtime.usage import (
    UsageSummary, CostTracker,
    pricing_for_model, max_tokens_for_model,
)
from core.runtime.session import Session
from core.runtime.session_store import SessionStore
from core.providers.base import AssistantMessage, ToolUseBlock


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


TMPDIR: Path = None


def _setup():
    global TMPDIR
    TMPDIR = Path(tempfile.mkdtemp(prefix="clawcode_adv_"))


def _teardown():
    if TMPDIR and TMPDIR.exists():
        shutil.rmtree(TMPDIR)


# ============================================================
# Config
# ============================================================

def test_deep_merge():
    print("== config: deep_merge ==")
    base = {"a": 1, "b": {"c": 2, "d": 3}, "list": [1, 2]}
    over = {"b": {"c": 20, "e": 4}, "list": [9], "f": 5}
    out = deep_merge(base, over)
    return all([
        _assert(out["a"] == 1, "base のみ"),
        _assert(out["b"]["c"] == 20, "nested 上書き"),
        _assert(out["b"]["d"] == 3, "nested 保持"),
        _assert(out["b"]["e"] == 4, "nested 追加"),
        _assert(out["list"] == [9], "list は上書き"),
        _assert(out["f"] == 5, "新規"),
    ])


def test_config_3level_merge():
    print("== config: 3-level merge (user/project/local) ==")
    home = TMPDIR / "home"
    ws = TMPDIR / "ws"
    (home / ".claw").mkdir(parents=True)
    (ws / ".claw").mkdir(parents=True)
    (home / ".claw" / "settings.json").write_text(
        json.dumps({"model": "haiku", "permission_mode": "read_only",
                    "hooks": {"pre": ["user_hook"]}}),
        encoding="utf-8",
    )
    (ws / ".claw" / "settings.json").write_text(
        json.dumps({"model": "sonnet",
                    "hooks": {"post": ["proj_hook"]}}),
        encoding="utf-8",
    )
    (ws / ".claw" / "settings.local.json").write_text(
        json.dumps({"permission_mode": "workspace_write"}),
        encoding="utf-8",
    )
    cfg = load_runtime_config(workspace_root=ws, user_home=home)
    return all([
        _assert(cfg.model == "sonnet", "project 上書き"),
        _assert(cfg.permission_mode == "workspace_write",
                "local override"),
        _assert(cfg.hooks.get("pre") == ["user_hook"], "user hook 保持"),
        _assert(cfg.hooks.get("post") == ["proj_hook"], "project hook"),
    ])


def test_mcp_server_config():
    print("== config: McpServerConfig.from_dict ==")
    stdio = McpServerConfig.from_dict("slack", {
        "type": "stdio", "command": "mcp-slack",
        "args": ["--verbose"], "env": {"TOKEN": "x"},
    })
    remote = McpServerConfig.from_dict("api", {
        "type": "http", "url": "https://api.example.com",
        "headers": {"X": "y"},
    })
    sdk = McpServerConfig.from_dict("internal", {
        "type": "sdk", "name": "builtin-mcp",
    })
    return all([
        _assert(stdio.command == "mcp-slack" and stdio.args == ["--verbose"],
                "stdio"),
        _assert(stdio.env["TOKEN"] == "x", "stdio env"),
        _assert(remote.url == "https://api.example.com", "http url"),
        _assert(remote.headers["X"] == "y", "http headers"),
        _assert(sdk.sdk_name == "builtin-mcp", "sdk name"),
    ])


def test_oauth_config():
    print("== config: OAuthConfig ==")
    oauth = OAuthConfig.from_dict({
        "client_id": "cid", "callback_port": 8080,
        "authorize_url": "https://auth", "token_url": "https://tok",
        "scopes": ["read", "write"],
    })
    return all([
        _assert(oauth.client_id == "cid", "client_id"),
        _assert(oauth.callback_port == 8080, "callback_port"),
        _assert(oauth.scopes == ["read", "write"], "scopes"),
    ])


def test_config_missing_files():
    print("== config: ファイル欠損でも空 config を返す ==")
    empty = TMPDIR / "empty_ws"
    empty.mkdir()
    cfg = load_runtime_config(workspace_root=empty, user_home=empty)
    return all([
        _assert(cfg.model is None, "model None"),
        _assert(cfg.mcp_servers == {}, "mcp 空"),
        _assert(cfg.max_tools_per_cycle == 1, "default 1"),
    ])


# ============================================================
# Compaction
# ============================================================

def test_compaction_token_estimate():
    print("== compaction: estimate_session_tokens ==")
    s = Session()
    s.push_user_text("hello world" * 100)
    tokens = estimate_session_tokens(s)
    return _assert(tokens > 0, f"トークン推定 > 0 (got {tokens})")


def test_compaction_should_compact():
    print("== compaction: should_compact 閾値判定 ==")
    s = Session()
    s.push_user_text("x")
    return all([
        _assert(not should_compact(s, threshold=10_000),
                "短い session は false"),
        _assert(should_compact(s, threshold=0),
                "閾値 0 なら true"),
    ])


def test_compaction_compact():
    print("== compaction: compact_session ==")
    s = Session()
    for i in range(30):
        s.push_user_text(f"message {i} " + "x" * 100)
    before_count = len(s.messages)
    result = compact_session(s, keep_recent=5)
    return all([
        _assert(result.removed_count > 0, "removed > 0"),
        _assert(result.kept_count == 5, "keep_recent=5"),
        _assert(len(s.messages) == 6, "要約 1 + keep 5"),
        _assert("compacted history" in s.messages[0]["content"][0]["text"],
                "要約プレフィックス"),
    ])


def test_compaction_custom_summarizer():
    print("== compaction: カスタム summarizer ==")
    s = Session()
    for i in range(10):
        s.push_user_text(f"msg {i}")
    result = compact_session(
        s,
        summarize_fn=lambda msgs: f"CUSTOM SUMMARY of {len(msgs)} msgs",
        keep_recent=2,
    )
    head_text = s.messages[0]["content"][0]["text"]
    return _assert("CUSTOM SUMMARY" in head_text, "custom summarizer 採用")


def test_compaction_continuation_message():
    print("== compaction: continuation message ==")
    msg = get_compact_continuation_message()
    return _assert("compacted" in msg, "文字列含む")


# ============================================================
# Git Context
# ============================================================

def test_git_context_not_a_repo():
    print("== git_context: 非 git ディレクトリ ==")
    non_git = TMPDIR / "non_git"
    non_git.mkdir()
    ctx = GitContext.detect(cwd=non_git)
    return all([
        _assert(not ctx.is_repo, "is_repo False"),
        _assert(ctx.to_prompt_block() == "", "prompt 空"),
    ])


def test_git_context_real_repo():
    print("== git_context: プロジェクト repo ==")
    # 本リポジトリで実行
    proj_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    ctx = GitContext.detect(cwd=proj_root)
    if not ctx.is_repo:
        print("  [SKIP] not in a git repo")
        return True
    block = ctx.to_prompt_block()
    return all([
        _assert(ctx.is_repo, "is_repo True"),
        _assert(ctx.branch is not None, "branch 取得"),
        _assert("[git context]" in block, "prompt block header"),
        _assert(f"branch: {ctx.branch}" in block, "branch 表示"),
    ])


# ============================================================
# Sandbox
# ============================================================

def test_sandbox_wrap_off():
    print("== sandbox: filesystem=OFF で no-op ==")
    cfg = SandboxConfig(filesystem=FilesystemMode.OFF,
                        network_isolated=False)
    wrapped = wrap_command(["bash", "-c", "echo x"], cfg)
    return _assert(wrapped == ["bash", "-c", "echo x"], "そのまま")


def test_sandbox_wrap_network():
    print("== sandbox: network_isolated ==")
    import platform
    cfg = SandboxConfig(network_isolated=True)
    wrapped = wrap_command(["bash", "-c", "x"], cfg)
    if platform.system() == "Linux" and unshare_available():
        return _assert("unshare" in wrapped[0],
                       "Linux で unshare ラップ")
    return _assert(wrapped == ["bash", "-c", "x"],
                   "非 Linux では no-op")


def test_sandbox_detect_container():
    print("== sandbox: detect_container ==")
    result = detect_container()
    # 実環境 (非 container 想定) で None になるか、または特定値
    return _assert(result is None or isinstance(result, str),
                   f"戻り値型 OK (got {result!r})")


# ============================================================
# Bash Validation
# ============================================================

def test_validate_destructive():
    print("== bash_validation: destructive deny ==")
    r1 = validate_bash("rm -rf /")
    r2 = validate_bash(":(){ :|:& };:")
    r3 = validate_bash("dd if=/dev/zero of=/dev/sda")
    return all([
        _assert(r1.denied, "rm -rf / deny"),
        _assert(r2.denied, "fork bomb deny"),
        _assert(r3.denied, "dd to raw device deny"),
    ])


def test_validate_warn():
    print("== bash_validation: warn ==")
    r = validate_bash("sudo apt install foo")
    return _assert(r.severity == ValidationSeverity.WARN, "sudo warn")


def test_validate_safe():
    print("== bash_validation: 安全なコマンド ==")
    r = validate_bash("ls -la")
    return _assert(r.severity == ValidationSeverity.OK, "OK")


def test_validate_read_only_whitelist():
    print("== bash_validation: read_only mode ==")
    r1 = validate_bash("ls /tmp", read_only_mode=True)
    r2 = validate_bash("rm file.txt", read_only_mode=True)
    r3 = validate_bash("ls; rm x", read_only_mode=True)
    return all([
        _assert(r1.severity == ValidationSeverity.OK, "ls allow"),
        _assert(r2.denied, "rm deny"),
        _assert(r3.denied, "compound deny"),
    ])


def test_validate_empty():
    print("== bash_validation: 空 ==")
    r = validate_bash("")
    return _assert(r.denied, "空は deny")


# ============================================================
# Usage / Cost
# ============================================================

def test_usage_add():
    print("== usage: UsageSummary.add ==")
    u = UsageSummary()
    u.add({"input_tokens": 100, "output_tokens": 50}, model="claude-sonnet-4-6")
    u.add({"prompt_tokens": 30, "completion_tokens": 20}, model="claude-sonnet-4-6")
    return all([
        _assert(u.input_tokens == 130, "input 合算"),
        _assert(u.output_tokens == 70, "output 合算"),
        _assert(u.request_count == 2, "request"),
    ])


def test_cost_tracker():
    print("== usage: CostTracker.report ==")
    ct = CostTracker()
    ct.record({"input_tokens": 1000, "output_tokens": 500},
              model="claude-sonnet-4-6")
    report = ct.report()
    return all([
        _assert("Tokens" in report, "tokens 行"),
        _assert("Estimated cost" in report, "cost 行"),
    ])


def test_pricing_lookup():
    print("== usage: pricing_for_model ==")
    return all([
        _assert(pricing_for_model("claude-opus-4-6") is not None,
                "opus"),
        _assert(pricing_for_model("claude-sonnet-4-6") is not None,
                "sonnet"),
        _assert(pricing_for_model("unknown-model") is None,
                "未知"),
    ])


def test_max_tokens_lookup():
    print("== usage: max_tokens_for_model ==")
    return all([
        _assert(max_tokens_for_model("claude-sonnet-4-6") >= 32000,
                "sonnet >= 32k"),
        _assert(max_tokens_for_model("unknown") == 4096, "default"),
    ])


# ============================================================
# Session Store
# ============================================================

def test_session_store_save_load():
    print("== session_store: save/load ==")
    store = SessionStore(TMPDIR / "sessions")
    s = Session()
    s.push_user_text("hello")
    s.push_assistant_message(AssistantMessage(
        text="hi",
        tool_uses=[ToolUseBlock(id="t1", name="read_file",
                                input={"path": "a"})],
    ))
    sid = store.save(s, metadata={"provider": "anthropic"})
    loaded = store.load(sid)
    return all([
        _assert(loaded is not None, "load 成功"),
        _assert(len(loaded.messages) == 2, "2 message 復元"),
        _assert(loaded.messages[0]["role"] == "user", "user 復元"),
    ])


def test_session_store_list_latest():
    print("== session_store: list + load_latest ==")
    import time
    store = SessionStore(TMPDIR / "sessions2")
    s1 = Session(); s1.push_user_text("a")
    id1 = store.save(s1)
    time.sleep(1.1)  # ファイル名の秒単位タイムスタンプに差をつける
    s2 = Session(); s2.push_user_text("b")
    id2 = store.save(s2)
    lst = store.list_sessions()
    latest = store.load_latest()
    return all([
        _assert(len(lst) >= 2, ">= 2 sessions"),
        _assert(lst[0]["id"] == id2, "最新が先頭"),
        _assert(latest is not None, "latest load"),
        _assert(latest.messages[0]["content"][0]["text"] == "b",
                "latest 内容"),
    ])


def test_session_store_delete():
    print("== session_store: delete ==")
    store = SessionStore(TMPDIR / "sessions3")
    s = Session(); s.push_user_text("x")
    sid = store.save(s)
    ok = store.delete(sid)
    return all([
        _assert(ok, "delete success"),
        _assert(store.load(sid) is None, "削除後 load None"),
    ])


def main():
    _setup()
    try:
        tests = [
            test_deep_merge, test_config_3level_merge,
            test_mcp_server_config, test_oauth_config,
            test_config_missing_files,
            test_compaction_token_estimate, test_compaction_should_compact,
            test_compaction_compact, test_compaction_custom_summarizer,
            test_compaction_continuation_message,
            test_git_context_not_a_repo, test_git_context_real_repo,
            test_sandbox_wrap_off, test_sandbox_wrap_network,
            test_sandbox_detect_container,
            test_validate_destructive, test_validate_warn,
            test_validate_safe, test_validate_read_only_whitelist,
            test_validate_empty,
            test_usage_add, test_cost_tracker,
            test_pricing_lookup, test_max_tokens_lookup,
            test_session_store_save_load, test_session_store_list_latest,
            test_session_store_delete,
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
