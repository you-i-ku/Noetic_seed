"""C-2b migration: operation failures versus empty/partial successful searches."""
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from core.runtime.conversation import ConversationRuntime
from core.runtime.hooks import HookRunner, HookRunResult
from core.runtime.permissions import PermissionEnforcer, PermissionMode
from core.runtime.registry import ToolError, ToolRegistry, ToolResult
from core.runtime.tools import file_ops, shell, web
from core.runtime.hooks import make_file_access_guard


@pytest.fixture
def registry(tmp_path):
    reg = ToolRegistry()
    file_ops.register(reg, tmp_path)
    shell.register(reg)
    web.register(reg)
    return reg


def invoke(reg, name, args):
    hooks = HookRunner()
    success = Mock(return_value=HookRunResult.allow())
    failure = Mock(return_value=HookRunResult.allow())
    hooks.register_post(success)
    hooks.register_failure(failure)
    runtime = ConversationRuntime(SimpleNamespace(name="anthropic"), reg, hooks,
                                  PermissionEnforcer(PermissionMode.ALLOW))
    record = runtime._execute_tool_use("test", name, args)
    assert success.call_count == (not record.is_error)
    assert failure.call_count == record.is_error
    return record


@pytest.mark.parametrize("name", ["glob_search", "grep_search"])
@pytest.mark.parametrize("path", [".", ""])
@pytest.mark.parametrize("public", [True, False])
def test_root_search_filters_secret_paths_before_reads(registry, tmp_path, monkeypatch, name, path, public):
    secret_dir = tmp_path / "sandbox" / "secrets"
    secret_dir.mkdir(parents=True)
    (tmp_path / "secrets.json").write_text("MATCH SECRET_ROOT_CONTENT")
    (secret_dir / "hidden-key.txt").write_text("MATCH SECRET_NESTED_CONTENT")
    if public:
        (tmp_path / "public.md").write_text("MATCH PUBLIC_CONTENT")
    reads = []
    original = Path.read_bytes
    def read(current):
        reads.append(current)
        return original(current)
    monkeypatch.setattr(Path, "read_bytes", read)
    hooks = HookRunner()
    hooks.register_pre(make_file_access_guard(tmp_path))
    rt = ConversationRuntime(SimpleNamespace(name="anthropic"), registry, hooks,
                             PermissionEnforcer(PermissionMode.ALLOW))
    rec = rt._execute_tool_use("search", name, {"path": path, "pattern": "**/*" if name == "glob_search" else "MATCH"})
    assert not rec.is_error
    assert ("public.md" in rec.output) is public
    if name == "grep_search":
        assert ("PUBLIC_CONTENT" in rec.output) is public
    assert rec.detail == {"excluded_paths": 3}  # file + directory + child, independent of regex hits
    for text in ("secrets.json", "sandbox/secrets", "hidden-key", "SECRET_ROOT_CONTENT", "SECRET_NESTED_CONTENT"):
        assert text not in rec.output + str(rec.detail)
    assert all(p == tmp_path / "public.md" for p in reads)


@pytest.mark.parametrize("name", ["glob_search", "grep_search"])
@pytest.mark.parametrize("stage", ["secret_candidate", "enumeration", "start", "candidate", "recheck"])
def test_search_path_errors_never_expose_secret_names(registry, tmp_path, monkeypatch, name, stage):
    secret = tmp_path / "sandbox" / "secrets" / "private-key-name.txt"
    public = tmp_path / "public.txt"
    public.write_text("MATCH")
    original_resolve = Path.resolve
    resolutions = []
    def resolve(path, *args, **kwargs):
        resolutions.append(path)
        if (path == secret or (stage == "start" and path == tmp_path)
                or (stage == "candidate" and path == public)
                or (stage == "recheck" and path == public and resolutions.count(public) == 2)):
            raise OSError(f"cannot resolve {secret}")
        return original_resolve(path, *args, **kwargs)
    def glob(path, pattern):
        if stage == "enumeration":
            raise OSError(f"cannot enumerate {secret}")
        return iter([secret] if stage == "secret_candidate" else [public])
    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(Path, "glob", glob)
    args = {"path": ".", "pattern": "**/*" if name == "glob_search" else "MATCH"}
    rec = invoke(registry, name, args)
    exposed = rec.output + repr(rec.detail)
    assert "private-key-name" not in exposed and str(tmp_path) not in exposed
    if stage == "secret_candidate":
        assert not rec.is_error and rec.detail == {"excluded_paths": 1}
        assert secret not in resolutions  # Sanitizing a resolve failure alone is insufficient.
    else:
        assert rec.is_error and rec.error_kind == "tool_failure"
        assert rec.detail == {"exception_type": "OSError"}
        with pytest.raises(ToolError) as caught:
            resolutions.clear()
            registry.execute(name, args)
        assert "private-key-name" not in str(caught.value) + repr(caught.value.detail)


@pytest.mark.parametrize("name", ["glob_search", "grep_search"])
def test_search_aliases_and_outside_boundary(registry, tmp_path, name):
    (tmp_path / "secrets.json").write_text("MATCH SECRET_CONTENT")
    (tmp_path / "public.md").write_text("MATCH PUBLIC_CONTENT")
    outside = tmp_path.parent / (tmp_path.name + "-outside.txt")
    outside.write_text("MATCH OUTSIDE_CONTENT")
    try:
        (tmp_path / "alias.txt").symlink_to(tmp_path / "secrets.json")
        (tmp_path / "escape.txt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    rec = invoke(registry, name, {"path": ".", "pattern": "**/*" if name == "glob_search" else "MATCH"})
    assert not rec.is_error and "public.md" in rec.output
    assert rec.detail == {"excluded_paths": 3}
    assert all(text not in rec.output for text in ("alias.txt", "escape.txt", "SECRET_CONTENT", "OUTSIDE_CONTENT"))
    rec = invoke(registry, name, {"path": "..", "pattern": "*" if name == "glob_search" else "MATCH"})
    assert rec.is_error and rec.error_kind == "tool_failure"


@pytest.mark.parametrize("name", ["glob_search", "grep_search"])
def test_glob_braces_are_explicit_tool_failure(registry, tmp_path, name):
    (tmp_path / "sample.json").write_text("MATCH")
    (tmp_path / "sample.md").write_text("MATCH")
    args = {"path": ".", "pattern": "**/*.{json,md}"} if name == "glob_search" else {
        "path": ".", "pattern": "MATCH", "glob": "**/*.{json,md}"}
    rec = invoke(registry, name, args)
    assert rec.is_error and rec.error_kind == "tool_failure"
    assert "brace expansion is not supported" in rec.output
    assert "No matches" not in rec.output
    args["pattern" if name == "glob_search" else "glob"] = "**/*.md"
    rec = invoke(registry, name, args)
    assert not rec.is_error and "sample.md" in rec.output and "sample.json" not in rec.output


@pytest.mark.parametrize("name,args", [
    ("read_file", {"path": "missing"}), ("read_file", {"path": ""}),
    ("read_file", {"path": "../outside"}), ("read_file", {"path": "x", "offset": -1}),
    ("read_file", {"path": "x", "limit": "bad"}),
    ("write_file", {"path": "../outside", "content": ""}),
    ("write_file", {"path": "x", "content": 12}),
    ("edit_file", {"path": "missing", "old_string": "x", "new_string": ""}),
    ("glob_search", {"pattern": "*", "path": "missing"}),
    ("grep_search", {"pattern": "["}),
])
def test_validation_is_tool_failure(registry, name, args):
    record = invoke(registry, name, args)
    assert record.is_error and record.error_kind == "tool_failure"
    assert not record.output.startswith("tool execution error:")


def test_large_read_keeps_original_message_not_broad_except(registry, tmp_path, monkeypatch):
    (tmp_path / "x").write_text("1234")
    monkeypatch.setattr(file_ops, "MAX_FILE_SIZE", 3)
    with pytest.raises(ToolError, match=r"^Error: file too large \(4 bytes, max 3\)$"):
        registry.execute("read_file", {"path": "x"})


def test_empty_and_error_word_are_success(registry, tmp_path):
    (tmp_path / "empty").write_text("")
    (tmp_path / "text").write_text("エラー Error is quoted data", encoding="utf-8")
    for name, args in [("read_file", {"path": "empty"}),
                       ("read_file", {"path": "text", "offset": 100}),
                       ("write_file", {"path": "empty", "content": ""}),
                       ("edit_file", {"path": "text", "old_string": "Error", "new_string": "Error"}),
                       ("glob_search", {"pattern": "*.absent"}),
                       ("grep_search", {"pattern": "エラー"})]:
        rec = invoke(registry, name, args)
        assert not rec.is_error and rec.error_kind is None


@pytest.mark.parametrize("case", ["zero", "skipped", "partial", "all"])
def test_grep_four_outcomes(registry, tmp_path, monkeypatch, case):
    if case == "skipped":
        (tmp_path / "binary").write_bytes(b"\0data")
        (tmp_path / "large").write_bytes(b"x" * 100)
        monkeypatch.setattr(file_ops, "MAX_FILE_SIZE", 50)
    elif case in ("partial", "all"):
        (tmp_path / "bad").write_text("x")
        if case == "partial":
            (tmp_path / "good").write_text("Error is data")
        original = Path.read_bytes
        def read(path):
            if path.name == "bad":
                raise PermissionError("denied")
            return original(path)
        monkeypatch.setattr(Path, "read_bytes", read)
    rec = invoke(registry, "grep_search", {"pattern": "Error"})
    assert rec.is_error is (case == "all")
    if case in ("zero", "skipped"):
        assert rec.detail is None and rec.output == "No matches for pattern: Error"
    else:
        assert rec.detail == {"unreadable_files": 1, "readable_files": int(case == "partial")}
        assert rec.error_kind == ("tool_failure" if case == "all" else None)


@pytest.mark.parametrize("returncode", [0, 1, 42])
def test_bash_exit_status_and_facts(registry, monkeypatch, returncode):
    monkeypatch.setattr(shell, "_find_bash_executable", lambda: "mock-bash")
    monkeypatch.setattr(shell.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        stdout="Error is ordinary output", stderr="stderr text", returncode=returncode))
    rec = invoke(registry, "bash", {"command": "requested command"})
    assert rec.output == shell._format_result("Error is ordinary output", "stderr text", returncode)
    assert rec.is_error is (returncode != 0)
    if returncode:
        assert rec.detail == {"command": "requested command", "returncode": returncode,
                              "stdout": "Error is ordinary output", "stderr": "stderr text"}


def test_bash_timeout_retains_partial_output_after_redaction(registry, monkeypatch):
    monkeypatch.setattr(shell, "_find_bash_executable", lambda: "mock-bash")
    monkeypatch.setattr("core.auth._load_secrets", lambda: {"auth_profiles": {"x": {"token": "secret-value"}}})
    error = subprocess.TimeoutExpired("cmd", 2, output=b"partial stdout",
                                      stderr=("secret-value" * 300 + "END").encode())
    monkeypatch.setattr(shell.subprocess, "run", Mock(side_effect=error))
    rec = invoke(registry, "bash", {"command": "cmd", "timeout": 2})
    assert rec.output == "Error: timeout after 2s" and rec.error_kind == "tool_failure"
    assert rec.detail["stdout"] == "partial stdout" and rec.detail["stderr"].endswith("END")
    assert rec.detail["timeout"] == 2 and rec.detail["returncode"] is None
    assert "secret-value" not in str(rec.detail) and rec.detail["_truncated"] is True


def test_bash_background_is_acceptance(registry, monkeypatch):
    monkeypatch.setattr(shell, "_find_bash_executable", lambda: "mock-bash")
    monkeypatch.setattr(shell.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123))
    monkeypatch.setattr(shell, "_bg_tasks", {})
    rec = invoke(registry, "bash", {"command": "later", "run_in_background": True})
    assert not rec.is_error and "pid=123" in rec.output


@pytest.mark.parametrize("name,args", [("WebFetch", {"url": "https://example.test", "prompt": "x"}),
                                       ("WebSearch", {"query": "x"}),
                                       ("RemoteTrigger", {"url": "https://example.test"})])
def test_http_failure(registry, monkeypatch, name, args):
    monkeypatch.setattr("core.auth.apply_auth", lambda h, p, n: (h, p, None))
    response = httpx.Response(403, text="denied", request=httpx.Request("GET", "https://example.test"))
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: response)
    monkeypatch.setattr(httpx, "request", lambda *a, **kw: response)
    rec = invoke(registry, name, args)
    assert rec.is_error and rec.error_kind == "tool_failure" and "403" in rec.output


def test_missing_search_key(registry, monkeypatch):
    monkeypatch.setattr("core.auth.apply_auth", lambda h, p, n: (h, p, "key missing"))
    rec = invoke(registry, "WebSearch", {"query": "x"})
    assert rec.error_kind == "tool_failure" and rec.output.startswith("Error: key missing")


@pytest.mark.parametrize("kind", ["zero", "error_word", "bad_json", "empty_body", "redirect_error"])
def test_web_empty_content_and_search_results(registry, monkeypatch, kind):
    monkeypatch.setattr("core.auth.apply_auth", lambda h, p, n: (h, p, None))
    request = httpx.Request("GET", "https://example.test")
    if kind == "redirect_error":
        monkeypatch.setattr(httpx, "get", Mock(side_effect=httpx.TooManyRedirects("loop", request=request)))
    else:
        text = {"zero": '{"web":{"results":[]}}',
                "error_word": '{"web":{"results":[{"title":"Error example","url":"https://example.test"}]}}',
                "bad_json": 'invalid', "empty_body": ''}[kind]
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(200, text=text, request=request))
    name, args = ("WebFetch", {"url": "https://example.test", "prompt": "x"}) if kind == "empty_body" else ("WebSearch", {"query": "x"})
    rec = invoke(registry, name, args)
    assert rec.is_error is (kind in ("bad_json", "redirect_error"))
