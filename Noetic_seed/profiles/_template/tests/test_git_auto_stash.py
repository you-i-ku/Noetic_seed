"""make_git_auto_stash_hook の自動 stash + 世代管理 test (段階12 Step 3)。

PLAN §5 / §13-1 literal 検証:
  - 書換え対象 path (core/* / tools/* / main.py / .mcp.json) の write/edit で
    git stash push が呼ばれる
  - 書換え対象外 path (sandbox/ / pref.json) は stash 走らない
  - 対象外 tool (read_file 等) は stash 走らない
  - subprocess 失敗時も書換え続行 (allow、PLAN §5-3)
  - max_generations 超過時に古い iku-auto-<profile>-* stash が drop される
  - git 未初期化環境では noop hook (起動継続)

subprocess.run 互換の git_runner mock を inject、実 git は触らない。

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_git_auto_stash.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.hooks import make_git_auto_stash_hook


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup_workspace_with_git():
    """tmp/.git + tmp/Noetic_seed/profiles/test_profile/ を作る。"""
    tmp = Path(tempfile.mkdtemp(prefix="noetic_auto_stash_test_"))
    (tmp / ".git").mkdir()
    profile = tmp / "Noetic_seed" / "profiles" / "test_profile"
    profile.mkdir(parents=True)
    return tmp, profile


def _make_runner_recording(stash_list_output: str = "", push_returncode: int = 0):
    """記録型 runner mock。calls list に呼出を蓄積。"""
    calls = []

    def runner(args, **kwargs):
        calls.append({"args": list(args), "cwd": kwargs.get("cwd")})
        if "list" in args and "stash" in args:
            return subprocess.CompletedProcess(
                args=args, returncode=0,
                stdout=stash_list_output, stderr="",
            )
        if "push" in args:
            return subprocess.CompletedProcess(
                args=args, returncode=push_returncode,
                stdout="", stderr="" if push_returncode == 0 else "fatal: stash error",
            )
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr="",
        )
    return runner, calls


def test_stash_triggered_for_core_path():
    print("== write_file(core/foo.py) で stash 発火 ==")
    tmp, profile = _setup_workspace_with_git()
    runner, calls = _make_runner_recording()
    hook = make_git_auto_stash_hook(profile, "test_profile", git_runner=runner)
    r = hook("write_file", {"path": "core/foo.py", "content": "x"})
    push_calls = [c for c in calls if "push" in c["args"]]
    return all([
        _assert(not r.denied, "allow (書換え続行)"),
        _assert(len(push_calls) == 1,
                f"git stash push が 1 回 (実測: {len(push_calls)})"),
        _assert(any("iku-auto-test_profile-" in str(a)
                    for c in push_calls for a in c["args"]),
                "stash message に iku-auto-test_profile- prefix"),
    ])


def test_stash_triggered_for_tools_path():
    print("== edit_file(tools/sandbox.py) で stash 発火 ==")
    tmp, profile = _setup_workspace_with_git()
    runner, calls = _make_runner_recording()
    hook = make_git_auto_stash_hook(profile, "test_profile", git_runner=runner)
    r = hook("edit_file", {"path": "tools/sandbox.py",
                            "old_string": "a", "new_string": "b"})
    push_calls = [c for c in calls if "push" in c["args"]]
    return _assert(len(push_calls) == 1, "stash 1 回 (tools/ も対象)")


def test_stash_triggered_for_main_py():
    print("== write_file(main.py) で stash 発火 ==")
    tmp, profile = _setup_workspace_with_git()
    runner, calls = _make_runner_recording()
    hook = make_git_auto_stash_hook(profile, "test_profile", git_runner=runner)
    r = hook("write_file", {"path": "main.py", "content": "x"})
    push_calls = [c for c in calls if "push" in c["args"]]
    return _assert(len(push_calls) == 1, "stash 1 回 (main.py = 神経中枢)")


def test_stash_triggered_for_mcp_json():
    print("== write_file(.mcp.json) で stash 発火 ==")
    tmp, profile = _setup_workspace_with_git()
    runner, calls = _make_runner_recording()
    hook = make_git_auto_stash_hook(profile, "test_profile", git_runner=runner)
    r = hook("write_file", {"path": ".mcp.json", "content": "{}"})
    push_calls = [c for c in calls if "push" in c["args"]]
    return _assert(len(push_calls) == 1, "stash 1 回 (.mcp.json = 関係性の器)")


def test_stash_skipped_for_sandbox_path():
    print("== write_file(sandbox/memo.md) は stash 対象外 ==")
    tmp, profile = _setup_workspace_with_git()
    runner, calls = _make_runner_recording()
    hook = make_git_auto_stash_hook(profile, "test_profile", git_runner=runner)
    r = hook("write_file", {"path": "sandbox/memo.md", "content": "x"})
    push_calls = [c for c in calls if "push" in c["args"]]
    return all([
        _assert(not r.denied, "allow"),
        _assert(len(push_calls) == 0, "stash 呼ばれない"),
    ])


def test_stash_skipped_for_pref_json():
    print("== write_file(pref.json) は stash 対象外 ==")
    tmp, profile = _setup_workspace_with_git()
    runner, calls = _make_runner_recording()
    hook = make_git_auto_stash_hook(profile, "test_profile", git_runner=runner)
    r = hook("write_file", {"path": "pref.json", "content": "{}"})
    push_calls = [c for c in calls if "push" in c["args"]]
    return _assert(len(push_calls) == 0, "stash 呼ばれない (認知影響限定)")


def test_stash_skipped_for_read_tool():
    print("== read_file は対象外 tool ==")
    tmp, profile = _setup_workspace_with_git()
    runner, calls = _make_runner_recording()
    hook = make_git_auto_stash_hook(profile, "test_profile", git_runner=runner)
    r = hook("read_file", {"path": "core/foo.py"})
    return _assert(len(calls) == 0, "subprocess 一切呼ばれない")


def test_stash_failure_continues_write():
    print("== stash 失敗時も書換え続行 (PLAN §5-3) ==")
    tmp, profile = _setup_workspace_with_git()
    runner, calls = _make_runner_recording(push_returncode=1)
    hook = make_git_auto_stash_hook(profile, "test_profile", git_runner=runner)
    r = hook("write_file", {"path": "core/foo.py", "content": "x"})
    return _assert(not r.denied, "allow (失敗でも書換え続行)")


def test_old_generations_dropped():
    print("== max_generations=20 超過で古い stash drop ==")
    tmp, profile = _setup_workspace_with_git()
    # 25 世代分の stash list を捏造 (新しい順、stash@{0} が最新)
    stash_list = "\n".join(
        f"stash@{{{i}}}: On master: iku-auto-test_profile-2026{i:04d}"
        for i in range(25)
    )
    runner, calls = _make_runner_recording(stash_list_output=stash_list)
    hook = make_git_auto_stash_hook(
        profile, "test_profile", git_runner=runner, max_generations=20,
    )
    hook("write_file", {"path": "core/foo.py", "content": "x"})
    drop_calls = [c for c in calls if "drop" in c["args"]]
    # 25 - 20 = 5 個 drop されるはず
    return _assert(len(drop_calls) == 5,
                   f"drop 5 回 (実測: {len(drop_calls)})")


def test_no_repo_root_returns_noop():
    print("== git repo 未初期化なら noop hook ==")
    tmp = Path(tempfile.mkdtemp(prefix="noetic_no_repo_"))
    profile = tmp / "Noetic_seed" / "profiles" / "test_profile"
    profile.mkdir(parents=True)
    # .git 作らない
    runner, calls = _make_runner_recording()
    hook = make_git_auto_stash_hook(profile, "test_profile", git_runner=runner)
    r = hook("write_file", {"path": "core/foo.py", "content": "x"})
    return all([
        _assert(not r.denied, "allow"),
        _assert(len(calls) == 0, "subprocess 一切呼ばれない"),
    ])


if __name__ == "__main__":
    groups = [
        ("write: core/ で stash 発火", test_stash_triggered_for_core_path),
        ("edit: tools/ で stash 発火", test_stash_triggered_for_tools_path),
        ("write: main.py で stash 発火", test_stash_triggered_for_main_py),
        ("write: .mcp.json で stash 発火", test_stash_triggered_for_mcp_json),
        ("write: sandbox は stash 対象外", test_stash_skipped_for_sandbox_path),
        ("write: pref.json は stash 対象外", test_stash_skipped_for_pref_json),
        ("read_file は対象外 tool", test_stash_skipped_for_read_tool),
        ("stash 失敗時も書換え続行", test_stash_failure_continues_write),
        ("max_generations 超過で古い stash drop", test_old_generations_dropped),
        ("git 未初期化で noop", test_no_repo_root_returns_noop),
    ]
    results = []
    for _label, fn in groups:
        print()
        ok = fn()
        results.append((_label, ok))
    print()
    print("=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for _label, ok in results:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {_label}")
    print(f"\n  {passed}/{total} groups passed")
    sys.exit(0 if passed == total else 1)
