"""approval_rules.make_policy_fn の test (段階13 Phase 6.2 W 案、PLAN §5-4)。

AgentSpec DSL pattern (arxiv 2503.18666 / ICSE 2026) の Python port +
path_classifier helper 委譲 (W 案、Codex review 1-5 周目 root cause fix)。

検証ケース (CLAUDE.md §5 識別力 + §6 docstring 同期):
  - rules 順次評価 + tools フィルタ
  - body_modify boolean: path_classifier 委譲で Windows case / traversal /
    absolute 全対応 (Codex 4-5 周目 root cause fix)
  - default action: rules 漏れ時の fallback (approve = HITL)
  - schema validation: invalid rule で起動時 fail-fast
  - LLM injection 耐性: tool_input 任意 field 影響ゼロ
  - 識別力試験: rule 抜けの誤実装で auto/approve 反転 → fail
  - path quirk 一網打尽: traversal / absolute / Windows case で全 approve
    (helper 委譲確認、git_auto_stash と同じ正規化)

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_approval_policy.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.approval_rules import make_policy_fn


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# 標準 rules (settings.json と同等、test 単純化のため埋込、新 W schema)
DEFAULT_RULES = [
    {
        "tools": ["bash", "reboot", "secret_write"],
        "action": "approve",
    },
    {
        "tools": ["write_file", "edit_file"],
        "body_modify": True,  # path_classifier 委譲
        "action": "approve",
    },
    {
        "tools": ["write_file", "edit_file"],
        "action": "auto",
    },
    {
        "tools": ["camera_stream", "screen_peek", "mic_record", "http_request"],
        "action": "approve",
    },
    {
        "tools": ["memory_store", "read_file", "view_image", "WebSearch",
                  "x_post", "elyth_post"],
        "action": "auto",
    },
]


def _setup_workspace() -> Path:
    """tempdir を workspace_root として返す"""
    return Path(tempfile.mkdtemp(prefix="noetic_approval_W_test_")).resolve()


# ============================================================
# rule 順次評価 + tools フィルタ
# ============================================================

def test_destructive_tools_approve():
    print("== destructive (bash/reboot/secret_write) は approve ==")
    policy = make_policy_fn(DEFAULT_RULES, default_action="approve")
    return all([
        _assert(policy("bash", {"command": "ls"}) is True, "bash → True"),
        _assert(policy("reboot", {}) is True, "reboot → True"),
        _assert(policy("secret_write", {"name": "x"}) is True,
                "secret_write → True"),
    ])


def test_body_modify_path_approve():
    print("== write_file/edit_file with body_modify path → approve (helper 委譲) ==")
    root = _setup_workspace()
    policy = make_policy_fn(DEFAULT_RULES, default_action="approve",
                             workspace_root=root)
    return all([
        _assert(policy("write_file", {"path": "core/main.py"}) is True,
                "write_file core/ → True"),
        _assert(policy("write_file", {"path": "tools/sandbox.py"}) is True,
                "write_file tools/ → True"),
        _assert(policy("write_file", {"path": "main.py"}) is True,
                "write_file main.py → True"),
        _assert(policy("write_file", {"path": ".mcp.json"}) is True,
                "write_file .mcp.json → True"),
        _assert(policy("edit_file", {"path": "core/state.py"}) is True,
                "edit_file core/ → True"),
    ])


def test_non_body_modify_path_auto():
    print("== write_file/edit_file with 非 body_modify path → auto ==")
    root = _setup_workspace()
    policy = make_policy_fn(DEFAULT_RULES, default_action="approve",
                             workspace_root=root)
    return all([
        _assert(policy("write_file", {"path": "memory/2026-05.jsonl"}) is False,
                "write_file memory/*.jsonl → False"),
        _assert(policy("write_file", {"path": "sandbox/note.md"}) is False,
                "write_file sandbox/ → False"),
        _assert(policy("edit_file", {"path": "memory/2026-05.jsonl"}) is False,
                "edit_file memory/*.jsonl → False"),
    ])


def test_hardware_interference_approve():
    print("== camera/screen/mic/http_request → approve ==")
    policy = make_policy_fn(DEFAULT_RULES, default_action="approve")
    return all([
        _assert(policy("camera_stream", {}) is True, "camera_stream → True"),
        _assert(policy("screen_peek", {}) is True, "screen_peek → True"),
        _assert(policy("mic_record", {}) is True, "mic_record → True"),
        _assert(policy("http_request", {"url": "x"}) is True,
                "http_request → True"),
    ])


def test_normal_tools_auto():
    print("== read_only + write_safe + SNS は auto ==")
    policy = make_policy_fn(DEFAULT_RULES, default_action="approve")
    return all([
        _assert(policy("memory_store", {"content": "x"}) is False,
                "memory_store → False"),
        _assert(policy("read_file", {"path": "core/main.py"}) is False,
                "read_file core/ → False (body_modify rule なし、auto rule hit)"),
        _assert(policy("view_image", {"path": "x.jpg"}) is False,
                "view_image → False"),
        _assert(policy("WebSearch", {"query": "x"}) is False,
                "WebSearch → False"),
        _assert(policy("x_post", {"text": "x"}) is False,
                "x_post → False"),
        _assert(policy("elyth_post", {"content": "x"}) is False,
                "elyth_post → False"),
    ])


# ============================================================
# default action (rules 漏れ時の fallback)
# ============================================================

def test_default_approve_for_unknown_tool():
    print("== 未知 tool は default=approve で承認必要 (HITL fallback) ==")
    policy = make_policy_fn(DEFAULT_RULES, default_action="approve")
    return all([
        _assert(policy("unknown_tool_X", {}) is True,
                "unknown_tool_X → True (default approve)"),
        _assert(policy("future_iku_made_tool", {"any": "input"}) is True,
                "future_iku_made_tool → True"),
    ])


def test_default_auto_for_unknown_tool():
    print("== default=auto なら未知 tool は auto ==")
    policy = make_policy_fn(DEFAULT_RULES, default_action="auto")
    return _assert(policy("unknown_tool_X", {}) is False,
                   "unknown_tool_X → False (default auto)")


# ============================================================
# schema validation (起動時 fail-fast)
# ============================================================

def test_invalid_default_action_raises():
    print("== invalid default_action で ValueError ==")
    try:
        make_policy_fn([], default_action="invalid_action")
        return _assert(False, "ValueError 発生せず")
    except ValueError:
        return _assert(True, "ValueError 発生")


def test_invalid_rule_action_raises():
    print("== invalid rule.action で ValueError ==")
    bad_rules = [{"tools": ["bash"], "action": "invalid_action"}]
    try:
        make_policy_fn(bad_rules)
        return _assert(False, "ValueError 発生せず")
    except ValueError:
        return _assert(True, "ValueError 発生")


def test_invalid_rule_tools_raises():
    print("== empty tools list で ValueError ==")
    bad_rules = [{"tools": [], "action": "approve"}]
    try:
        make_policy_fn(bad_rules)
        return _assert(False, "ValueError 発生せず")
    except ValueError:
        return _assert(True, "ValueError 発生")


# ============================================================
# LLM injection 耐性 (Codex review 3 周目 P1 architecture level fix)
# ============================================================

def test_llm_injection_skip_approval_ignored():
    print("== LLM が tool_input に `_skip_approval=True` 仕込んでも policy_fn 影響なし ==")
    policy = make_policy_fn(DEFAULT_RULES, default_action="approve")
    r = policy("bash", {"command": "rm -rf /tmp", "_skip_approval": True})
    return _assert(r is True,
                   "bash with `_skip_approval=True` → True (rule 評価のみ、LLM field 無視)")


def test_llm_injection_safety_class_ignored():
    print("== LLM が `_safety_class=read_only` 仕込んでも影響なし ==")
    root = _setup_workspace()
    policy = make_policy_fn(DEFAULT_RULES, default_action="approve",
                             workspace_root=root)
    r = policy("write_file", {
        "path": "core/main.py",
        "content": "x",
        "_safety_class": "read_only",
        "_approval": "auto",
    })
    return _assert(r is True,
                   "write_file core/main.py with LLM 仕込み field → True")


# ============================================================
# 識別力試験 (CLAUDE.md §5)
# ============================================================

def test_discrimination_missing_destructive_rule():
    print("== 識別力: rule から bash 抜けた誤実装 → bash が default に落ちる ==")
    bad_rules = [
        {"tools": ["reboot", "secret_write"], "action": "approve"},
        {"tools": ["write_file", "edit_file"],
         "body_modify": True, "action": "approve"},
        {"tools": ["write_file", "edit_file"], "action": "auto"},
        {"tools": ["memory_store"], "action": "auto"},
    ]
    bad_policy = make_policy_fn(bad_rules, default_action="auto")
    bad_bash = bad_policy("bash", {"command": "rm -rf /"})

    good_policy = make_policy_fn(DEFAULT_RULES, default_action="approve")
    good_bash = good_policy("bash", {"command": "rm -rf /"})

    return all([
        _assert(bad_bash is False,
                "誤実装 (bash 抜け + default=auto): bash → False (危険)"),
        _assert(good_bash is True,
                "正実装: bash → True (approve)"),
    ])


def test_discrimination_missing_body_modify_flag():
    print("== 識別力: body_modify rule 自体が抜けた誤実装 → 全 write_file が auto ==")
    root = _setup_workspace()
    bad_rules = [
        {"tools": ["bash"], "action": "approve"},
        # body_modify rule が抜けの誤実装、write_file 全部 auto
        {"tools": ["write_file", "edit_file"], "action": "auto"},
    ]
    bad_policy = make_policy_fn(bad_rules, default_action="approve",
                                 workspace_root=root)
    bad_core = bad_policy("write_file", {"path": "core/main.py"})

    good_policy = make_policy_fn(DEFAULT_RULES, default_action="approve",
                                  workspace_root=root)
    good_core = good_policy("write_file", {"path": "core/main.py"})

    return all([
        _assert(bad_core is False,
                "誤実装: write_file(core/main.py) → False (危険、自己改変無承認)"),
        _assert(good_core is True,
                "正実装: write_file(core/main.py) → True (approve)"),
    ])


# ============================================================
# rules 順序の重要性
# ============================================================

def test_rule_order_priority():
    print("== rules 順序: 最初に hit した rule が勝つ ==")
    rules = [
        {"tools": ["write_file"], "action": "approve"},  # 全部 approve
        {"tools": ["write_file"], "body_modify": True,
         "action": "auto"},  # unreachable
    ]
    root = _setup_workspace()
    policy = make_policy_fn(rules, default_action="auto",
                             workspace_root=root)
    return all([
        _assert(policy("write_file", {"path": "core/main.py"}) is True,
                "core/main.py → True (rule[0] hit)"),
        _assert(policy("write_file", {"path": "memory/x.jsonl"}) is True,
                "memory/x.jsonl → True (rule[0] hit、rule[1] unreachable)"),
    ])


# ============================================================
# path quirk 一網打尽 (W 案、Codex 2/4/5 周目 root cause fix の helper 委譲確認)
# ============================================================

def test_path_traversal_to_body_modify_approve():
    print("== `sandbox/../core/foo.py` traversal → 承認必要 (helper canonical 化) ==")
    root = _setup_workspace()
    policy = make_policy_fn(DEFAULT_RULES, default_action="approve",
                             workspace_root=root)
    return all([
        _assert(policy("write_file", {"path": "sandbox/../core/foo.py"}) is True,
                "sandbox/../core/foo.py → True"),
        _assert(policy("write_file", {"path": "memory/../tools/x.py"}) is True,
                "memory/../tools/x.py → True"),
        _assert(policy("write_file", {"path": "./core/foo.py"}) is True,
                "./core/foo.py → True"),
        _assert(policy("edit_file", {"path": "sandbox/../main.py"}) is True,
                "edit_file sandbox/../main.py → True"),
    ])


def test_path_traversal_to_non_body_modify_auto():
    print("== `core/../sandbox/x.md` traversal → auto ==")
    root = _setup_workspace()
    policy = make_policy_fn(DEFAULT_RULES, default_action="approve",
                             workspace_root=root)
    return all([
        _assert(policy("write_file", {"path": "core/../sandbox/x.md"}) is False,
                "core/../sandbox/x.md → False (canonical: sandbox/x.md)"),
        _assert(policy("write_file", {"path": "tools/../memory/x.jsonl"}) is False,
                "tools/../memory/x.jsonl → False"),
    ])


def test_absolute_path_canonical_match():
    print("== 絶対 path (workspace 内) で body_modify 判定 ==")
    root = _setup_workspace()
    abs_core = str(root / "core" / "main.py")
    policy = make_policy_fn(DEFAULT_RULES, default_action="approve",
                             workspace_root=root)
    return _assert(policy("write_file", {"path": abs_core}) is True,
                   f"絶対 path → True")


def test_backslash_path_body_modify():
    print("== backslash path (Windows) → 承認必要 (helper canonical 化) ==")
    root = _setup_workspace()
    policy = make_policy_fn(DEFAULT_RULES, default_action="approve",
                             workspace_root=root)
    return _assert(policy("write_file", {"path": "core\\foo.py"}) is True,
                   "core\\foo.py → True (backslash POSIX 化)")


def test_windows_case_body_modify():
    print("== Windows case (大文字小文字) → 承認必要 (Codex 5 周目 P2 fix) ==")
    import os
    if os.name != "nt":
        print("  [SKIP] non-Windows env、case sensitivity 試験 skip")
        return True
    root = _setup_workspace()
    policy = make_policy_fn(DEFAULT_RULES, default_action="approve",
                             workspace_root=root)
    return all([
        _assert(policy("write_file", {"path": "Core/foo.py"}) is True,
                "Core/foo.py → True (case insensitive)"),
        _assert(policy("write_file", {"path": "TOOLS/x.py"}) is True,
                "TOOLS/x.py → True"),
        _assert(policy("write_file", {"path": "Main.py"}) is True,
                "Main.py → True"),
        _assert(policy("write_file", {"path": ".MCP.JSON"}) is True,
                ".MCP.JSON → True"),
    ])


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    groups = [
        ("destructive tools approve", test_destructive_tools_approve),
        ("body_modify path approve (helper)", test_body_modify_path_approve),
        ("non body_modify path auto", test_non_body_modify_path_auto),
        ("hardware interference approve", test_hardware_interference_approve),
        ("normal tools auto", test_normal_tools_auto),
        ("default approve for unknown tool", test_default_approve_for_unknown_tool),
        ("default auto for unknown tool", test_default_auto_for_unknown_tool),
        ("invalid default_action raises", test_invalid_default_action_raises),
        ("invalid rule action raises", test_invalid_rule_action_raises),
        ("invalid rule tools raises", test_invalid_rule_tools_raises),
        ("LLM injection _skip_approval ignored",
         test_llm_injection_skip_approval_ignored),
        ("LLM injection _safety_class ignored",
         test_llm_injection_safety_class_ignored),
        ("識別力: bash rule 抜け誤実装",
         test_discrimination_missing_destructive_rule),
        ("識別力: body_modify rule 抜け誤実装",
         test_discrimination_missing_body_modify_flag),
        ("rules 順序 priority", test_rule_order_priority),
        ("path quirk: traversal → body_modify",
         test_path_traversal_to_body_modify_approve),
        ("path quirk: traversal → non body_modify",
         test_path_traversal_to_non_body_modify_auto),
        ("path quirk: absolute path",
         test_absolute_path_canonical_match),
        ("path quirk: backslash path",
         test_backslash_path_body_modify),
        ("path quirk: Windows case",
         test_windows_case_body_modify),
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
