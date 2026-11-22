"""path_resolver helper の test (段階13 Phase 6.2 W3-min、PLAN §5-4)。

`path_resolve_for_policy` / `is_body_modify_path` の path quirk family
完全対応を検証 (Codex review 1-6 周目 + Codex rescue 提案を踏まえた
strict 仕様)。

検証ケース:
  - basic: prefix match (`core/`, `tools/`) + 完全一致 (`main.py`, `.mcp.json`)
  - non body_modify: `memory/`, `sandbox/`, etc.
  - traversal: `sandbox/../core/foo.py` → canonical `core/foo.py`
  - absolute path: workspace 内 absolute → relative 化
  - backslash: `core\\foo.py` → POSIX 化
  - Windows case insensitive: `Core/foo.py`, `.MCP.JSON`
  - whitespace: `" core/main.py "` → strip + canonical (Codex 6 周目 P1)
  - null byte / 制御 byte: `"core/main.py\x00"` → None reject (W3-min)
  - workspace 外: relative_to fail → None (raw fallback 削除、W3-min 核心)
  - workspace_root None: raw strip 返却 (test 後方互換)
  - 空 / None: None
  - dot 含み: `core/./main.py` → canonical 吸収

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_path_resolver.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.path_resolver import (
    is_body_modify_path, path_resolve_for_policy,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup() -> Path:
    return Path(tempfile.mkdtemp(prefix="noetic_path_resolver_test_")).resolve()


# ============================================================
# basic (is_body_modify_path)
# ============================================================

def test_basic_dir_prefix():
    print("== basic dir prefix (core/, tools/) ==")
    root = _setup()
    return all([
        _assert(is_body_modify_path("core/main.py", root) is True,
                "core/main.py → True"),
        _assert(is_body_modify_path("core/sub/x.py", root) is True,
                "core/sub/x.py → True"),
        _assert(is_body_modify_path("tools/sandbox.py", root) is True,
                "tools/sandbox.py → True"),
    ])


def test_basic_filenames_exact():
    print("== basic filenames 完全一致 (main.py, .mcp.json) ==")
    root = _setup()
    return all([
        _assert(is_body_modify_path("main.py", root) is True,
                "main.py → True"),
        _assert(is_body_modify_path(".mcp.json", root) is True,
                ".mcp.json → True"),
    ])


def test_non_body_modify():
    print("== 非 body_modify (memory/, sandbox/) ==")
    root = _setup()
    return all([
        _assert(is_body_modify_path("memory/x.jsonl", root) is False,
                "memory/x.jsonl → False"),
        _assert(is_body_modify_path("sandbox/note.md", root) is False,
                "sandbox/note.md → False"),
        _assert(is_body_modify_path("pref.json", root) is False,
                "pref.json → False"),
    ])


# ============================================================
# path quirk family (Codex 2/4/5/6 周目 root cause fix)
# ============================================================

def test_traversal_to_body_modify():
    print("== traversal → body_modify (canonical 化) ==")
    root = _setup()
    return all([
        _assert(is_body_modify_path("sandbox/../core/foo.py", root) is True,
                "sandbox/../core/foo.py → True"),
        _assert(is_body_modify_path("memory/../tools/x.py", root) is True,
                "memory/../tools/x.py → True"),
        _assert(is_body_modify_path("./core/foo.py", root) is True,
                "./core/foo.py → True"),
    ])


def test_traversal_to_non_body_modify():
    print("== traversal → 非 body_modify ==")
    root = _setup()
    return all([
        _assert(is_body_modify_path("core/../sandbox/x.md", root) is False,
                "core/../sandbox/x.md → False (canonical: sandbox/x.md)"),
        _assert(is_body_modify_path("tools/../memory/x.jsonl", root) is False,
                "tools/../memory/x.jsonl → False"),
    ])


def test_absolute_path_inside_workspace():
    print("== absolute path (workspace 内) → relative 化で判定 ==")
    root = _setup()
    abs_core = str(root / "core" / "main.py")
    abs_memory = str(root / "memory" / "x.jsonl")
    return all([
        _assert(is_body_modify_path(abs_core, root) is True,
                f"絶対 → True (canonical: core/main.py)"),
        _assert(is_body_modify_path(abs_memory, root) is False,
                f"絶対 → False (canonical: memory/x.jsonl)"),
    ])


def test_backslash_path():
    print("== backslash path (Windows) → POSIX 化で判定 ==")
    root = _setup()
    return all([
        _assert(is_body_modify_path("core\\foo.py", root) is True,
                "core\\foo.py → True"),
        _assert(is_body_modify_path("tools\\sub\\x.py", root) is True,
                "tools\\sub\\x.py → True"),
        _assert(is_body_modify_path("sandbox\\note.md", root) is False,
                "sandbox\\note.md → False"),
    ])


def test_windows_case_insensitive():
    print("== Windows case → 大小無視 (Windows のみ) ==")
    if os.name != "nt":
        print("  [SKIP] non-Windows env")
        return True
    root = _setup()
    return all([
        _assert(is_body_modify_path("Core/foo.py", root) is True,
                "Core/foo.py → True"),
        _assert(is_body_modify_path("TOOLS/x.py", root) is True,
                "TOOLS/x.py → True"),
        _assert(is_body_modify_path("Main.py", root) is True,
                "Main.py → True"),
        _assert(is_body_modify_path(".MCP.JSON", root) is True,
                ".MCP.JSON → True"),
    ])


def test_whitespace_strip():
    print("== whitespace strip (Codex 6 周目 P1 root cause fix) ==")
    root = _setup()
    return all([
        _assert(is_body_modify_path(" core/main.py ", root) is True,
                "前後 space → True (strip 後 canonical)"),
        _assert(is_body_modify_path("\tcore/main.py\n", root) is True,
                "tab/LF 含み → None reject? いや whitespace strip 範囲、要 verify"),
        _assert(is_body_modify_path("  main.py  ", root) is True,
                "  main.py   → True"),
    ])


def test_dot_path():
    print("== `.` 含み path → canonical で吸収 ==")
    root = _setup()
    return all([
        _assert(is_body_modify_path("core/./main.py", root) is True,
                "core/./main.py → True"),
        _assert(is_body_modify_path("./tools/x.py", root) is True,
                "./tools/x.py → True"),
    ])


# ============================================================
# strict 仕様 (W3-min 核心: raw fallback 削除、None 返却)
# ============================================================

def test_workspace_outside_returns_none():
    print("== workspace 外 → None (raw fallback 削除、W3-min) ==")
    root = _setup()
    other = _setup()
    abs_outside = str(other / "core" / "x.py")
    rel = path_resolve_for_policy(abs_outside, root)
    return all([
        _assert(rel is None, f"workspace 外 → None (raw 返却なし)"),
        _assert(is_body_modify_path(abs_outside, root) is False,
                "workspace 外 body_modify → False"),
    ])


def test_null_byte_rejected():
    print("== null byte / 制御 byte → None reject (W3-min) ==")
    root = _setup()
    return all([
        _assert(path_resolve_for_policy("core/main.py\x00", root) is None,
                "null byte 含み → None"),
        _assert(path_resolve_for_policy("core\x01main.py", root) is None,
                "制御 byte 含み → None"),
        _assert(is_body_modify_path("core/main.py\x00.txt", root) is False,
                "null byte body_modify → False (raw fallback で True にならない)"),
    ])


def test_empty_path_returns_none():
    print("== 空 path / None / whitespace のみ → None ==")
    root = _setup()
    return all([
        _assert(path_resolve_for_policy("", root) is None, "空 path → None"),
        _assert(path_resolve_for_policy("   ", root) is None,
                "whitespace のみ → None"),
        _assert(is_body_modify_path("", root) is False, "空 body_modify → False"),
    ])


# ============================================================
# 後方互換 (workspace_root None で raw strip 返却)
# ============================================================

def test_workspace_root_none_raw_fallback():
    print("== workspace_root None → raw strip 返却 (test 後方互換) ==")
    return all([
        _assert(path_resolve_for_policy("core/main.py", None) == "core/main.py",
                "raw → core/main.py"),
        _assert(path_resolve_for_policy(" core/main.py ", None) == "core/main.py",
                "前後 space → strip 済 raw"),
        _assert(is_body_modify_path("core/main.py", None) is True,
                "raw core/main.py body_modify → True (startswith hit)"),
        _assert(is_body_modify_path("memory/x.jsonl", None) is False,
                "raw memory/x.jsonl body_modify → False"),
    ])


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    groups = [
        ("basic dir prefix", test_basic_dir_prefix),
        ("basic filenames exact", test_basic_filenames_exact),
        ("non body_modify", test_non_body_modify),
        ("traversal → body_modify", test_traversal_to_body_modify),
        ("traversal → non body_modify", test_traversal_to_non_body_modify),
        ("absolute path inside workspace", test_absolute_path_inside_workspace),
        ("backslash path", test_backslash_path),
        ("Windows case insensitive", test_windows_case_insensitive),
        ("whitespace strip (Codex 6周目 P1)", test_whitespace_strip),
        ("dot path canonical", test_dot_path),
        ("workspace outside → None (W3-min)",
         test_workspace_outside_returns_none),
        ("null byte rejected", test_null_byte_rejected),
        ("empty path → None", test_empty_path_returns_none),
        ("workspace_root None raw fallback",
         test_workspace_root_none_raw_fallback),
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
