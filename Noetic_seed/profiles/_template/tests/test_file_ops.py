"""file_ops tests (claw-code 準拠版)。

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_file_ops.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.registry import ToolRegistry
from core.runtime.tools import file_ops


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# テスト用の一時 workspace_root
TMPDIR: Path = None


def _setup() -> Path:
    global TMPDIR
    TMPDIR = Path(tempfile.mkdtemp(prefix="clawcode_fileops_"))
    (TMPDIR / "hello.txt").write_text("Hello\nWorld\nThird\n", encoding="utf-8")
    (TMPDIR / "code.py").write_text(
        "x = 1\ny = 2\n# comment\nprint(x + y)\n", encoding="utf-8"
    )
    (TMPDIR / "sub").mkdir(exist_ok=True)
    (TMPDIR / "sub" / "nested.txt").write_text("nested\n", encoding="utf-8")
    (TMPDIR / "dupes.txt").write_text("aa bb aa cc aa\n", encoding="utf-8")
    # binary
    (TMPDIR / "binary.bin").write_bytes(b"\x00\x01\x02binary\x00")
    return TMPDIR


def _teardown():
    if TMPDIR and TMPDIR.exists():
        shutil.rmtree(TMPDIR)


def _get_tools():
    reg = ToolRegistry()
    file_ops.register(reg, TMPDIR)
    return reg


# ============================================================
# read_file
# ============================================================

def test_read_basic():
    print("== read_file: 基本 ==")
    reg = _get_tools()
    out = reg.execute("read_file", {"path": "hello.txt"})
    return all([
        _assert("Hello" in out, "content"),
        _assert("World" in out, "全行"),
        _assert("lines 1-3/3" in out, "header"),
    ])


def test_read_offset_limit():
    print("== read_file: offset/limit ==")
    reg = _get_tools()
    out = reg.execute("read_file", {"path": "code.py",
                                    "offset": 1, "limit": 2})
    return all([
        _assert("y = 2" in out, "offset=1"),
        _assert("# comment" in out, "limit=2"),
        _assert("print" not in out, "limit 超過"),
    ])


def test_read_path_traversal():
    print("== read_file: path traversal block ==")
    reg = _get_tools()
    out = reg.execute("read_file", {"path": "../../../etc/passwd"})
    return _assert("outside workspace" in out, "outside ブロック")


def test_read_not_found():
    print("== read_file: not found ==")
    reg = _get_tools()
    out = reg.execute("read_file", {"path": "missing.txt"})
    return _assert("not found" in out, "not found エラー")


def test_read_binary():
    print("== read_file: binary 拒否 ==")
    reg = _get_tools()
    out = reg.execute("read_file", {"path": "binary.bin"})
    return _assert("binary" in out.lower(), "binary エラー")


def test_read_empty_path():
    print("== read_file: 空 path ==")
    reg = _get_tools()
    out = reg.execute("read_file", {"path": ""})
    return _assert("required" in out.lower(), "empty path エラー")


# ============================================================
# write_file
# ============================================================

def test_write_basic():
    print("== write_file: 基本 ==")
    reg = _get_tools()
    out = reg.execute("write_file", {"path": "new.txt", "content": "new"})
    return all([
        _assert("Wrote" in out, "成功"),
        _assert((TMPDIR / "new.txt").read_text(encoding="utf-8") == "new",
                "実ファイル"),
    ])


def test_write_create_subdir():
    print("== write_file: サブディレクトリ作成 ==")
    reg = _get_tools()
    out = reg.execute("write_file", {"path": "deep/a/b.txt", "content": "x"})
    return all([
        _assert("Wrote" in out, "成功"),
        _assert((TMPDIR / "deep" / "a" / "b.txt").exists(),
                "ディレクトリも作成"),
    ])


def test_write_path_traversal():
    print("== write_file: path traversal block ==")
    reg = _get_tools()
    out = reg.execute("write_file", {"path": "../outside.txt", "content": "x"})
    return _assert("outside workspace" in out, "outside ブロック")


# ============================================================
# edit_file
# ============================================================

def test_edit_single():
    print("== edit_file: 単一置換 ==")
    (TMPDIR / "edit1.txt").write_text("foo bar baz\n", encoding="utf-8")
    reg = _get_tools()
    out = reg.execute("edit_file", {
        "path": "edit1.txt", "old_string": "bar", "new_string": "BAR",
    })
    return all([
        _assert("Edited" in out, "成功"),
        _assert((TMPDIR / "edit1.txt").read_text() == "foo BAR baz\n",
                "実ファイル置換"),
    ])


def test_edit_multi_without_all():
    print("== edit_file: 複数マッチ + replace_all なし → エラー ==")
    reg = _get_tools()
    out = reg.execute("edit_file", {
        "path": "dupes.txt", "old_string": "aa", "new_string": "XX",
    })
    content = (TMPDIR / "dupes.txt").read_text()
    return all([
        _assert("matches" in out and "replace_all" in out, "エラー"),
        _assert(content == "aa bb aa cc aa\n", "変更なし"),
    ])


def test_edit_replace_all():
    print("== edit_file: replace_all ==")
    (TMPDIR / "edit2.txt").write_text("aa bb aa cc aa\n", encoding="utf-8")
    reg = _get_tools()
    out = reg.execute("edit_file", {
        "path": "edit2.txt", "old_string": "aa", "new_string": "X",
        "replace_all": True,
    })
    content = (TMPDIR / "edit2.txt").read_text()
    return all([
        _assert("3 replacement" in out, "3 箇所"),
        _assert(content == "X bb X cc X\n", "全件置換"),
    ])


def test_edit_not_found():
    print("== edit_file: old_string not found ==")
    reg = _get_tools()
    out = reg.execute("edit_file", {
        "path": "hello.txt", "old_string": "ZZZ", "new_string": "x",
    })
    return _assert("not found" in out, "エラー")


# ============================================================
# glob_search
# ============================================================

def test_glob_basic():
    print("== glob_search: 基本 ==")
    reg = _get_tools()
    out = reg.execute("glob_search", {"pattern": "*.txt"})
    return all([
        _assert("hello.txt" in out, "hello.txt"),
        _assert("Found" in out, "header"),
    ])


def test_glob_recursive():
    print("== glob_search: recursive ==")
    reg = _get_tools()
    out = reg.execute("glob_search", {"pattern": "**/*.txt"})
    return all([
        _assert("hello.txt" in out, "top"),
        _assert("sub/nested.txt" in out, "sub"),
    ])


def test_glob_no_match():
    print("== glob_search: no match ==")
    reg = _get_tools()
    out = reg.execute("glob_search", {"pattern": "*.nonexistent"})
    return _assert("No matches" in out, "no match")


# ============================================================
# grep_search
# ============================================================

def test_grep_basic():
    print("== grep_search: 基本 ==")
    reg = _get_tools()
    out = reg.execute("grep_search", {"pattern": "World"})
    return all([
        _assert("hello.txt" in out, "ファイル名"),
        _assert("World" in out, "マッチ"),
    ])


def test_grep_regex():
    print("== grep_search: regex ==")
    reg = _get_tools()
    out = reg.execute("grep_search", {"pattern": r"^\w = \d+",
                                      "glob": "*.py"})
    return all([
        _assert("x = 1" in out, "x = 1"),
        _assert("y = 2" in out, "y = 2"),
        _assert("comment" not in out, "comment 除外"),
    ])


def test_grep_ignore_case():
    print("== grep_search: -i ==")
    reg = _get_tools()
    out = reg.execute("grep_search", {"pattern": "HELLO", "-i": True})
    return _assert("Hello" in out, "case insensitive")


def test_grep_head_limit():
    print("== grep_search: head_limit ==")
    (TMPDIR / "many.txt").write_text(
        "\n".join(f"line {i}" for i in range(10)), encoding="utf-8"
    )
    reg = _get_tools()
    out = reg.execute("grep_search", {"pattern": "line", "head_limit": 3})
    # header 行を除いてマッチ数をカウント
    matches = [l for l in out.splitlines()
               if not l.startswith("Found") and ":" in l]
    return _assert(len(matches) == 3, f"head_limit=3 ({len(matches)}件)")


def test_grep_binary_skip():
    print("== grep_search: binary skip ==")
    reg = _get_tools()
    out = reg.execute("grep_search", {"pattern": "binary"})
    # binary.bin には "binary" 文字列があるが、binary file は読まない
    return _assert("binary.bin" not in out, "binary.bin スキップ")


# ============================================================
# register
# ============================================================

def test_register_count():
    print("== register: 5 tool ==")
    reg = ToolRegistry()
    file_ops.register(reg, TMPDIR)
    expected = {"read_file", "write_file", "edit_file",
                "glob_search", "grep_search"}
    return _assert(expected.issubset(set(reg.all_names())),
                   "5 tool 全登録")


def main():
    _setup()
    try:
        tests = [
            test_read_basic, test_read_offset_limit,
            test_read_path_traversal, test_read_not_found,
            test_read_binary, test_read_empty_path,
            test_write_basic, test_write_create_subdir,
            test_write_path_traversal,
            test_edit_single, test_edit_multi_without_all,
            test_edit_replace_all, test_edit_not_found,
            test_glob_basic, test_glob_recursive, test_glob_no_match,
            test_grep_basic, test_grep_regex, test_grep_ignore_case,
            test_grep_head_limit, test_grep_binary_skip,
            test_register_count,
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
