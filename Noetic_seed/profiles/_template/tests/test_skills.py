"""Skills Registry + Loader tests."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.skills import (
    Skill,
    SkillRegistry,
    parse_skill_file,
    default_skill_dirs,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


TMPDIR: Path = None


def _setup():
    global TMPDIR
    TMPDIR = Path(tempfile.mkdtemp(prefix="clawcode_skills_"))
    # top level .md
    (TMPDIR / "hello.md").write_text(
        '---\n'
        'name: hello\n'
        'description: greet the user\n'
        'triggers: ["hi", "hello"]\n'
        'allowed_tools: ["read_file"]\n'
        '---\n'
        '# Hello\n\nSay hi to the user.',
        encoding="utf-8",
    )
    # subdir SKILL.md
    (TMPDIR / "commit").mkdir()
    (TMPDIR / "commit" / "SKILL.md").write_text(
        '---\n'
        'name: commit-helper\n'
        'description: git commit helper\n'
        '---\n'
        'Commit the staged changes.',
        encoding="utf-8",
    )
    # malformed (no frontmatter)
    (TMPDIR / "bare.md").write_text(
        "Just body, no frontmatter.\nname is inferred from filename.",
        encoding="utf-8",
    )


def _teardown():
    if TMPDIR and TMPDIR.exists():
        shutil.rmtree(TMPDIR)


def test_parse_full_frontmatter():
    print("== parse_skill_file: 完全な frontmatter ==")
    sk = parse_skill_file(TMPDIR / "hello.md")
    return all([
        _assert(sk is not None, "parse 成功"),
        _assert(sk.name == "hello", "name"),
        _assert("greet" in sk.description, "description"),
        _assert(sk.triggers == ["hi", "hello"], "triggers list"),
        _assert(sk.allowed_tools == ["read_file"], "allowed_tools"),
        _assert("Say hi" in sk.body, "body"),
    ])


def test_parse_subdir_skill():
    print("== parse_skill_file: subdir/SKILL.md ==")
    sk = parse_skill_file(TMPDIR / "commit" / "SKILL.md")
    return all([
        _assert(sk is not None, "parse"),
        _assert(sk.name == "commit-helper", "name from frontmatter"),
    ])


def test_parse_bare_markdown():
    print("== parse_skill_file: frontmatter なし ==")
    sk = parse_skill_file(TMPDIR / "bare.md")
    return all([
        _assert(sk is not None, "parse"),
        _assert(sk.name == "bare", "name はファイル名から"),
        _assert("Just body" in sk.body, "body"),
        _assert(sk.triggers == [], "triggers 空"),
    ])


def test_parse_nonexistent():
    print("== parse_skill_file: 存在しないファイル ==")
    sk = parse_skill_file(TMPDIR / "no_such.md")
    return _assert(sk is None, "None")


def test_registry_register_get():
    print("== SkillRegistry: register/get ==")
    r = SkillRegistry()
    s = Skill(name="a", description="A")
    r.register(s)
    return all([
        _assert(r.has("a"), "has"),
        _assert(r.get("a") is s, "get"),
        _assert(r.get("missing") is None, "missing"),
    ])


def test_registry_load_dir():
    print("== SkillRegistry.load: ディレクトリ読込 ==")
    r = SkillRegistry()
    n = r.load([str(TMPDIR)])
    names = {s.name for s in r.list_all()}
    return all([
        _assert(n >= 3, f"3 以上ロード (got {n})"),
        _assert("hello" in names, "hello"),
        _assert("commit-helper" in names, "commit-helper (subdir)"),
        _assert("bare" in names, "bare (frontmatter なし)"),
    ])


def test_registry_load_no_overwrite():
    print("== SkillRegistry.load: 同名は上書きしない ==")
    r = SkillRegistry()
    r.load([str(TMPDIR)])
    # 同じディレクトリを再ロードしても増えない
    n2 = r.load([str(TMPDIR)])
    return _assert(n2 == 0, f"2 回目は 0 件 (got {n2})")


def test_registry_find_by_trigger():
    print("== SkillRegistry.find_by_trigger ==")
    r = SkillRegistry()
    r.load([str(TMPDIR)])
    hits = r.find_by_trigger("hello")
    return all([
        _assert(len(hits) >= 1, "hello ヒット"),
        _assert(any(s.name == "hello" for s in hits), "hello skill"),
    ])


def test_registry_find_by_description():
    print("== SkillRegistry.find_by_trigger: description マッチ ==")
    r = SkillRegistry()
    r.load([str(TMPDIR)])
    hits = r.find_by_trigger("commit")
    return _assert(any(s.name == "commit-helper" for s in hits),
                   "description マッチで commit-helper ヒット")


def test_default_skill_dirs():
    print("== default_skill_dirs ==")
    dirs = default_skill_dirs(workspace_root=TMPDIR)
    return all([
        _assert(any(".claw" in d for d in dirs), ".claw/skills 含む"),
        _assert(any(".claude" in d for d in dirs), ".claude/skills 含む"),
    ])


def test_nonexistent_dir_safe():
    print("== SkillRegistry.load: 存在しないディレクトリでも落ちない ==")
    r = SkillRegistry()
    n = r.load(["/does/not/exist", str(TMPDIR), ""])
    return _assert(n > 0, "有効なディレクトリは読める")


def main():
    _setup()
    try:
        tests = [
            test_parse_full_frontmatter,
            test_parse_subdir_skill,
            test_parse_bare_markdown,
            test_parse_nonexistent,
            test_registry_register_get,
            test_registry_load_dir,
            test_registry_load_no_overwrite,
            test_registry_find_by_trigger,
            test_registry_find_by_description,
            test_default_skill_dirs,
            test_nonexistent_dir_safe,
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
