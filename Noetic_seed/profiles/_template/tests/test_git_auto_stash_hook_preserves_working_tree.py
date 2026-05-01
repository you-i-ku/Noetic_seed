"""make_git_auto_stash_hook の Fix 5 (working tree 保持) 統合 test。

Issue 6 (2026-05-02 smoke12 で発見、ゆう gut 起源):
  iku が write_file core/x.py を呼ぶ瞬間に G-2 hook が
  `git stash push -u --message ... -- <profile>/` 実行 → working tree が
  HEAD (initial commit) に退避 → 在席 hotfix や進行中の改変が消失。

Fix 5 対応 (2026-05-02):
  `git stash push` 直後に `git stash apply stash@{0}` を追加して、
  履歴のみ保存しつつ working tree は保持する設計に変更。

本 test は実 git を使った統合 test。subprocess mock では stash の
状態管理を再現できないため、tmpdir に実 git repo を作って apply の
挙動を検証する (既存 test_git_auto_stash.py は mock 使用、別軸の test)。

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_git_auto_stash_hook_preserves_working_tree.py
"""
import shutil
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


def _git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=str(cwd),
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def _setup_test_repo():
    """tmp に git repo を作って initial commit を作成。

    initial commit に含めるファイル:
      core/initial.py     (hook 対象 path、書換え対象想定)
      other_tracked.py    (hook 非対象、在席 hotfix 模擬で改変される)

    Returns:
        Path: tmp = repo_root = profile_root (smoke12 と同型構成)
    """
    tmp = Path(tempfile.mkdtemp(prefix="noetic_fix5_"))
    _git("init", cwd=tmp)
    _git("config", "user.email", "test@example.com", cwd=tmp)
    _git("config", "user.name", "test", cwd=tmp)
    (tmp / "core").mkdir()
    (tmp / "core" / "initial.py").write_text("# initial\n", encoding="utf-8")
    (tmp / "other_tracked.py").write_text("ORIGINAL\n", encoding="utf-8")
    _git("add", ".", cwd=tmp)
    _git("commit", "-m", "initial commit", cwd=tmp)
    return tmp


def test_tracked_file_changes_preserved():
    """★ Issue 6 中核 — 別 tracked file の working tree 改変が hook 通過後も保持される。

    Fix 5 なし: stash push で working tree が HEAD に戻り、
                other_tracked.py が "ORIGINAL" に巻き戻る (Issue 6)
    Fix 5 あり: 直後 apply で working tree 復元、"HOTFIX_MODIFIED" のまま
    """
    print("== other_tracked.py (在席 hotfix 模擬) の改変が hook 通過後も保持 ==")
    tmp = _setup_test_repo()
    try:
        (tmp / "other_tracked.py").write_text("HOTFIX_MODIFIED\n", encoding="utf-8")

        hook = make_git_auto_stash_hook(tmp, tmp.name)
        r = hook("write_file", {"path": "core/initial.py", "content": "x"})

        actual = (tmp / "other_tracked.py").read_text(encoding="utf-8")
        return all([
            _assert(not r.denied, "allow (書換え続行)"),
            _assert(
                actual == "HOTFIX_MODIFIED\n",
                f"other_tracked.py 改変保持 (actual={actual!r})",
            ),
        ])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_untracked_file_preserved():
    """untracked file (iku の進行中改変) も apply で復元される。"""
    print("== untracked file (iku 進行中改変模擬) が hook 通過後も保持 ==")
    tmp = _setup_test_repo()
    try:
        (tmp / "in_progress.py").write_text("WIP\n", encoding="utf-8")

        hook = make_git_auto_stash_hook(tmp, tmp.name)
        r = hook("write_file", {"path": "core/initial.py", "content": "x"})

        path = tmp / "in_progress.py"
        exists = path.exists()
        content = path.read_text(encoding="utf-8") if exists else ""
        return all([
            _assert(not r.denied, "allow"),
            _assert(exists, "untracked file が working tree に残存"),
            _assert(content == "WIP\n", f"内容保持 (actual={content!r})"),
        ])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stash_history_retained_after_apply():
    """Fix 5 の apply は drop しない、stash 履歴は安全網として残る。"""
    print("== stash 履歴が apply 後も retain される (Step 6 が後で参照可能) ==")
    tmp = _setup_test_repo()
    try:
        (tmp / "other_tracked.py").write_text("CHANGED\n", encoding="utf-8")

        hook = make_git_auto_stash_hook(tmp, tmp.name)
        hook("write_file", {"path": "core/initial.py", "content": "x"})

        r = _git("stash", "list", cwd=tmp)
        return _assert(
            "iku-auto-" in r.stdout,
            f"iku-auto-* stash 残存 (list={r.stdout!r})",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_clean_working_tree_skips_apply():
    """working tree がクリーンなら stash 自体走らない (apply も skip)。

    `git stash push` が "No local changes to save" を返すケース。
    Fix 5 では `if "No local changes" not in ...` ガード経由で apply skip。
    """
    print("== クリーン tree では stash + apply 両方 skip ==")
    tmp = _setup_test_repo()
    try:
        # working tree は initial commit のまま、変更なし
        hook = make_git_auto_stash_hook(tmp, tmp.name)
        r = hook("write_file", {"path": "core/initial.py", "content": "x"})

        list_r = _git("stash", "list", cwd=tmp)
        return all([
            _assert(not r.denied, "allow"),
            _assert(
                list_r.stdout.strip() == "",
                f"stash list 空 (list={list_r.stdout!r})",
            ),
        ])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_non_body_modify_path_no_stash():
    """body_modify 対象外 path (sandbox/) は stash 走らない (既存挙動維持)。"""
    print("== sandbox/ path は対象外 (既存挙動の Fix 5 後回帰確認) ==")
    tmp = _setup_test_repo()
    try:
        (tmp / "other_tracked.py").write_text("CHANGED\n", encoding="utf-8")

        hook = make_git_auto_stash_hook(tmp, tmp.name)
        r = hook("write_file", {"path": "sandbox/memo.md", "content": "x"})

        list_r = _git("stash", "list", cwd=tmp)
        actual = (tmp / "other_tracked.py").read_text(encoding="utf-8")
        return all([
            _assert(not r.denied, "allow"),
            _assert(
                list_r.stdout.strip() == "",
                "sandbox/ では stash 走らない",
            ),
            _assert(
                actual == "CHANGED\n",
                "other_tracked.py の改変も触られない",
            ),
        ])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    groups = [
        ("★ tracked file 改変保持 (Issue 6 中核)",
            test_tracked_file_changes_preserved),
        ("untracked file 保持", test_untracked_file_preserved),
        ("stash 履歴 retain (Step 6 補完)",
            test_stash_history_retained_after_apply),
        ("clean tree → stash + apply skip",
            test_clean_working_tree_skips_apply),
        ("非対象 path は stash 走らない (回帰確認)",
            test_non_body_modify_path_no_stash),
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
