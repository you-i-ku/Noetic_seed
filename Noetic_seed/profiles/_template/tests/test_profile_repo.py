"""profile_repo.ensure_profile_repo の起動時 git init test (段階12 PLAN §6 / §7 改訂)。

profile は親 git の branch ではなく、profile 配下に独立した git repository を
持つ。1 個体 = 1 repo = 1 履歴。本 test では:
  - profile 内に .git なければ自動 init + 初期 commit が走ること
  - 既存 .git は触らないこと (idempotent)
  - git CLI 不在環境では warning + skip (CI 想定)

を確認する。

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_profile_repo.py
  (pytest tests/test_profile_repo.py でも動く)
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import profile_repo


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    assert cond, label


def _git_available() -> bool:
    try:
        result = subprocess.run(
            ["git", "--version"], capture_output=True, text=True,
        )
        return result.returncode == 0
    except (FileNotFoundError, OSError):
        return False


def test_no_repo_creates_init_with_initial_commit():
    """profile 内に .git なければ自動 init + 初期 commit する。"""
    if not _git_available():
        print("  [SKIP] git CLI 不在、test_no_repo_creates_init_with_initial_commit を skip")
        return

    tmp = Path(tempfile.mkdtemp(prefix="noetic_profile_repo_"))
    try:
        # 何か add 対象を作る (空 profile だと initial commit 作らない仕様)
        (tmp / "main.py").write_text('print("hi")', encoding="utf-8")
        (tmp / "config.json").write_text("{}", encoding="utf-8")

        _assert(not (tmp / ".git").exists(), "前提: .git なし")

        profile_repo.ensure_profile_repo(tmp)

        _assert((tmp / ".git").exists(), "ensure 後: .git 作成済")

        # 初期 commit が存在
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=str(tmp),
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        _assert(
            result.returncode == 0 and bool(result.stdout.strip()),
            f"初期 commit が存在 (log: {result.stdout.strip()[:80]})",
        )
        _assert(
            "initial" in result.stdout.lower() or tmp.name in result.stdout,
            "初期 commit message に 'initial' or profile 名が含まれる",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_existing_repo_is_noop():
    """profile 内に .git 既存なら何もしない (idempotent)。"""
    tmp = Path(tempfile.mkdtemp(prefix="noetic_profile_repo_existing_"))
    try:
        (tmp / ".git").mkdir()
        marker = tmp / ".git" / "marker"
        marker.write_text("preserved", encoding="utf-8")

        profile_repo.ensure_profile_repo(tmp)

        _assert(
            marker.exists() and marker.read_text(encoding="utf-8") == "preserved",
            "既存 .git は触らない (marker 残存)",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_idempotent_second_call_no_change():
    """同じ profile に 2 回 ensure 呼んでも 1 回目と同じ状態 (idempotent)。"""
    if not _git_available():
        print("  [SKIP] git CLI 不在、test_idempotent_second_call_no_change を skip")
        return

    tmp = Path(tempfile.mkdtemp(prefix="noetic_profile_repo_idempotent_"))
    try:
        (tmp / "main.py").write_text("x", encoding="utf-8")

        profile_repo.ensure_profile_repo(tmp)
        first_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(tmp), capture_output=True, text=True,
        ).stdout.strip()

        profile_repo.ensure_profile_repo(tmp)
        second_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(tmp), capture_output=True, text=True,
        ).stdout.strip()

        _assert(
            first_commit == second_commit and first_commit != "",
            f"2 回目の ensure で HEAD 変化なし ({first_commit[:8]})",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=== test_profile_repo ===")
    test_no_repo_creates_init_with_initial_commit()
    test_existing_repo_is_noop()
    test_idempotent_second_call_no_change()
    print("=== all green ===")
