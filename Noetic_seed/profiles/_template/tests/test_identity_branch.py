"""identity_branch.enforce_identity_branch の起動時ガード test。

PLAN §7-1 / §13-1 の 3 パターン + git 未初期化 fallback を検証。

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_identity_branch.py
  (pytest tests/test_identity_branch.py でも動く)
"""
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import identity_branch


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    assert cond, label


def _make_subprocess_mock(branch_exists: bool, current_branch: str):
    """rev-parse / show-ref に応答する subprocess.run mock factory。"""
    def fake_run(args, **kwargs):
        cmd = args[1] if len(args) > 1 else ""
        if cmd == "rev-parse":
            return subprocess.CompletedProcess(
                args=args, returncode=0,
                stdout=current_branch + "\n", stderr="",
            )
        if cmd == "show-ref":
            return subprocess.CompletedProcess(
                args=args, returncode=0 if branch_exists else 1,
                stdout="", stderr="",
            )
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr="",
        )
    return fake_run


def _setup_tmp_with_git():
    tmp = Path(tempfile.mkdtemp(prefix="noetic_idbranch_"))
    (tmp / ".git").mkdir()
    return tmp


def test_pattern_a_branch_not_exists():
    """パターン a: 期待ブランチ未作成 → exit(0) + 作成案内。"""
    tmp = _setup_tmp_with_git()
    mock = _make_subprocess_mock(branch_exists=False, current_branch="master")
    raised = False
    with patch.object(subprocess, "run", mock):
        try:
            identity_branch.enforce_identity_branch(
                profile_name="testprof", base_dir=tmp,
            )
        except SystemExit as e:
            raised = True
            _assert(e.code == 0, "パターン a: exit code = 0")
    _assert(raised, "パターン a: SystemExit が上がる")


def test_pattern_b_wrong_branch():
    """パターン b: 別 branch 上 → exit(0) + 切替案内。"""
    tmp = _setup_tmp_with_git()
    mock = _make_subprocess_mock(branch_exists=True, current_branch="master")
    raised = False
    with patch.object(subprocess, "run", mock):
        try:
            identity_branch.enforce_identity_branch(
                profile_name="testprof", base_dir=tmp,
            )
        except SystemExit as e:
            raised = True
            _assert(e.code == 0, "パターン b: exit code = 0")
    _assert(raised, "パターン b: SystemExit が上がる")


def test_pattern_c_correct_branch():
    """パターン c: 期待 branch 上 → exit せず起動続行 (None 返り)。"""
    tmp = _setup_tmp_with_git()
    mock = _make_subprocess_mock(
        branch_exists=True, current_branch="identity/testprof",
    )
    with patch.object(subprocess, "run", mock):
        result = identity_branch.enforce_identity_branch(
            profile_name="testprof", base_dir=tmp,
        )
    _assert(result is None, "パターン c: 起動続行 (None 返り)")


def test_no_git_repo_warning():
    """git 未初期化: warning + 起動続行 (CI / 一時テスト環境想定)。"""
    tmp = Path(tempfile.mkdtemp(prefix="noetic_idbranch_nogit_"))
    # tmp 内に .git は作らない
    result = identity_branch.enforce_identity_branch(
        profile_name="testprof", base_dir=tmp,
    )
    _assert(result is None, "git 未初期化: 起動続行 (None 返り)")


if __name__ == "__main__":
    print("=== test_identity_branch ===")
    test_pattern_a_branch_not_exists()
    test_pattern_b_wrong_branch()
    test_pattern_c_correct_branch()
    test_no_git_repo_warning()
    print("=== all green ===")
