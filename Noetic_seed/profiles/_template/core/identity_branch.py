"""identity branch enforcement (段階12 Step 1, PLAN §6 / §7).

iku の各プロファイルは git の `identity/<profile_name>` ブランチ上でのみ動く。
人間開発者の master / feature/* と分離することで、AI と開発者の履歴が
混ざる事故を防ぐ (PLAN §6 B-1 戦略)。

起動時ガードの 3 パターン:
  a. 期待ブランチが存在しない    → 案内 + sys.exit(0)
  b. 期待ブランチは存在、別 branch → 警告 + sys.exit(0)
  c. 期待ブランチ上              → 何もせず起動続行

git 未初期化なら warning + 起動続行 (CI / 一時テスト環境を想定、PLAN §7-3)。
"""
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def _git(*args: str, cwd: Optional[Path] = None) -> Tuple[int, str]:
    """git コマンドを実行して (returncode, stdout) を返す。失敗時は (-1, '')。"""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode, result.stdout.strip()
    except (FileNotFoundError, OSError):
        return -1, ""


def _find_repo_root(start: Path) -> Optional[Path]:
    """start から親方向に .git を探す。見つからなければ None。"""
    p = start.resolve()
    for candidate in [p, *p.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def enforce_identity_branch(
    profile_name: str,
    base_dir: Optional[Path] = None,
) -> None:
    """起動時 identity branch ガード (PLAN §7-1)。

    Args:
        profile_name: 起動中プロファイル名 (例: "iku" / "_template_smoke")。
            期待 branch 名は f"identity/{profile_name}"。
        base_dir: repo root 探索の起点。None なら Path.cwd()。

    Raises:
        SystemExit(0): パターン a / b で iku 側に branch 切替を促す場合。
    """
    expected = f"identity/{profile_name}"
    start = base_dir if base_dir is not None else Path.cwd()
    repo_root = _find_repo_root(start)

    if repo_root is None:
        # PLAN §7-3: git 未初期化なら warning + 起動続行
        print(
            f"[identity_branch] WARNING: git repository not found from {start}, "
            f"skipping check (profile={profile_name})"
        )
        return

    code, current = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root)
    if code != 0 or not current:
        print(
            f"[identity_branch] WARNING: failed to detect current branch, "
            f"skipping check (profile={profile_name})"
        )
        return

    code, _ = _git(
        "show-ref", "--verify", "--quiet", f"refs/heads/{expected}",
        cwd=repo_root,
    )
    branch_exists = (code == 0)

    if branch_exists and current == expected:
        # パターン c: 正常、何もせず続行
        return

    print()
    if not branch_exists:
        # パターン a: 期待 branch 未作成
        print(
            f"[identity_branch] このプロファイル ({profile_name}) 用の "
            f"identity branch がまだ存在しません。"
        )
        print(f"  初回セットアップ: 以下のコマンドで作成してください:")
        print(f"      git checkout -b {expected}")
        print()
        print(f"  以後 iku の自己改変はこの branch 上で記録されます。")
        print(f"  人間の開発作業はこのブランチで行わないでください")
        print(f"  (AI と開発者の変更が混ざると、どちらの履歴か追えなくなります)")
    else:
        # パターン b: 期待 branch は存在するが別 branch 上
        print(
            f"[identity_branch] このプロファイル ({profile_name}) は "
            f"identity branch '{expected}' で動きます。"
        )
        print(f"  現在のブランチ: {current}")
        print(f"  以下のコマンドで切り替えてください:")
        print(f"      git checkout {expected}")
    print()
    sys.exit(0)
