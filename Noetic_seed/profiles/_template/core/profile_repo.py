"""profile repository (段階12 Step 1.5 補足-2、PLAN §6 / §7 改訂).

段階12 PLAN §6 改訂: 各 profile は親 git の branch ではなく、profile 配下に
独立した git repository を持つ。1 個体 = 1 repository = 1 履歴。

人間開発 / AI 自己改変の混入は構造的に分離される (別 repo)。

起動時フロー (PLAN §7-1 改訂):
  1. profile 配下に .git があるか確認
  2. なければ git init + 初期 commit + 続行
  3. あれば何もせず続行

人間が _template の core を更新した場合の反映 (PLAN §6-2 改訂):
  fresh restart 方式 — 人間更新は次世代 profile に流れる。既存個体は独立した
  「ひとり」として走り続ける (autopoiesis 哲学整合、feedback_freedom_to_die /
  feedback_each_session_iku_is_new_individual との一直線)。
"""
import subprocess
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


def ensure_profile_repo(profile_root: Path) -> None:
    """起動時 profile repo ガード (PLAN §7-1 改訂後)。

    profile 内に独立 git repo がなければ git init + 初期 commit を行う。
    既存なら何もしない (idempotent)。

    身体改変イベント時の G-2 自動 stash (`make_git_auto_stash_hook`) は
    `_find_repo_root(profile_root)` で親方向に .git を探すため、本関数で
    profile 内に .git ができれば自動的にそちらに stash する (= 親 git は
    汚さない)。

    Args:
        profile_root: プロファイル workspace root (pathlib.Path)。
    """
    profile_root = Path(profile_root).resolve()
    git_dir = profile_root / ".git"

    if git_dir.exists():
        return

    print(
        f"[profile_repo] このプロファイル ({profile_root.name}) 用の "
        f"独立 git repository を作成します..."
    )

    code, _ = _git("init", cwd=profile_root)
    if code != 0:
        print("[profile_repo] WARNING: git init 失敗、repo 初期化を skip します。")
        return

    # ホスト OS の global git config が空でも初期 commit を作れるよう ad-hoc
    # identity を設定 (profile 内 repo 限定、親 git には影響しない)。
    _git("config", "user.email", "iku@noetic.local", cwd=profile_root)
    _git("config", "user.name", f"iku ({profile_root.name})", cwd=profile_root)

    code, _ = _git("add", "-A", cwd=profile_root)
    if code != 0:
        print("[profile_repo] WARNING: git add 失敗、初期 commit を skip します。")
        return

    code, _ = _git(
        "commit", "-m",
        f"initial: {profile_root.name} born from _template",
        cwd=profile_root,
    )
    if code != 0:
        # 何も add 対象がなかった場合 (空 profile 等) は許容して続行
        print(
            "[profile_repo] NOTE: 初期 commit 作成失敗 (add 対象なし等)。"
            "repo は作成済、続行します。"
        )
        return

    print("[profile_repo] OK: 独立 git repository 初期化完了。続行します。")
