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


def _git(*args: str, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """git コマンドを実行して (returncode, stdout, stderr) を返す。

    2026-04-29 hotfix: stderr を戻り値に追加 (旧版は捨ててて真因不明だった)。
    失敗時は (-1, '', error_msg) を返す (FileNotFoundError 等の外部 OS error)。
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except (FileNotFoundError, OSError) as e:
        return -1, "", f"OSError: {e}"


def _has_any_commit(profile_root: Path) -> bool:
    """profile 内 repo に commit が 1 個以上あるかチェック (HEAD 解決可)。"""
    code, out, _ = _git("rev-parse", "HEAD", cwd=profile_root)
    return code == 0 and bool(out)


def ensure_profile_repo(profile_root: Path) -> None:
    """起動時 profile repo ガード (PLAN §7-1 改訂後)。

    profile 内 git repo の正常な初期状態を保証する 3 状態:
      ① .git なし                → git init + 初期 commit (新規個体誕生)
      ② .git あり、commit なし   → 半完成状態 → repair (add + commit retry)
      ③ .git あり、commit あり   → noop (idempotent)

    ② は段階12 補足-3 改までの実装 (warning + return) で「.git だけ残った
    半完成 profile」が発生していたケースの修復経路 (2026-04-29 hotfix)。

    身体改変イベント時の G-2 自動 stash (`make_git_auto_stash_hook`) は
    `_find_repo_root(profile_root)` で親方向に .git を探すため、本関数で
    profile 内に .git ができれば自動的にそちらに stash する (= 親 git は
    汚さない)。

    Args:
        profile_root: プロファイル workspace root (pathlib.Path)。
    """
    profile_root = Path(profile_root).resolve()
    git_dir = profile_root / ".git"

    needs_init = not git_dir.exists()

    if needs_init:
        print(
            f"[profile_repo] このプロファイル ({profile_root.name}) 用の "
            f"独立 git repository を作成します..."
        )
        code, _, err = _git("init", cwd=profile_root)
        if code != 0:
            print("[profile_repo] WARNING: git init 失敗、repo 初期化を skip します。")
            if err:
                print(f"  stderr: {err[:300]}")
            return
    else:
        if _has_any_commit(profile_root):
            return
        # .git 存在するが initial commit 0 個 = 半完成、repair に進む
        print(
            f"[profile_repo] このプロファイル ({profile_root.name}) は repo "
            f"初期化済だが initial commit がありません。repair します..."
        )

    # ad-hoc identity (空 global config 環境向け)。profile 内 repo 限定、
    # 親 git には影響しない。
    _git("config", "user.email", "iku@noetic.local", cwd=profile_root)
    _git("config", "user.name", f"iku ({profile_root.name})", cwd=profile_root)

    # Windows MAX_PATH (260 chars) 制限の自動回避 (2026-04-29 hotfix)。
    # huggingface-hub 等の deeply nested cache が原因で git add が失敗する
    # 可能性への防御。core.longpaths=true で 32767 chars まで拡張、害なし。
    _git("config", "core.longpaths", "true", cwd=profile_root)

    code, _, err = _git("add", "-A", cwd=profile_root)
    if code != 0:
        print("[profile_repo] WARNING: git add 失敗、初期 commit を skip します。")
        if err:
            # 真因観察のため stderr の頭 500 char を表示 (旧版は捨ててて真因不明)
            print(f"  stderr (head 500): {err[:500]}")
        return

    code, _, err = _git(
        "commit", "-m",
        f"initial: {profile_root.name} born from _template",
        cwd=profile_root,
    )
    if code != 0:
        # 何も add 対象がなかった場合 (空 profile 等) は許容して続行
        msg = err or "(no stderr)"
        print(
            "[profile_repo] NOTE: 初期 commit 作成失敗。repo は作成済、続行します。\n"
            f"  stderr: {msg[:300]}"
        )
        return

    print("[profile_repo] OK: 独立 git repository 初期化完了。続行します。")
