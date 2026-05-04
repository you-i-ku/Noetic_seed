"""Path resolver — workspace path 正規化の単一 source of truth (W3-min)。

段階13 Phase 6.2 (PLAN §5-4 W3-min 案、2026-05-04) で確立。Codex review
6 周連続 BLOCKER detection (path 表現 quirk family 4 連発) への root cause
fix として、Codex rescue の設計提案 literal を採用。

## (W3-min) 設計方針

Codex review 1-6 周目 + Codex rescue (codex:rescue agent) で identified された
root cause:

> 「approval policy が見ている path 表現と、実 file 操作が canonical 化する
> path 表現の差で bypass」が連発。各周で姑息 fix (case insensitive / strip /
> traversal canonical / 等) を続けるパターンを構造的に止めるには:
>   1. quirk denylist じゃなく **canonical resolver 1 個** に通す
>   2. resolve 失敗時の raw fallback **そのものが穴**、消して strict に None 返却
>   3. 4 サイト (approval_rules + git_auto_stash + pending_body_modify +
>      file_access_guard) で **同じ resolver を共有**、分類完全一致

実行基盤 (claw-code) との疎結合を保つため、`file_ops.py` (claw native) は
触らず Noetic 固有 hook 層 4 サイトのみ統一。将来 claw-code を別実行基盤に
乗り換える時、Noetic 側 path 評価 logic は無傷で残せる
(CLAUDE.md「インターフェースは抽象化、中身だけ差し替え可能」整合)。

## API

### `path_resolve_for_policy(path, workspace_root) -> Optional[str]`

input path を canonical な workspace 相対 POSIX path に変換。失敗時 None。
**raw fallback はしない** (Codex 提案の核心、現行 P1 BLOCKER の最大の穴)。

正規化契約:
1. `.strip()` で前後 whitespace 除去 (Codex 6 周目 P1)
2. null byte / 制御 byte (0x00-0x1f / 0x7f) を含む path → None で reject
3. `~` / `$VAR` / `%PATH%` 等の expansion は **しない** (raw のまま resolve)
4. `(workspace_root / path).resolve()` で canonical 化
   (traversal `..`、絶対 path、backslash、symlink 等を一括吸収)
5. `relative_to(workspace_root)` で workspace 相対化、外なら None
6. POSIX path 化 (`as_posix()`)
7. Windows なら lowercase 化 (case insensitive 環境対応)
8. `workspace_root` が None なら raw を `.strip()` して返す (test 後方互換)

### `is_body_modify_path(path, workspace_root, dir_prefixes, filenames) -> bool`

上の resolver を呼んで、身体改変領域 (core/* / tools/* / main.py / .mcp.json)
かどうかを判定する薄い predicate。

## 4 サイト統一

本 helper は以下の 4 サイトで共有される (path 評価の単一 source of truth):

1. `core/runtime/approval_rules.py` (Phase 6.2 新規) — approval policy
2. `core/runtime/hooks.py:make_git_auto_stash_hook` (段階12 Step 3)
3. `core/runtime/hooks.py:make_post_body_modify_pending_hook` (段階12 Step 5)
4. `core/runtime/hooks.py:make_file_access_guard` (H-2 C.4 + 段階12 Step 2)

新 path quirk が発見されたら本 helper 1 か所修正で 4 サイト全反映。
"""
import os
from pathlib import Path
from typing import Iterable, Optional


# null byte + 制御 byte (path に含まれるべきでない)
# 0x09 (tab) / 0x0a (LF) / 0x0d (CR) も含める (whitespace 系で path に紛れる罠)
_DISALLOWED_CHARS = frozenset(chr(c) for c in range(0x00, 0x20)) | {chr(0x7f)}


def path_resolve_for_policy(
    path: str,
    workspace_root,
) -> Optional[str]:
    """input path を canonical な workspace 相対 POSIX path に変換。

    失敗時 (workspace 外 / disallowed char / resolve error) は **None**。
    raw fallback はしない (Codex review root cause fix)。

    Args:
        path: tool_input["path"] から取得した raw 文字列
        workspace_root: profile workspace root (pathlib.Path or str or None)。
            None なら raw を strip して返す (test 後方互換)。
            ★ **本番運用では必ず workspace_root 指定** (Codex rescue
            7 周目 RISK-6 評価)、None fallback は test 用のみ。main.py の
            `make_policy_fn(workspace_root=BASE_DIR)` 経由で必ず渡される。

    Returns:
        canonical workspace 相対 POSIX path (Windows lowercase) または None
    """
    if not path:
        return None
    sanitized = path.strip()
    if not sanitized:
        return None
    # null byte / 制御 byte を含む path は reject
    if any(c in _DISALLOWED_CHARS for c in sanitized):
        return None
    if workspace_root is None:
        # test 後方互換: workspace 知らないので raw 返却 (canonical 化なし)
        return sanitized
    try:
        root = Path(workspace_root).resolve()
        target = (root / sanitized).resolve()
        rel = target.relative_to(root).as_posix()
    except (OSError, ValueError):
        return None
    if os.name == "nt":
        rel = rel.lower()
    return rel


def is_body_modify_path(
    path: str,
    workspace_root,
    dir_prefixes: Iterable[str] = ("core/", "tools/"),
    filenames: Iterable[str] = ("main.py", ".mcp.json"),
) -> bool:
    """path が iku の身体改変領域 (.py / .mcp.json) に該当するか判定。

    `path_resolve_for_policy` の戻り値で判定する薄い predicate。
    workspace 外 / 不正 path / 空 path は False (= body_modify ではない、
    file_access_guard 等の上位 layer に deny を委譲)。

    Args:
        path: tool_input["path"] から取得した raw 文字列
        workspace_root: profile workspace root
        dir_prefixes: 身体改変扱いの dir prefix tuple (POSIX、末尾 `/` 必須)
        filenames: 身体改変扱いの file 名 tuple

    Returns:
        True なら身体改変経路、False なら通常経路
    """
    rel = path_resolve_for_policy(path, workspace_root)
    if rel is None:
        return False

    if os.name == "nt":
        prefixes = tuple(p.lower() for p in dir_prefixes)
        names = frozenset(n.lower() for n in filenames)
    else:
        prefixes = tuple(dir_prefixes)
        names = frozenset(filenames)

    if rel in names:
        return True
    if any(rel.startswith(p) for p in prefixes):
        return True
    return False
