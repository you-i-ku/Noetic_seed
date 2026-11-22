"""View Persistence — mutable materialized view atomic rewrite + 軽量 registry pattern.

F-005 (`subjective_entries.jsonl` 永続化) の中核。raw_events (immutable, source of
truth) / subjective_entries (mutable, materialized view) 二層構造で、mutable 側の
compaction / retro mutation を atomic rewrite 経路で永続化する。

将来の同パターン view (goal_shadows / trajectories / skill_candidates 等、
orchestration plan §3-4 P2-P4) は同 registry に register するだけで同経路に乗る。

たとえ:
  「図書館に view 一覧を登録、各部署 (mutation site) は『変更しました』シールを
   貼るだけ。cycle 末に図書館員 (flush hook) がシール付きの帳簿だけまとめて
   書き戻す。新しい部署 (新 view) を作るとき → 図書館に登録するだけ、
   シール貼るだけ。各部署が個別ルールを持たなくてよい。」

正典: `WORLD_MODEL_DESIGN/PLAN_CODE_TRACEABILITY_TABLE.md` §3.1 F-005、
      `WORLD_MODEL_DESIGN/NOETIC_INTEGRATED_ORCHESTRATION_PLAN.md` Slice 1
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

# ============================================================
# Module-level registry (singleton, プロセス全体で共有)
# ============================================================

# {view_name: {"jsonl_path": Path, "state_key": str, "index_path": Path|None}}
_VIEWS: dict[str, dict] = {}
# 次 flush で書き戻す対象の view name set
_DIRTY: set[str] = set()


def register_view(
    name: str,
    jsonl_path: Path,
    state_key: str,
    index_path: Optional[Path] = None,
) -> None:
    """View を registry に登録する (冪等、同 name 再登録は path/key を更新)。

    起動時 1 回呼出が想定運用 (main.py の load_state 直後)。

    Args:
        name: view の論理名 (例: ``"subjective_entries"``)
        jsonl_path: 永続化先 jsonl の絶対 path
        state_key: ``state`` dict 内の対応 list key (mutate 対象)
        index_path: ``index.json`` の path。指定時は flush 直後に
            ``count/from/to`` を **actual entries 数に同期** する。
            ``None`` なら index は触らない。
    """
    _VIEWS[name] = {
        "jsonl_path": Path(jsonl_path),
        "state_key": state_key,
        "index_path": Path(index_path) if index_path else None,
    }


def mark_view_dirty(name: str) -> None:
    """View を dirty 状態にする。次の ``flush_dirty_views()`` 時に書き戻し対象。

    ``name`` が未登録なら silent no-op (defensive、起動順序の race 等を吸収)。
    冪等: 同 cycle 内に何度呼んでも 1 回しか flush されない (set semantics)。
    """
    if name in _VIEWS:
        _DIRTY.add(name)


def flush_dirty_views(state: dict) -> int:
    """dirty 立ってる全 view を atomic rewrite + index 同期 + flag clear。

    cycle 末で main.py から 1 回呼出が想定運用。順序は registry 登録順
    (Python dict 挿入順保持)、view 間の dependency は想定しない (各 view
    は独立 jsonl 永続化)。

    Args:
        state: state dict (``state[view["state_key"]]`` が永続化対象)

    Returns:
        flush した view 数 (0 なら no-op、smoke 観察 / metrics 用)
    """
    flushed = 0
    for name in list(_DIRTY):
        view = _VIEWS.get(name)
        if view is None:
            _DIRTY.discard(name)
            continue
        entries = state.get(view["state_key"], []) or []
        atomic_rewrite_jsonl(view["jsonl_path"], entries)
        if view["index_path"] is not None:
            _sync_index_to_actual(
                view["index_path"], view["jsonl_path"].name, entries
            )
        _DIRTY.discard(name)
        flushed += 1
    return flushed


# ============================================================
# Primitives
# ============================================================

def atomic_rewrite_jsonl(path: Path, entries: list) -> None:
    """jsonl ファイルを atomic に丸ごと書き戻す (tmp + os.replace)。

    ``core/state.py:_atomic_write`` と同流儀。compaction や retro mutation 後の
    mutable view 永続化に使う。append-only 系 (``raw_events.jsonl`` 等、
    immutable 原則) には使わない。

    Args:
        path: 書き出し先 jsonl の path
        entries: 永続化したい全 entry list (rewrite するので「全件」、append
            ではない)。空 list なら空ファイルを書く (compaction で全削除した
            場合の正しい挙動)。
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    text = "".join(
        json.dumps(e, ensure_ascii=False) + "\n" for e in entries
    )
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _sync_index_to_actual(
    index_path: Path, filename: str, entries: list
) -> None:
    """``index.json`` の ``filename`` entry を **現時点の jsonl 実 state** に同期。

    既存 ``memory.py:_update_jsonl_index`` が append 累積 (``count +=``) なのに
    対し、rewrite 系 (compaction / retro 修正) では actual count に同期する。

    挙動:
      - ``count = len(entries)`` (rewrite なので append じゃなく置換)
      - ``from`` は **履歴最古を維持** (一度設定したら更新しない)。compaction で
        最古 entry が削除されても「いつ最初に書かれたか」の歴史性は保つ
      - ``to`` は entries 末尾の time (compaction 後の現在最新)
      - ``entries`` 空でも index entry 自体は維持 (count=0 で残す、削除はしない)
    """
    index_path = Path(index_path)
    index_path.parent.mkdir(exist_ok=True, parents=True)
    index: dict = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}
    if filename not in index:
        index[filename] = {"count": 0, "from": "", "to": ""}
    index[filename]["count"] = len(entries)
    if entries:
        if not index[filename].get("from"):
            index[filename]["from"] = entries[0].get("time", "")
        index[filename]["to"] = entries[-1].get("time", "")
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# Test helpers (外部 API ではない、_test_ prefix で隔離)
# ============================================================

def _is_dirty_for_test(name: str) -> bool:
    """test 専用: dirty flag の現状確認。本番経路では呼ばない。"""
    return name in _DIRTY


def _is_registered_for_test(name: str) -> bool:
    """test 専用: registry 登録済か確認。本番経路では呼ばない。"""
    return name in _VIEWS


def _clear_registry_for_test() -> None:
    """test 専用: registry を完全リセット (test 間の汚染防止)。"""
    _VIEWS.clear()
    _DIRTY.clear()
