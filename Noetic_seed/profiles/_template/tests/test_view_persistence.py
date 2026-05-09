"""F-005 view_persistence — registry pattern + atomic rewrite + 統合 test。

正典: `WORLD_MODEL_DESIGN/PLAN_CODE_TRACEABILITY_TABLE.md` §3.1 F-005
      `WORLD_MODEL_DESIGN/NOETIC_INTEGRATED_ORCHESTRATION_PLAN.md` Slice 1

検証範囲 (CLAUDE.md §5 識別力 + §6 docstring 同期 遵守):

A. 純粋 registry 振る舞い (`view_persistence.py` 単体):
   A1. register_view → mark_view_dirty → flush_dirty_views で entries 書かれる
   A2. flush は dirty 立ってる view のみ書く (未登録は touch しない、識別力)
   A3. flush 2 回目は no-op (dirty clear 後は書き換わらない、識別力)
   A4. 複数 view 登録 → 個別 mark_dirty で個別 flush 制御

B. atomic_rewrite_jsonl primitive:
   B1. 既存 jsonl の内容を完全置換 (append じゃない、識別力)
   B2. 空 list rewrite → 空ファイル (compaction で全削除した時の正しい挙動)

C. _sync_index_to_actual:
   C1. count は actual entries 数に同期 (既存 _update_jsonl_index の += と区別、識別力)
   C2. from は維持 (履歴最古は変えない)、to は actual 末尾

D. 統合: maybe_compress_log → flush:
   D1. compaction Trigger1 (HARD_LIMIT) 後、jsonl が trim 後 content と一致
   D2. 識別力: compaction 前の jsonl と異なる、state とも一致

E. 統合: _apply_retro_e2 → flush:
   E1. retro e2 修正後、jsonl の該当 entry の e2 が更新値
   E2. 識別力: 他 entry は無変更 (state と jsonl の対応一致)

F. 起動再起動 round-trip:
   F1. compaction → save_state → 新 state で load_state → trim 後の view 復元
   F2. 識別力: F-005 fix なしだと undo されて元に戻る (現状 bug を本 test が捕まえる)

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_view_persistence.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.memory as cm
import core.state as cs
import core.view_persistence as vp
from core.state import load_state, save_state, _rebuild_views_from_jsonl


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup_tmp_dirs():
    """tmp dir を作って core.memory + core.state の MEMORY_DIR / STATE_FILE を差し替える。"""
    tmp = Path(tempfile.mkdtemp(prefix="noetic_view_persistence_"))
    originals = (cm.MEMORY_DIR, cs.MEMORY_DIR, cs.STATE_FILE)
    cm.MEMORY_DIR = tmp
    cs.MEMORY_DIR = tmp
    cs.STATE_FILE = tmp / "state.json"
    vp._clear_registry_for_test()
    return tmp, originals


def _restore(originals):
    cm.MEMORY_DIR, cs.MEMORY_DIR, cs.STATE_FILE = originals
    vp._clear_registry_for_test()


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ============================================================
# A. 純粋 registry 振る舞い
# ============================================================

def test_a1_register_mark_flush_basic():
    """A1: register → mark_dirty → flush で entries 書かれる基本フロー。"""
    print("== A1: register → mark_dirty → flush 基本フロー ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        target = tmp / "view_a.jsonl"
        vp.register_view("view_a", target, "view_a_data")
        state = {"view_a_data": [{"id": "x", "v": 1}, {"id": "y", "v": 2}]}
        vp.mark_view_dirty("view_a")
        flushed = vp.flush_dirty_views(state)
        ok = True
        ok &= _assert(flushed == 1, f"flushed=1 (got {flushed})")
        ok &= _assert(target.exists(), "target jsonl 作成された")
        records = _read_jsonl(target)
        ok &= _assert(len(records) == 2, f"jsonl 2 件 (got {len(records)})")
        ok &= _assert(
            records[0]["id"] == "x" and records[1]["id"] == "y",
            "順序保持 + 内容一致 (識別力: 1 件しか書かない実装だと fail)",
        )
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def test_a2_flush_only_dirty_views():
    """A2: flush は dirty 立ってる view のみ書く。未 dirty / 未登録は touch しない (識別力)。

    識別力 fixture: 2 view 登録、片方 mark_dirty。flush 後、片方のみ jsonl 存在。
    両方書く誤実装 (`for name in _VIEWS:`) なら両方の jsonl が作られて fail。
    """
    print("== A2: flush は dirty な view のみ書く ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        target_a = tmp / "view_a.jsonl"
        target_b = tmp / "view_b.jsonl"
        vp.register_view("view_a", target_a, "view_a_data")
        vp.register_view("view_b", target_b, "view_b_data")
        state = {
            "view_a_data": [{"id": "a1"}],
            "view_b_data": [{"id": "b1"}],
        }
        vp.mark_view_dirty("view_a")  # B は dirty 立てない
        flushed = vp.flush_dirty_views(state)
        ok = True
        ok &= _assert(flushed == 1, f"flushed=1、B は触らない (got {flushed})")
        ok &= _assert(target_a.exists(), "A は書かれた")
        ok &= _assert(not target_b.exists(), "B は書かれない (識別力: 全 view 書く実装だと fail)")
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def test_a3_flush_idempotent_after_clear():
    """A3: flush 2 回目は dirty clear 済なので no-op (識別力)。

    識別力 fixture: 1 回 flush 後、target を rm。次の flush で再作成されたら fail
    (clear せず毎回 flush する誤実装)。
    """
    print("== A3: flush 後 dirty clear、再 flush は no-op ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        target = tmp / "view_a.jsonl"
        vp.register_view("view_a", target, "view_a_data")
        state = {"view_a_data": [{"id": "x"}]}
        vp.mark_view_dirty("view_a")
        first = vp.flush_dirty_views(state)
        ok = True
        ok &= _assert(first == 1, f"1 回目 flushed=1 (got {first})")
        target.unlink()  # 削除して 2 回目で再作成されないことを観測
        second = vp.flush_dirty_views(state)
        ok &= _assert(second == 0, f"2 回目 flushed=0 (no-op、got {second})")
        ok &= _assert(
            not target.exists(),
            "target 再作成されない (識別力: dirty clear 漏れ実装だと再作成される)",
        )
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def test_a4_unregistered_mark_is_noop():
    """A4: 未登録 name の mark_view_dirty は silent no-op、flush 影響なし。"""
    print("== A4: 未登録 name の mark は silent no-op ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        vp.mark_view_dirty("not_registered_view")  # 例外発生せずに済むこと
        ok = True
        ok &= _assert(
            not vp._is_dirty_for_test("not_registered_view"),
            "未登録 name は dirty 集合に入らない",
        )
        flushed = vp.flush_dirty_views({})
        ok &= _assert(flushed == 0, f"flushed=0 (got {flushed})")
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# B. atomic_rewrite_jsonl primitive
# ============================================================

def test_b1_atomic_rewrite_replaces_content():
    """B1: 既存 jsonl の content を完全置換 (append じゃない、識別力)。

    識別力 fixture: 既存 file に 3 件、rewrite で 1 件渡す。期待: rewrite 後 1 件のみ。
    append 実装なら 4 件になって fail。
    """
    print("== B1: atomic_rewrite_jsonl は完全置換 (append じゃない) ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        target = tmp / "test.jsonl"
        target.write_text(
            "\n".join(json.dumps({"id": f"old_{i}"}) for i in range(3)) + "\n",
            encoding="utf-8",
        )
        ok = True
        ok &= _assert(len(_read_jsonl(target)) == 3, "事前 3 件あり")
        vp.atomic_rewrite_jsonl(target, [{"id": "new_only"}])
        records = _read_jsonl(target)
        ok &= _assert(
            len(records) == 1 and records[0]["id"] == "new_only",
            "rewrite 後 1 件のみ (識別力: append 実装なら 4 件になって fail)",
        )
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def test_b2_atomic_rewrite_empty_list_yields_empty_file():
    """B2: 空 list rewrite → 空ファイル (compaction で全削除する場合の正しい挙動)。"""
    print("== B2: 空 list rewrite で空ファイル ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        target = tmp / "test.jsonl"
        target.write_text(json.dumps({"id": "old"}) + "\n", encoding="utf-8")
        vp.atomic_rewrite_jsonl(target, [])
        ok = True
        ok &= _assert(target.exists(), "ファイルは存在する")
        ok &= _assert(target.read_text(encoding="utf-8") == "", "中身は空")
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# C. _sync_index_to_actual
# ============================================================

def test_c1_index_sync_count_replaces_not_increment():
    """C1: rewrite 後 count = actual entries 数に同期 (既存 _update_jsonl_index の += と識別)。

    識別力 fixture: pre count=10、entries=3 件で sync。期待: count=3。
    += 実装なら count=13 になって fail。
    """
    print("== C1: _sync_index_to_actual は count 置換 (= ではなく +=) ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        index_path = tmp / "index.json"
        index_path.write_text(json.dumps({
            "view_a.jsonl": {"count": 10, "from": "t0", "to": "t9"}
        }), encoding="utf-8")
        entries = [{"id": "a", "time": "t100"}, {"id": "b", "time": "t101"}, {"id": "c", "time": "t102"}]
        vp._sync_index_to_actual(index_path, "view_a.jsonl", entries)
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        ok = True
        ok &= _assert(
            idx["view_a.jsonl"]["count"] == 3,
            f"count=3 (識別力: += 実装なら 13、got {idx['view_a.jsonl']['count']})",
        )
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def test_c2_index_from_preserved_to_updated():
    """C2: from は履歴最古を維持、to は actual 末尾の time に更新。"""
    print("== C2: from 維持 + to 更新 ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        index_path = tmp / "index.json"
        index_path.write_text(json.dumps({
            "view_a.jsonl": {"count": 5, "from": "t_origin", "to": "t_old_last"}
        }), encoding="utf-8")
        entries = [{"id": "a", "time": "t_new_first"}, {"id": "b", "time": "t_new_last"}]
        vp._sync_index_to_actual(index_path, "view_a.jsonl", entries)
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        ok = True
        ok &= _assert(
            idx["view_a.jsonl"]["from"] == "t_origin",
            "from は履歴最古を維持 (識別力: 上書きすると最古性失われる)",
        )
        ok &= _assert(
            idx["view_a.jsonl"]["to"] == "t_new_last",
            "to は actual 末尾",
        )
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# D. 統合: maybe_compress_log → flush
# ============================================================

def test_d1_compaction_persists_to_jsonl():
    """D1: maybe_compress_log Trigger1 (HARD_LIMIT) 発火後、save_state で jsonl が trim 後 content と一致。

    識別力 fixture: HARD_LIMIT(150) 件超 entries、compaction → save_state →
    jsonl 読み直しで件数が trim 後と一致。F-005 fix なしの実装 (state mutate のみ)
    なら jsonl は元の件数のままで fail。
    """
    print("== D1: compaction → save_state → jsonl trim 後と一致 ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        target = tmp / "subjective_entries.jsonl"
        vp.register_view(
            "subjective_entries", target, "subjective_entries",
            index_path=tmp / "index.json",
        )
        # HARD_LIMIT(150) 超え state 用意
        from core.config import LOG_HARD_LIMIT, LOG_KEEP
        n = LOG_HARD_LIMIT + 5  # 155 件
        state = {
            "subjective_entries": [
                {"id": f"s_{i}", "time": f"t_{i:04d}", "tool": "x", "intent": f"intent_{i}"}
                for i in range(n)
            ],
            "summaries": [],
        }
        # 事前: jsonl も同 state で初期化 (compaction 前の jsonl 想定)
        vp.atomic_rewrite_jsonl(target, state["subjective_entries"])
        pre_jsonl = _read_jsonl(target)
        ok = True
        ok &= _assert(len(pre_jsonl) == n, f"事前 jsonl {n} 件")
        # compaction 発火
        cm.maybe_compress_log(state, tool_names={"x"})
        # save_state で flush 発火
        save_state(state)
        post_jsonl = _read_jsonl(target)
        ok &= _assert(
            len(post_jsonl) == len(state["subjective_entries"]),
            f"jsonl と state の subj 件数一致 ({len(post_jsonl)} == {len(state['subjective_entries'])})",
        )
        ok &= _assert(
            len(post_jsonl) < n,
            f"jsonl が compaction で trim された (識別力: F-005 fix なしなら {n} 件のまま、got {len(post_jsonl)})",
        )
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# E. 統合: _apply_retro_e2 → flush
# ============================================================

def test_e1_retro_e2_persists_to_jsonl():
    """E1: _apply_retro_e2 で e2 mutate → save_state で jsonl 該当 entry が更新値。

    識別力 fixture: 2 entry の e2 が "30%" / "50%"、log_id="e_target" に +40%。
    期待: jsonl の e_target は "70%"、他 entry は無変更。
    F-005 fix なしなら jsonl 側は state mutate 反映されず元のままで fail。
    """
    print("== E1: _apply_retro_e2 → save_state → jsonl 該当 entry 更新 ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        target = tmp / "subjective_entries.jsonl"
        vp.register_view(
            "subjective_entries", target, "subjective_entries",
            index_path=tmp / "index.json",
        )
        from core.pending_unified import _apply_retro_e2
        state = {
            "subjective_entries": [
                {"id": "e_other", "e2": "30%", "intent": "no_change"},
                {"id": "e_target", "e2": "50%", "intent": "boost_me"},
            ],
        }
        # 事前 jsonl 初期化
        vp.atomic_rewrite_jsonl(target, state["subjective_entries"])
        # retro e2 boost
        result = _apply_retro_e2(state, "e_target", 40)
        ok = True
        ok &= _assert(result is True, "retro e2 適用成功")
        # state 側: target は 90%、other は無変更
        ok &= _assert(
            state["subjective_entries"][1]["e2"] == "90%",
            f"state target e2=90% (got {state['subjective_entries'][1]['e2']})",
        )
        # save_state で flush
        save_state(state)
        post_jsonl = _read_jsonl(target)
        ok &= _assert(
            post_jsonl[1]["e2"] == "90%",
            f"jsonl target e2=90% (識別力: F-005 fix なしなら 50% のまま、got {post_jsonl[1]['e2']})",
        )
        ok &= _assert(
            post_jsonl[0]["e2"] == "30%",
            f"jsonl other e2=30% 無変更 (got {post_jsonl[0]['e2']})",
        )
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# F. 起動再起動 round-trip
# ============================================================

def test_f1_restart_persists_compaction():
    """F1: compaction → save_state → 新 state で load_state → trim 後の view が復元される。

    識別力 fixture: compaction → save_state 後、新セッション (vp registry リセット +
    新 load_state) で state["subjective_entries"] が trim 後の件数。
    F-005 fix なしの実装なら jsonl が古いまま、_rebuild_views_from_jsonl で元の件数に戻り fail。
    本 test は F-005 の主目的「再起動で undo されない」を機械的に検証する。
    """
    print("== F1: compaction → save → reload で trim 後 view 復元 ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        target = tmp / "subjective_entries.jsonl"
        vp.register_view(
            "subjective_entries", target, "subjective_entries",
            index_path=tmp / "index.json",
        )
        from core.config import LOG_HARD_LIMIT
        n = LOG_HARD_LIMIT + 5
        state = {
            "subjective_entries": [
                {"id": f"s_{i}", "time": f"t_{i:04d}", "tool": "x", "intent": f"intent_{i}"}
                for i in range(n)
            ],
            "summaries": [],
        }
        vp.atomic_rewrite_jsonl(target, state["subjective_entries"])
        cm.maybe_compress_log(state, tool_names={"x"})
        trimmed_count = len(state["subjective_entries"])
        save_state(state)
        # 新セッション 想定: registry reset + 新 load_state
        vp._clear_registry_for_test()
        vp.register_view(
            "subjective_entries", target, "subjective_entries",
            index_path=tmp / "index.json",
        )
        new_state = load_state()
        ok = True
        ok &= _assert(
            len(new_state["subjective_entries"]) == trimmed_count,
            f"reload 後の subj 件数={trimmed_count} (識別力: F-005 fix なしなら {n} 件で fail、got {len(new_state['subjective_entries'])})",
        )
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    tests = [
        test_a1_register_mark_flush_basic,
        test_a2_flush_only_dirty_views,
        test_a3_flush_idempotent_after_clear,
        test_a4_unregistered_mark_is_noop,
        test_b1_atomic_rewrite_replaces_content,
        test_b2_atomic_rewrite_empty_list_yields_empty_file,
        test_c1_index_sync_count_replaces_not_increment,
        test_c2_index_from_preserved_to_updated,
        test_d1_compaction_persists_to_jsonl,
        test_e1_retro_e2_persists_to_jsonl,
        test_f1_restart_persists_compaction,
    ]
    results = []
    for t in tests:
        try:
            results.append((t.__name__, t()))
        except Exception as e:
            print(f"  [FAIL] {t.__name__} 例外: {e}")
            import traceback
            traceback.print_exc()
            results.append((t.__name__, False))
        print()
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"=== {passed}/{total} passed ===")
    sys.exit(0 if passed == total else 1)
