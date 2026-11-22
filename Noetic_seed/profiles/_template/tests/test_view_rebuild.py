"""段階13 Phase 0.1.B — 起動時 view rebuild + index.json 拡張 test。

PLAN §18-2「段階7 materialized view パターン継承」: jsonl が source of truth、
state["raw_events"] / state["subjective_entries"] は in-memory view、起動時に
必ず rebuild して state.json の値より jsonl 側を優先する。

検証範囲:
  1. clean profile (jsonl 不在) → state[key] = []
  2. populated jsonl から全 entry を loaded
  3. 壊れた行 (json parse 失敗) は skip して続行
  4. state[key] に stale 値があっても rebuild で完全置換 (jsonl 優先)
  5. 片側だけの jsonl (raw だけ / subj だけ) でも他方は []
  6. _archive_entries → _rebuild_views_from_jsonl の round-trip 整合
  7. _archive_entries 後 index.json に raw_events / subjective_entries の
     count/from/to が追跡される (B-1 behavior)

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_view_rebuild.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.memory as cm
import core.state as cs
from core.state import _rebuild_views_from_jsonl


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup_tmp_dirs():
    """tmp dir を作って core.memory + core.state の MEMORY_DIR を差し替える。
    STATE_FILE も tmp に向けて load_state を独立稼働可能にする。
    """
    tmp = Path(tempfile.mkdtemp(prefix="noetic_view_rebuild_"))
    originals = (cm.MEMORY_DIR, cs.MEMORY_DIR, cs.STATE_FILE)
    cm.MEMORY_DIR = tmp
    cs.MEMORY_DIR = tmp
    cs.STATE_FILE = tmp / "state.json"
    return tmp, originals


def _restore(originals):
    cm.MEMORY_DIR, cs.MEMORY_DIR, cs.STATE_FILE = originals


def _write_jsonl(path: Path, records):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_rebuild_clean_profile():
    """jsonl 不在 (clean profile 起動) → state[key] は空 list で初期化。"""
    print("== rebuild: clean profile (jsonl 不在) ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        state = {"raw_events": [{"stale": True}], "subjective_entries": [{"stale": True}]}
        _rebuild_views_from_jsonl(state)
        ok = True
        ok &= _assert(state["raw_events"] == [], "raw_events 空 list (stale 値も払拭)")
        ok &= _assert(state["subjective_entries"] == [], "subjective_entries 空 list")
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def test_rebuild_loads_all_entries():
    """jsonl の全 entry が in-memory view に loaded されること。"""
    print("== rebuild: 全 entry loaded ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        raw_records = [
            {"id": "r1", "time": "t1", "tool": "x"},
            {"id": "r2", "time": "t2", "tool": "y"},
            {"id": "r3", "time": "t3", "tool": "z"},
        ]
        subj_records = [
            {"id": "r1", "intent": "a"},
            {"id": "r2", "intent": "b"},
        ]
        _write_jsonl(tmp / "raw_events.jsonl", raw_records)
        _write_jsonl(tmp / "subjective_entries.jsonl", subj_records)
        state = {}
        _rebuild_views_from_jsonl(state)
        ok = True
        ok &= _assert(len(state["raw_events"]) == 3, "raw_events 3 件")
        ok &= _assert(len(state["subjective_entries"]) == 2, "subjective_entries 2 件")
        ok &= _assert(
            state["raw_events"][0]["id"] == "r1" and state["raw_events"][2]["id"] == "r3",
            "raw_events 順序保持",
        )
        ok &= _assert(
            state["subjective_entries"][1]["intent"] == "b",
            "subjective_entries 内容正しい",
        )
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def test_rebuild_skips_corrupted_lines():
    """壊れた行 (json parse 失敗) は skip して続行する。"""
    print("== rebuild: 壊れた行 skip ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        raw_path = tmp / "raw_events.jsonl"
        raw_path.write_text(
            '{"id": "r1", "time": "t1"}\n'
            'this-is-not-json\n'
            '{"id": "r2", "time": "t2"}\n'
            '\n'
            '{broken json\n'
            '{"id": "r3", "time": "t3"}\n',
            encoding="utf-8",
        )
        state = {}
        _rebuild_views_from_jsonl(state)
        ok = True
        ok &= _assert(
            len(state["raw_events"]) == 3,
            "壊れた行 skip して 3 件 loaded",
        )
        ok &= _assert(
            [e["id"] for e in state["raw_events"]] == ["r1", "r2", "r3"],
            "loaded 順序が source 順と一致",
        )
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def test_rebuild_overwrites_stale_view():
    """state[key] に stale 値があっても rebuild で完全置換 (jsonl 優先)。"""
    print("== rebuild: stale view を完全置換 ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        _write_jsonl(tmp / "raw_events.jsonl", [{"id": "r_new"}])
        _write_jsonl(tmp / "subjective_entries.jsonl", [{"id": "s_new"}])
        state = {
            "raw_events": [{"id": "r_stale_a"}, {"id": "r_stale_b"}],
            "subjective_entries": [{"id": "s_stale_x"}],
        }
        _rebuild_views_from_jsonl(state)
        ok = True
        ok &= _assert(
            state["raw_events"] == [{"id": "r_new"}],
            "raw_events: stale 2 件が jsonl 1 件で置換",
        )
        ok &= _assert(
            state["subjective_entries"] == [{"id": "s_new"}],
            "subjective_entries: stale 1 件が jsonl 1 件で置換",
        )
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def test_rebuild_partial_files():
    """raw だけ / subj だけ存在の片側 case。他方は [] のまま。"""
    print("== rebuild: 片側だけ存在 ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        # raw だけ存在
        _write_jsonl(tmp / "raw_events.jsonl", [{"id": "r1"}, {"id": "r2"}])
        state_a = {}
        _rebuild_views_from_jsonl(state_a)
        ok = True
        ok &= _assert(len(state_a["raw_events"]) == 2, "raw だけ存在 → raw 2 件")
        ok &= _assert(state_a["subjective_entries"] == [], "subjective は []")

        # raw 削除して subj だけ
        (tmp / "raw_events.jsonl").unlink()
        _write_jsonl(tmp / "subjective_entries.jsonl", [{"id": "s1"}])
        state_b = {}
        _rebuild_views_from_jsonl(state_b)
        ok &= _assert(state_b["raw_events"] == [], "subj だけ存在 → raw は []")
        ok &= _assert(len(state_b["subjective_entries"]) == 1, "subjective 1 件")
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def test_archive_to_rebuild_roundtrip():
    """_archive_entries で書いた entry を rebuild で再構築、整合する。"""
    print("== round-trip: archive → rebuild ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        entries = [
            {
                "id": "e1",
                "time": "2026-05-02 12:00:00",
                "tool": "search_memory",
                "args": {"q": "a"},
                "result": "ok",
                "intent": "test",
                "e2": 0.5,
                "perspective": {"viewer": "self"},
            },
            {
                "id": "e2",
                "time": "2026-05-02 12:00:30",
                "tool": "view_image",
                "args": {"path": "x.png"},
                "result": "loaded",
                "intent": "見る",
                "e2": 0.7,
                "perspective": {"viewer": "self"},
            },
        ]
        cm._archive_entries(entries)
        state = {}
        _rebuild_views_from_jsonl(state)
        ok = True
        ok &= _assert(len(state["raw_events"]) == 2, "raw_events に 2 件 rebuild")
        ok &= _assert(len(state["subjective_entries"]) == 2, "subjective_entries に 2 件")
        ok &= _assert(
            [e["id"] for e in state["raw_events"]] == ["e1", "e2"],
            "raw_events: id 順序保持",
        )
        ok &= _assert(
            "tool" in state["raw_events"][0] and "intent" not in state["raw_events"][0],
            "raw_events: tool あり intent なし",
        )
        ok &= _assert(
            "intent" in state["subjective_entries"][0]
            and "tool" not in state["subjective_entries"][0],
            "subjective_entries: intent あり tool なし",
        )
        ok &= _assert(
            state["raw_events"][0]["perspective"]
            == state["subjective_entries"][0]["perspective"],
            "perspective は両側で同値 (partition meta)",
        )
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def test_index_json_tracks_raw_subjective():
    """段階13 Phase 0.1.B-1: _archive_entries 後、index.json に
    raw_events.jsonl / subjective_entries.jsonl の count/from/to が追跡される。"""
    print("== index.json: raw / subjective 追跡 ==")
    tmp, originals = _setup_tmp_dirs()
    try:
        e1 = {"id": "e1", "time": "2026-05-02 09:00:00", "tool": "x", "intent": "y"}
        e2 = {"id": "e2", "time": "2026-05-02 09:01:00", "tool": "x", "intent": "y"}
        e3 = {"id": "e3", "time": "2026-05-02 09:02:00", "tool": "x", "intent": "y"}
        # 1 回目 batch (2 件)
        cm._archive_entries([e1, e2])
        idx1 = json.loads((tmp / "index.json").read_text(encoding="utf-8"))
        ok = True
        ok &= _assert(
            "raw_events.jsonl" in idx1 and "subjective_entries.jsonl" in idx1,
            "index に raw / subjective 両 key 追加",
        )
        ok &= _assert(
            idx1["raw_events.jsonl"]["count"] == 2,
            "raw count = 2",
        )
        ok &= _assert(
            idx1["raw_events.jsonl"]["from"] == "2026-05-02 09:00:00"
            and idx1["raw_events.jsonl"]["to"] == "2026-05-02 09:01:00",
            "raw from/to 正しい",
        )
        ok &= _assert(
            idx1["subjective_entries.jsonl"]["count"] == 2,
            "subjective count = 2",
        )

        # 2 回目 batch (1 件) → count 加算、from 不変、to 更新
        cm._archive_entries([e3])
        idx2 = json.loads((tmp / "index.json").read_text(encoding="utf-8"))
        ok &= _assert(idx2["raw_events.jsonl"]["count"] == 3, "raw count 加算 → 3")
        ok &= _assert(
            idx2["raw_events.jsonl"]["from"] == "2026-05-02 09:00:00",
            "from は最初の batch の値が保持",
        )
        ok &= _assert(
            idx2["raw_events.jsonl"]["to"] == "2026-05-02 09:02:00",
            "to は最新 batch の値で更新",
        )
        return ok
    finally:
        _restore(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [
        test_rebuild_clean_profile,
        test_rebuild_loads_all_entries,
        test_rebuild_skips_corrupted_lines,
        test_rebuild_overwrites_stale_view,
        test_rebuild_partial_files,
        test_archive_to_rebuild_roundtrip,
        test_index_json_tracks_raw_subjective,
    ]
    results = []
    for t in tests:
        try:
            results.append((t.__name__, t()))
        except Exception as e:
            print(f"  [FAIL] {t.__name__} 例外: {e}")
            results.append((t.__name__, False))
        print()
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"FAILED: {len(failed)} / {len(tests)}: {failed}")
        sys.exit(1)
    print(f"ALL {len(tests)} OK")


if __name__ == "__main__":
    main()
