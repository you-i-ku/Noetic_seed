"""段階13 Phase 0.1.A — raw/subjective dual emit + 未分類検出機構 test。

PLAN §20-2 / handoff memo の判断軸 7 (RAW/SUBJECTIVE/PARTITION_META) に従って
entry が分配されること、未知 field が _unclassified_fields マーカー + 永続記録
(schema_warnings.jsonl) で見える化されること、cap 1000 超で archive_schema_warnings.jsonl
に move されることを検証する。

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_event_emission.py
"""
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.memory as cm
from core.memory import (
    _split_entry_fields,
    _archive_entries,
    _record_schema_warning,
    SCHEMA_WARNINGS_CAP,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup_tmp_memory():
    """tmp dir を作って core.memory.MEMORY_DIR を差し替え、原値を返す。"""
    tmp = Path(tempfile.mkdtemp(prefix="noetic_event_emit_"))
    original = cm.MEMORY_DIR
    cm.MEMORY_DIR = tmp
    return tmp, original


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_split_known_fields_only():
    """既知 field のみの entry が RAW / SUBJECTIVE に正しく分配されること。"""
    print("== _split_entry_fields: 既知 field のみ ==")
    entry = {
        "id": "abc123",
        "time": "2026-05-02 12:00:00",
        "tool": "search_memory",
        "args": {"query": "test"},
        "result": "ok",
        "type": "tool",
        "channel": "internal",
        "intent": "memory 検索",
        "expect": "result が返る",
        "e1": 0.5, "e2": 0.6, "e3": 0.7, "e4": 0.5,
        "predicted_e2": 60, "actual_e2": 65, "prediction_error": 5,
        "predicted_ec": 0.5, "actual_ec": 0.55, "prediction_error_ec": 0.05,
        "per_tool": [],
        "perspective": {"viewer": "self"},
    }
    raw, subj, unknown = _split_entry_fields(entry)
    ok = True
    ok &= _assert(unknown == [], "unknown は空 list")
    ok &= _assert("_unclassified_fields" not in raw, "raw に marker 追加されてない")
    ok &= _assert(raw["id"] == "abc123" and subj["id"] == "abc123", "id 両側に重複")
    ok &= _assert("tool" in raw and "tool" not in subj, "tool は raw のみ")
    ok &= _assert("intent" in subj and "intent" not in raw, "intent は subj のみ")
    ok &= _assert("e2" in subj and "e2" not in raw, "e2 (mutable) は subj のみ")
    ok &= _assert(
        raw["perspective"] == subj["perspective"],
        "perspective は両側同値",
    )
    return ok


def test_split_partition_meta_duplicated():
    """to_perspective が perspective と並んで両側に重複保持されること。"""
    print("== _split_entry_fields: to_perspective も両側 ==")
    entry = {
        "id": "x1",
        "time": "t",
        "tool": "output_display",
        "perspective": {"viewer": "self"},
        "to_perspective": {"viewer": "claude"},
        "intent": "応答",
    }
    raw, subj, unknown = _split_entry_fields(entry)
    ok = True
    ok &= _assert(
        "to_perspective" in raw and "to_perspective" in subj,
        "to_perspective は両側に存在",
    )
    ok &= _assert(
        raw["to_perspective"]["viewer"] == "claude"
        and subj["to_perspective"]["viewer"] == "claude",
        "to_perspective の値が保持される",
    )
    return ok


def test_archive_dual_emit():
    """_archive_entries が raw_events.jsonl / subjective_entries.jsonl に dual emit すること。"""
    print("== _archive_entries: dual emit ==")
    tmp, original = _setup_tmp_memory()
    try:
        entry = {
            "id": "e1",
            "time": "2026-05-02 12:00:00",
            "tool": "search_memory",
            "result": "ok",
            "type": "tool",
            "intent": "test",
            "e2": 0.5,
            "perspective": {"viewer": "self"},
        }
        _archive_entries([entry])
        raw_events = _read_jsonl(tmp / "raw_events.jsonl")
        subj_entries = _read_jsonl(tmp / "subjective_entries.jsonl")
        # legacy archive は実装が datetime.now() で命名するので test も動的生成
        # (Codex review 指摘 P2 反映、日付依存の hardcode 撤去)
        today = datetime.now().strftime("%Y%m%d")
        legacy = _read_jsonl(tmp / f"archive_{today}.jsonl")
        ok = True
        ok &= _assert(len(raw_events) == 1, "raw_events.jsonl に 1 行")
        ok &= _assert(len(subj_entries) == 1, "subjective_entries.jsonl に 1 行")
        ok &= _assert(
            raw_events[0]["id"] == "e1" and subj_entries[0]["id"] == "e1",
            "id 両側に同値",
        )
        ok &= _assert(
            "tool" in raw_events[0] and "intent" in subj_entries[0],
            "raw 側に tool / subj 側に intent",
        )
        ok &= _assert(
            len(legacy) == 1, "既存 archive_YYYYMMDD.jsonl 経路も並走で動く",
        )
        return ok
    finally:
        cm.MEMORY_DIR = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_unclassified_field_workflow():
    """未知 field が raw 側に保全 + _unclassified_fields marker + schema_warnings.jsonl に記録されること。"""
    print("== 未分類 field の見える化 ==")
    tmp, original = _setup_tmp_memory()
    try:
        entry = {
            "id": "e2",
            "time": "t",
            "tool": "x",
            "intent": "y",
            "future_field_alpha": "future_value",  # ← 未知 field
        }
        raw, subj, unknown = _split_entry_fields(entry)
        ok = True
        ok &= _assert(unknown == ["future_field_alpha"], "unknown に未知 field")
        ok &= _assert(
            raw.get("future_field_alpha") == "future_value",
            "未知 field は raw 側に保全",
        )
        ok &= _assert(
            raw.get("_unclassified_fields") == ["future_field_alpha"],
            "raw に _unclassified_fields マーカー",
        )
        ok &= _assert(
            "future_field_alpha" not in subj,
            "未知 field は subj 側には行かない",
        )
        # _archive_entries 経由で schema_warnings.jsonl にも記録される
        _archive_entries([entry])
        warnings = _read_jsonl(tmp / "schema_warnings.jsonl")
        ok &= _assert(len(warnings) == 1, "schema_warnings.jsonl に 1 行")
        ok &= _assert(
            warnings[0]["entry_id"] == "e2"
            and warnings[0]["fields"] == ["future_field_alpha"],
            "warning の entry_id / fields が正しい",
        )
        return ok
    finally:
        cm.MEMORY_DIR = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_schema_warnings_cap_archive():
    """SCHEMA_WARNINGS_CAP 超で archive_schema_warnings.jsonl に move、新規スタートすること。"""
    print(f"== schema_warnings cap (={SCHEMA_WARNINGS_CAP}) で archive ==")
    tmp, original = _setup_tmp_memory()
    try:
        # cap ぴったりまで埋める (cap 件は move されない、cap+1 件目で move)
        for i in range(SCHEMA_WARNINGS_CAP):
            _record_schema_warning(f"e{i}", [f"f{i}"])
        warnings = _read_jsonl(tmp / "schema_warnings.jsonl")
        archive = tmp / "archive_schema_warnings.jsonl"
        ok = True
        ok &= _assert(
            len(warnings) == SCHEMA_WARNINGS_CAP,
            f"{SCHEMA_WARNINGS_CAP} 件入った時点では archive 未発生",
        )
        ok &= _assert(not archive.exists(), "archive_schema_warnings.jsonl 未作成")

        # cap+1 件目: move trigger
        _record_schema_warning("e_overflow", ["overflow_field"])
        warnings_after = _read_jsonl(tmp / "schema_warnings.jsonl")
        archive_after = _read_jsonl(archive)
        ok &= _assert(
            len(warnings_after) == 1 and warnings_after[0]["entry_id"] == "e_overflow",
            "schema_warnings.jsonl は新規 1 件で再スタート",
        )
        ok &= _assert(
            len(archive_after) == SCHEMA_WARNINGS_CAP,
            f"archive に {SCHEMA_WARNINGS_CAP} 件 move された",
        )
        return ok
    finally:
        cm.MEMORY_DIR = original
        shutil.rmtree(tmp, ignore_errors=True)


def test_channel_field_optional():
    """channel field の有無で main entry / _ext_entry の両方で動くこと
    (channel field 不在 = internal の意、空のまま raw に行く)。"""
    print("== channel あり/なし両方で動く ==")
    tmp, original = _setup_tmp_memory()
    try:
        # _ext_entry 模擬 (channel あり)
        ext_entry = {
            "id": "ext1",
            "time": "t",
            "tool": "[claude_input]",
            "type": "external",
            "channel": "claude",
            "result": "hi",
            "perspective": {"viewer": "claude"},
        }
        # main entry 模擬 (channel なし、internal)
        main_entry = {
            "id": "main1",
            "time": "t",
            "tool": "search_memory",
            "args": {"query": "x"},
            "result": "ok",
            "intent": "test",
            "e2": 0.5,
            "perspective": {"viewer": "self"},
        }
        _archive_entries([ext_entry, main_entry])
        raw_events = _read_jsonl(tmp / "raw_events.jsonl")
        ok = True
        ok &= _assert(len(raw_events) == 2, "raw_events に 2 行")
        ok &= _assert(
            raw_events[0]["channel"] == "claude",
            "_ext_entry の channel は raw 側に保持",
        )
        ok &= _assert(
            "channel" not in raw_events[1],
            "main entry の channel field 不在 = internal 信号として保存",
        )
        return ok
    finally:
        cm.MEMORY_DIR = original
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [
        test_split_known_fields_only,
        test_split_partition_meta_duplicated,
        test_archive_dual_emit,
        test_unclassified_field_workflow,
        test_schema_warnings_cap_archive,
        test_channel_field_optional,
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
