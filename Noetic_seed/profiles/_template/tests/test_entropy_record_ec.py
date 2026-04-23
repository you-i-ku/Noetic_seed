"""test_entropy_record_ec.py — 段階11-B Phase 3 Step 3.4 副次実装。

検証対象 (record_ec_prediction_error):
  - 基本呼出で state["prediction_error_history_ec"] に magnitude append
  - state["prediction_error_history_by_source"][source] に detail record append
    (magnitude / reason / context / timestamp field 揃ってる)
  - 複数 source を 1 state で並立保持
  - history_max 超過時 FIFO trim
  - context / reason は optional (未指定で空/空 dict)
  - 既存 history を持つ state にも追加可能 (破壊ゼロ)

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_entropy_record_ec.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.entropy import record_ec_prediction_error


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


# =========================================================================
# Section A: 基本呼出で history 書込
# =========================================================================
print("=== Section A: 基本呼出 ===")

state_a = {}
record_ec_prediction_error(
    state_a,
    source="reconciliation",
    magnitude=0.7,
    reason="contradict severity high",
    context={"new_entry_id": "mem_new", "existing_entry_id": "mem_old"},
)

_assert(
    state_a.get("prediction_error_history_ec") == [0.7],
    f"A-1 prediction_error_history_ec に 0.7 append (got {state_a.get('prediction_error_history_ec')})",
)

by_source = state_a.get("prediction_error_history_by_source", {})
src_hist = by_source.get("reconciliation", [])
_assert(len(src_hist) == 1, f"A-2 source 'reconciliation' に 1 件 (got {len(src_hist)})")

if src_hist:
    rec = src_hist[0]
    _assert(rec.get("magnitude") == 0.7, "A-3 magnitude=0.7 保存")
    _assert(rec.get("reason") == "contradict severity high", "A-4 reason 保存")
    _assert(
        rec.get("context") == {"new_entry_id": "mem_new", "existing_entry_id": "mem_old"},
        "A-5 context 保存",
    )
    _assert("timestamp" in rec, "A-6 timestamp 付与")


# =========================================================================
# Section B: 複数回呼出 / 複数 source 並立
# =========================================================================
print("=== Section B: 複数回 / 複数 source ===")

state_b = {}
record_ec_prediction_error(state_b, source="reconciliation", magnitude=0.3, reason="r1")
record_ec_prediction_error(state_b, source="reconciliation", magnitude=0.5, reason="r2")
record_ec_prediction_error(state_b, source="other_layer", magnitude=0.9, reason="ol")

_assert(
    state_b.get("prediction_error_history_ec") == [0.3, 0.5, 0.9],
    "B-1 EC history が時系列順 append",
)
recon_hist = state_b["prediction_error_history_by_source"].get("reconciliation", [])
other_hist = state_b["prediction_error_history_by_source"].get("other_layer", [])
_assert(len(recon_hist) == 2, f"B-2 reconciliation 2 件 (got {len(recon_hist)})")
_assert(len(other_hist) == 1, f"B-3 other_layer 1 件 (got {len(other_hist)})")
_assert(
    [r["magnitude"] for r in recon_hist] == [0.3, 0.5],
    "B-4 source 別も時系列順",
)


# =========================================================================
# Section C: history_max trim (FIFO)
# =========================================================================
print("=== Section C: history_max trim ===")

state_c = {}
for i in range(15):
    record_ec_prediction_error(
        state_c, source="test_src",
        magnitude=float(i) / 10.0,
        history_max=10,
    )
src_c = state_c["prediction_error_history_by_source"]["test_src"]
_assert(len(src_c) == 10, f"C-1 history_max=10 で 10 件に trim (got {len(src_c)})")
# FIFO: 最古 (0, 1, 2, 3, 4) が落ちて (5, 6, 7, 8, 9, 10, 11, 12, 13, 14) が残る
_assert(
    src_c[0]["magnitude"] == 0.5,
    f"C-2 最古は 0.5 (0.0-0.4 は trim、got {src_c[0]['magnitude']})",
)
_assert(
    src_c[-1]["magnitude"] == 1.4,
    f"C-3 最新は 1.4 (got {src_c[-1]['magnitude']})",
)
# EC history は trim しない (共通 weight 経路、既存 predictor 流用)
_assert(
    len(state_c["prediction_error_history_ec"]) == 15,
    f"C-4 prediction_error_history_ec は trim しない (got {len(state_c['prediction_error_history_ec'])})",
)


# =========================================================================
# Section D: context / reason optional
# =========================================================================
print("=== Section D: optional 引数 ===")

state_d = {}
record_ec_prediction_error(state_d, source="test", magnitude=0.4)
rec_d = state_d["prediction_error_history_by_source"]["test"][0]
_assert(rec_d["reason"] == "", "D-1 reason 未指定で空文字")
_assert(rec_d["context"] == {}, "D-2 context 未指定で空 dict")
_assert(rec_d["magnitude"] == 0.4, "D-3 magnitude 保存")


# =========================================================================
# Section E: 既存 history がある state に追加可能
# =========================================================================
print("=== Section E: 既存 state への追加 ===")

state_e = {
    "prediction_error_history_ec": [0.1, 0.2],  # 既存
    "prediction_error_history_by_source": {
        "reconciliation": [
            {"magnitude": 0.1, "reason": "old", "context": {}, "timestamp": "old"}
        ],
    },
}
record_ec_prediction_error(state_e, source="reconciliation", magnitude=0.6, reason="new")

_assert(
    state_e["prediction_error_history_ec"] == [0.1, 0.2, 0.6],
    "E-1 既存 EC history に追加",
)
_assert(
    len(state_e["prediction_error_history_by_source"]["reconciliation"]) == 2,
    "E-2 既存 source history に追加 (2 件)",
)


# =========================================================================
# Section F: magnitude 型変換 (int / str float を float に)
# =========================================================================
print("=== Section F: magnitude 型変換 ===")

state_f = {}
record_ec_prediction_error(state_f, source="test", magnitude=1)  # int
rec_f = state_f["prediction_error_history_by_source"]["test"][0]
_assert(rec_f["magnitude"] == 1.0 and isinstance(rec_f["magnitude"], float),
        "F-1 int → float 変換")


# =========================================================================
print("=" * 60)
_pass = sum(1 for r, _ in results if r)
_total = len(results)
print(f"結果: {_pass}/{_total} passed")
for ok, msg in results:
    if not ok:
        print(f"  FAIL: {msg}")
print("=" * 60)

sys.exit(0 if _pass == _total else 1)
