"""test_reconciliation.py — 段階11-B Phase 3 Step 3-C 本体検証 (β 方針、厚め)。

検証対象:
  [A] _parse_contradict_response: JSON robust parse + severity clamp
  [B] _llm_judge_contradiction: LLM mock で矛盾判定
  [C] check_on_write: Tier 1 (entity_name match) 矛盾検出
  [D] check_on_write: 同 content 早期 skip (重複非矛盾)
  [E] check_on_write: Tier 2/3 embedding mock 判定
  [F] check_on_write: LLM error graceful fallback
  [G] memory_store hook: _state なしで発火しない
  [H] memory_store hook: _state 渡しで check_on_write 呼ばれる (mock で確認)
  [I] bitemporal 凍結: 既存 fact 書き換えない (新 fact 追加 + EC 誤差独立記録)
  [J] pressure 接続: prediction_error_history_by_source から pressure 加算

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_reconciliation.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_recon_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

from core.entropy import calc_pressure_signals
from core.memory import memory_store
from core.perspective import default_self_perspective
from core.reconciliation import (
    _llm_judge_contradiction,
    _parse_contradict_response,
    check_on_write,
)
from core.tag_registry import register_standard_tags


register_standard_tags()


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


# =========================================================================
# Section A: _parse_contradict_response robustness
# =========================================================================
print("=== Section A: _parse_contradict_response ===")

p1 = _parse_contradict_response(
    '{"is_contradict": true, "severity": 0.8, "reason": "直接対立"}'
)
_assert(p1["is_contradict"] is True, "A-1 is_contradict=True 抽出")
_assert(p1["severity"] == 0.8, "A-2 severity=0.8 抽出")
_assert(p1["reason"] == "直接対立", "A-3 reason 抽出")

p2 = _parse_contradict_response('前置き\n{"is_contradict": false, "severity": 0.0, "reason": "無関係"}\n後置き')
_assert(p2["is_contradict"] is False, "A-4 前後文付き → JSON 抽出可能")

p3 = _parse_contradict_response('not a json')
_assert(
    p3["is_contradict"] is False and p3["severity"] == 0.0,
    "A-5 非 JSON で graceful 空返却",
)

p4 = _parse_contradict_response('{"is_contradict": true, "severity": 1.8, "reason": "clamp test"}')
_assert(p4["severity"] == 1.0, f"A-6 severity 1.0 clamp (got {p4['severity']})")

p5 = _parse_contradict_response('{"is_contradict": true, "severity": -0.3, "reason": "neg clamp"}')
_assert(p5["severity"] == 0.0, f"A-7 severity 0.0 clamp (got {p5['severity']})")


# =========================================================================
# Section B: _llm_judge_contradiction with mock
# =========================================================================
print("=== Section B: _llm_judge_contradiction ===")

mock_responses = iter([
    '{"is_contradict": true, "severity": 0.7, "reason": "逆の主張"}',
    '{"is_contradict": false, "severity": 0.1, "reason": "同じ系統の主張"}',
])


def _mock_llm(prompt, max_tokens=200, temperature=0.2):
    return next(mock_responses)


new_e = {"content": "チョコは嫌い"}
existing_e = {"content": "チョコが大好き"}
v1 = _llm_judge_contradiction(new_e, existing_e, llm_call_fn=_mock_llm)
_assert(v1["is_contradict"] is True and v1["severity"] == 0.7, "B-1 mock 応答 1: 矛盾検出")

v2 = _llm_judge_contradiction(new_e, existing_e, llm_call_fn=_mock_llm)
_assert(
    v2["is_contradict"] is False and v2["severity"] == 0.1,
    "B-2 mock 応答 2: 非矛盾",
)


def _mock_llm_raise(prompt, max_tokens=200, temperature=0.2):
    raise RuntimeError("mock llm error")


v3 = _llm_judge_contradiction(new_e, existing_e, llm_call_fn=_mock_llm_raise)
_assert(
    v3["is_contradict"] is False and v3["severity"] == 0.0,
    "B-3 LLM error で graceful fallback (非矛盾扱い)",
)


# =========================================================================
# Section C: check_on_write — Tier 1 entity_name match
# =========================================================================
print("=== Section C: check_on_write Tier 1 ===")

existing_yuu = memory_store(
    "entity", "チョコが大好き", {"entity_name": "ゆう"},
    origin="fixture", perspective=default_self_perspective(),
    _auto_metadata=False,
)

new_yuu_contradict = memory_store(
    "entity", "チョコが大嫌い", {"entity_name": "ゆう"},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)

state_c = {}


def _mock_llm_contradict(prompt, max_tokens=200, temperature=0.2):
    return '{"is_contradict": true, "severity": 0.8, "reason": "好嫌反転"}'


contradictions = check_on_write(
    new_yuu_contradict, state_c,
    llm_call_fn=_mock_llm_contradict,
)

_assert(len(contradictions) >= 1, f"C-1 Tier 1 で矛盾検出 ({len(contradictions)} 件)")
# EC history に記録
_assert(
    0.8 in state_c.get("prediction_error_history_ec", []),
    f"C-2 EC history に severity 0.8 append (got {state_c.get('prediction_error_history_ec')})",
)
# source 別 detail
recon_detail = state_c.get("prediction_error_history_by_source", {}).get("reconciliation", [])
_assert(len(recon_detail) >= 1, "C-3 source='reconciliation' の detail 記録")
if recon_detail:
    _assert(recon_detail[0]["magnitude"] == 0.8, "C-4 detail magnitude 0.8")
    _assert("好嫌反転" in recon_detail[0]["reason"], "C-5 detail reason 保存")
    _assert(recon_detail[0]["context"].get("tier") == 1, "C-6 context.tier=1")


# =========================================================================
# Section D: 同 content は early skip (重複ではあるが矛盾ではない)
# =========================================================================
print("=== Section D: 同 content skip ===")

_ = memory_store(
    "entity", "重複 content", {"entity_name": "dup_ent"},
    origin="fixture", perspective=default_self_perspective(),
    _auto_metadata=False,
)
new_dup = memory_store(
    "entity", "重複 content", {"entity_name": "dup_ent"},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)
state_d = {}
_mock_called = []


def _mock_llm_d(prompt, max_tokens=200, temperature=0.2):
    _mock_called.append(True)
    return '{"is_contradict": true, "severity": 1.0, "reason": "should not be called"}'


c_d = check_on_write(new_dup, state_d, llm_call_fn=_mock_llm_d)
_assert(len(c_d) == 0, "D-1 同 content で矛盾検出ゼロ (early skip)")
_assert(len(_mock_called) == 0, "D-2 LLM judge は同 content で呼ばれない (cost 抑制)")


# =========================================================================
# Section E: Tier 2 embedding mock
# =========================================================================
print("=== Section E: Tier 2 embedding mock ===")

_ = memory_store(
    "opinion", "A は正しい", {"confidence": 0.7},
    origin="fixture", perspective=default_self_perspective(),
    _auto_metadata=False,
)
new_op = memory_store(
    "opinion", "A は誤っている", {"confidence": 0.6},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)


def _mock_embed_same(texts):
    return [[1.0, 0.0, 0.0] for _ in texts]


def _mock_cos(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _mock_llm_tier2(prompt, max_tokens=200, temperature=0.2):
    return '{"is_contradict": true, "severity": 0.6, "reason": "semantic opposition"}'


state_e = {}
c_e = check_on_write(
    new_op, state_e,
    embed_fn=_mock_embed_same, cosine_fn=_mock_cos,
    llm_call_fn=_mock_llm_tier2,
)
_t2_hits = [(c, t, v) for c, t, v in c_e if t == 2]
_assert(len(_t2_hits) >= 1, f"E-1 Tier 2 矛盾検出 ({len(_t2_hits)} 件)")
_assert(
    0.6 in state_e.get("prediction_error_history_ec", []),
    "E-2 Tier 2 も EC history に記録",
)


# =========================================================================
# Section F: LLM error graceful fallback
# =========================================================================
print("=== Section F: LLM error graceful ===")

existing_yuu2 = memory_store(
    "entity", "コーヒー党", {"entity_name": "ゆう_coffee"},
    origin="fixture", perspective=default_self_perspective(),
    _auto_metadata=False,
)
new_yuu2 = memory_store(
    "entity", "紅茶党", {"entity_name": "ゆう_coffee"},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)
state_f = {}


def _mock_llm_raise_f(prompt, max_tokens=200, temperature=0.2):
    raise RuntimeError("F test llm fail")


c_f = check_on_write(new_yuu2, state_f, llm_call_fn=_mock_llm_raise_f)
_assert(len(c_f) == 0, "F-1 LLM error で矛盾検出ゼロ (graceful)")
_assert(
    state_f.get("prediction_error_history_ec", []) == [],
    "F-2 LLM error で history 追記なし (severity 0.0 は記録しない)",
)


# =========================================================================
# Section G: memory_store hook — _state=None で発火しない
# =========================================================================
print("=== Section G: memory_store hook、_state=None で no-op ===")

_orig_check = __import__("core.reconciliation", fromlist=["check_on_write"]).check_on_write
_check_called = []


def _mock_check(new_entry, state, *, embed_fn=None, cosine_fn=None, llm_call_fn=None, limit=50):
    _check_called.append(new_entry.get("id", ""))
    return []


import core.reconciliation as _rec
_rec.check_on_write = _mock_check  # memory.py 内 import で参照される
# memory.py は reconciliation を lazy import なので、module 側で差し替え有効

try:
    _e_g = memory_store(
        "opinion", "hook テスト", {"confidence": 0.5},
        origin="test", perspective=default_self_perspective(),
        _auto_metadata=False,
        # _state=None (未指定)
    )
    _assert(len(_check_called) == 0, f"G-1 _state=None で check_on_write 呼ばれない (called {len(_check_called)})")
finally:
    _rec.check_on_write = _orig_check


# =========================================================================
# Section H: memory_store hook — _state 渡しで発火
# =========================================================================
print("=== Section H: memory_store hook、_state 渡しで発火 ===")

_check_called2 = []


def _mock_check_h(new_entry, state, *, embed_fn=None, cosine_fn=None, llm_call_fn=None, limit=50):
    _check_called2.append((new_entry.get("id", ""), id(state)))
    return []


_rec.check_on_write = _mock_check_h
state_h = {"marker": "h_state"}

try:
    _e_h = memory_store(
        "entity", "hook 発火 test", {"entity_name": "hook_ent"},
        origin="test", perspective=default_self_perspective(),
        _auto_metadata=False,
        _state=state_h,
    )
    _assert(len(_check_called2) == 1, f"H-1 _state 渡しで check_on_write 1 回呼出 (called {len(_check_called2)})")
    if _check_called2:
        _assert(
            _check_called2[0][0] == _e_h.get("id"),
            "H-2 hook に new_entry が渡される",
        )
        _assert(
            _check_called2[0][1] == id(state_h),
            "H-3 hook に state が渡される (同一オブジェクト)",
        )
finally:
    _rec.check_on_write = _orig_check


# =========================================================================
# Section I: bitemporal 凍結 — 既存 fact は書き換えない
# =========================================================================
print("=== Section I: bitemporal 凍結 ===")

existing_i = memory_store(
    "entity", "既存 fact (凍結対象)", {"entity_name": "bitemporal_test"},
    origin="fixture", perspective=default_self_perspective(),
    _auto_metadata=False,
)
_existing_id = existing_i.get("id")
_existing_content = existing_i.get("content")

# 矛盾する新 fact + state 渡し (実 reconciliation 発火)
state_i = {}


def _mock_llm_i(prompt, max_tokens=200, temperature=0.2):
    return '{"is_contradict": true, "severity": 0.7, "reason": "凍結 test"}'


new_i = memory_store(
    "entity", "矛盾する新 fact", {"entity_name": "bitemporal_test"},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
    _state=state_i,
    _reconcile_llm_fn=_mock_llm_i,
)

# 既存 entry が jsonl で書き換わってないか
from core.memory import list_records
recs_i = list_records("entity", limit=50)
_found_existing = [r for r in recs_i if r.get("id") == _existing_id]
_assert(len(_found_existing) == 1, "I-1 既存 entry が jsonl に残ってる")
if _found_existing:
    _assert(
        _found_existing[0].get("content") == _existing_content,
        "I-2 既存 content が書き換わっていない (bitemporal 凍結)",
    )
# 新 fact は独立に追加
_found_new = [r for r in recs_i if r.get("id") == new_i.get("id")]
_assert(len(_found_new) == 1, "I-3 新 fact は通常通り追加")
# EC 誤差は独立記録
_assert(
    state_i.get("prediction_error_history_ec", []) != [],
    "I-4 矛盾は EC 誤差として独立記録",
)


# =========================================================================
# Section J: pressure 接続 — reconciliation history が signals["prediction_error"] に加算
# =========================================================================
print("=== Section J: pressure 接続 ===")

# ベースライン: reconciliation history なしの pressure
state_j_base = {"last_prediction_error": 20.0}  # 0-100 scale、0.2 に正規化
sig_base = calc_pressure_signals(state_j_base)
pe_base = sig_base["prediction_error"]

# reconciliation history あり
state_j_recon = {
    "last_prediction_error": 20.0,
    "prediction_error_history_by_source": {
        "reconciliation": [
            {"magnitude": 0.5, "reason": "", "context": {}, "timestamp": ""},
            {"magnitude": 0.3, "reason": "", "context": {}, "timestamp": ""},
        ],
    },
}
sig_recon = calc_pressure_signals(state_j_recon)
pe_recon = sig_recon["prediction_error"]

_assert(
    pe_recon > pe_base,
    f"J-1 reconciliation history で pressure 増加 (base={pe_base:.4f} → recon={pe_recon:.4f})",
)

# 5 件以上の history → 直近 5 件 mean (list は append-order で [-5:] が最新)
state_j_five = {
    "last_prediction_error": 0.0,
    "prediction_error_history_by_source": {
        "reconciliation": [
            {"magnitude": 0.0, "reason": "", "context": {}, "timestamp": ""},  # 最古 (window 外)
            {"magnitude": 1.0, "reason": "", "context": {}, "timestamp": ""},
            {"magnitude": 1.0, "reason": "", "context": {}, "timestamp": ""},
            {"magnitude": 1.0, "reason": "", "context": {}, "timestamp": ""},
            {"magnitude": 1.0, "reason": "", "context": {}, "timestamp": ""},
            {"magnitude": 1.0, "reason": "", "context": {}, "timestamp": ""},  # 最新
        ],
    },
}
sig_five = calc_pressure_signals(state_j_five)
# 直近 5 件 all 1.0 → mean 1.0 → combined_pe = min(1.0, 0 + 1.0) = 1.0
# w_prediction_error 流用で signals = 1.0 * w_prediction_error
from core.entropy import ENTROPY_PARAMS
_expected_max = 1.0 * ENTROPY_PARAMS["w_prediction_error"]
_assert(
    abs(sig_five["prediction_error"] - _expected_max) < 1e-6,
    f"J-2 直近 5 件 mean=1.0 で pressure = w_prediction_error (got {sig_five['prediction_error']:.4f}, expected {_expected_max:.4f})",
)

# cap 1.0 — base_pe=0.5 + recon_pe=0.8 → combined 1.0 (1.3 にならない)
state_j_cap = {
    "last_prediction_error": 50.0,  # 0-100 → 0.5
    "prediction_error_history_by_source": {
        "reconciliation": [
            {"magnitude": 0.8, "reason": "", "context": {}, "timestamp": ""},
        ],
    },
}
sig_cap = calc_pressure_signals(state_j_cap)
# combined_pe = min(1.0, 0.5 + 0.8) = 1.0
_expected_cap = 1.0 * ENTROPY_PARAMS["w_prediction_error"]
_assert(
    abs(sig_cap["prediction_error"] - _expected_cap) < 1e-6,
    f"J-3 combined_pe cap 1.0 (got {sig_cap['prediction_error']:.4f}, expected {_expected_cap:.4f})",
)


# =========================================================================
print("=" * 60)
_pass = sum(1 for r, _ in results if r)
_total = len(results)
print(f"結果: {_pass}/{_total} passed")
for ok, msg in results:
    if not ok:
        print(f"  FAIL: {msg}")
print("=" * 60)

try:
    shutil.rmtree(_tmp_root, ignore_errors=True)
except Exception:
    pass

sys.exit(0 if _pass == _total else 1)
