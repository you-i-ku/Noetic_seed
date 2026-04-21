"""段階10 柱 C: controller._predicted_outcome_multiplier 正規化加重テスト。

STAGE10 PLAN v1 §3-C の要件:
  - signature: (prediction, candidate, state, cfg) -> float
  - predicted_ec あり + 正常 conf:
    combined = (pe2_ratio * e2_conf + pec * ec_conf) / (e2_conf + ec_conf)
  - predicted_ec 欠如 (Light fallback 等): pe2_ratio のみ (段階9 挙動)
  - 両 conf ゼロ: pass-through 1.0 (predictor 層 bypass)
  - combined < 0.4: candidate["penalties"] に "low_outcome=..." 追記
  - floor = cfg.get("predicted_e2_floor", 0.05)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_controller_ec_weighted.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.controller import _predicted_outcome_multiplier


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


CFG = {"predicted_e2_floor": 0.05}


# ============================================================
# predicted_ec あり: 正規化加重
# ============================================================

def test_weighted_both_present_default_conf():
    print("== pe2 + pec 併記、conf default (0.7/0.7) で正規化加重 ==")
    # default 0.7/0.7 → 等加重 (pe2 + pec) / 2
    # pe2=80, pec=0.4 → (0.8 * 0.7 + 0.4 * 0.7) / 1.4 = (0.56 + 0.28) / 1.4 = 0.6
    pred = {"predicted_e2": 80, "predicted_ec": 0.4}
    cand = {"tool": "output_display"}
    state = {}  # predictor_confidence 未 init → default 0.7/0.7
    mult = _predicted_outcome_multiplier(pred, cand, state, CFG)
    return _assert(abs(mult - 0.6) < 1e-9, f"(0.8*0.7 + 0.4*0.7)/1.4 = 0.6 (actual: {mult})")


def test_weighted_e2_conf_dominant():
    print("== e2_conf 高 / ec_conf 低 → pe2 寄りに加重 ==")
    # e2_conf=0.9, ec_conf=0.1 → pe2 が支配的
    # pe2=0.8, pec=0.3 → (0.8*0.9 + 0.3*0.1) / 1.0 = 0.75
    pred = {"predicted_e2": 80, "predicted_ec": 0.3}
    cand = {"tool": "tool_A"}
    state = {"predictor_confidence": {"tool_A": {"e2_conf": 0.9, "ec_conf": 0.1}}}
    mult = _predicted_outcome_multiplier(pred, cand, state, CFG)
    return _assert(abs(mult - 0.75) < 1e-9, f"(0.72 + 0.03)/1.0 = 0.75 (actual: {mult})")


def test_weighted_ec_conf_dominant():
    print("== ec_conf 高 / e2_conf 低 → pec 寄りに加重 (E2 循環性緩和) ==")
    # e2_conf=0.2, ec_conf=0.8 → pec が支配的
    # pe2=0.95, pec=0.3 → (0.95*0.2 + 0.3*0.8) / 1.0 = 0.19 + 0.24 = 0.43
    pred = {"predicted_e2": 95, "predicted_ec": 0.3}
    cand = {"tool": "tool_B"}
    state = {"predictor_confidence": {"tool_B": {"e2_conf": 0.2, "ec_conf": 0.8}}}
    mult = _predicted_outcome_multiplier(pred, cand, state, CFG)
    return _assert(abs(mult - 0.43) < 1e-9, f"(0.19 + 0.24)/1.0 = 0.43 (actual: {mult})")


# ============================================================
# predicted_ec 欠如: 段階9 挙動維持
# ============================================================

def test_no_pec_backward_compat():
    print("== predicted_ec 欠如 → pe2_ratio のみ (段階9 挙動) ==")
    pred = {"predicted_e2": 70}  # predicted_ec なし
    cand = {"tool": "output_display"}
    state = {}
    mult = _predicted_outcome_multiplier(pred, cand, state, CFG)
    return _assert(abs(mult - 0.7) < 1e-9, f"pe2=70 → 0.7 (actual: {mult})")


def test_no_pec_ignores_conf_weights():
    print("== predicted_ec 欠如時は conf 加重も効かない (段階9 挙動維持) ==")
    pred = {"predicted_e2": 70}
    cand = {"tool": "tool_C"}
    state = {"predictor_confidence": {"tool_C": {"e2_conf": 0.2, "ec_conf": 0.8}}}
    mult = _predicted_outcome_multiplier(pred, cand, state, CFG)
    # conf が極端でも pec 無いなら pe2 のみ
    return _assert(abs(mult - 0.7) < 1e-9, f"conf 無視、pe2=70 → 0.7 (actual: {mult})")


# ============================================================
# 両 conf ゼロ: pass-through 1.0
# ============================================================

def test_both_conf_zero_passthrough():
    print("== 両 conf ゼロ → pass-through 1.0 (predictor 層 bypass) ==")
    pred = {"predicted_e2": 20, "predicted_ec": 0.1}
    cand = {"tool": "tool_D"}
    state = {"predictor_confidence": {"tool_D": {"e2_conf": 0.0, "ec_conf": 0.0}}}
    mult = _predicted_outcome_multiplier(pred, cand, state, CFG)
    return _assert(mult == 1.0, f"両 conf=0 → 1.0 (actual: {mult})")


# ============================================================
# penalty / floor
# ============================================================

def test_low_outcome_penalty():
    print("== combined < 0.4 で penalty 追記 ==")
    # pe2=20, pec=0.3, default 0.7/0.7 → (0.2*0.7 + 0.3*0.7) / 1.4 = 0.25
    pred = {"predicted_e2": 20, "predicted_ec": 0.3}
    cand = {"tool": "output_display"}
    state = {}
    _ = _predicted_outcome_multiplier(pred, cand, state, CFG)
    penalties = cand.get("penalties", [])
    return all([
        _assert(any("low_outcome" in p for p in penalties), f"penalty 追記 (actual: {penalties})"),
    ])


def test_floor_applied():
    print("== combined が floor 未満なら floor に clamp ==")
    # pe2=0, pec=0, default 0.7/0.7 → 0.0 combined → floor 0.05
    pred = {"predicted_e2": 0, "predicted_ec": 0.0}
    cand = {"tool": "output_display"}
    state = {}
    mult = _predicted_outcome_multiplier(pred, cand, state, CFG)
    return _assert(mult == 0.05, f"floor 0.05 適用 (actual: {mult})")


def test_no_penalty_high_outcome():
    print("== combined >= 0.4 は penalty なし ==")
    pred = {"predicted_e2": 70, "predicted_ec": 0.6}
    cand = {"tool": "output_display"}
    state = {}
    _ = _predicted_outcome_multiplier(pred, cand, state, CFG)
    penalties = cand.get("penalties", [])
    return _assert(not any("low_outcome" in p for p in penalties), "penalty なし")


# ============================================================
# 型堅牢性
# ============================================================

def test_invalid_pec_fallback():
    print("== 不正 pec (文字列) → fallback 0.5 ==")
    pred = {"predicted_e2": 50, "predicted_ec": "abc"}  # type: ignore
    cand = {"tool": "output_display"}
    state = {}
    mult = _predicted_outcome_multiplier(pred, cand, state, CFG)
    # pe2=50 (0.5), pec fallback=0.5, default 0.7/0.7 → (0.5*0.7 + 0.5*0.7)/1.4 = 0.5
    return _assert(abs(mult - 0.5) < 1e-9, f"不正 pec → 0.5 fallback、combined=0.5 (actual: {mult})")


def test_invalid_pe2_fallback():
    print("== 不正 pe2 → neutral 50 扱い ==")
    pred = {"predicted_e2": "foo", "predicted_ec": 0.5}  # type: ignore
    cand = {"tool": "output_display"}
    state = {}
    mult = _predicted_outcome_multiplier(pred, cand, state, CFG)
    # pe2 fallback=50 (0.5), pec=0.5, default → (0.35 + 0.35)/1.4 = 0.5
    return _assert(abs(mult - 0.5) < 1e-9, f"不正 pe2 → neutral、combined=0.5 (actual: {mult})")


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    tests = [
        test_weighted_both_present_default_conf,
        test_weighted_e2_conf_dominant,
        test_weighted_ec_conf_dominant,
        test_no_pec_backward_compat,
        test_no_pec_ignores_conf_weights,
        test_both_conf_zero_passthrough,
        test_low_outcome_penalty,
        test_floor_applied,
        test_no_penalty_high_outcome,
        test_invalid_pec_fallback,
        test_invalid_pe2_fallback,
    ]
    results = [t() for t in tests]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
