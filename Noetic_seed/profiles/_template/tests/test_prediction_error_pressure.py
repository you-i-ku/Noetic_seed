"""段階10 柱 A: prediction_error → pressure 加算 テスト。

STAGE10 PLAN v1 §3-A の要件:
  - ENTROPY_PARAMS に w_prediction_error 追加、初期 1.0
  - calc_pressure_signals が signals["prediction_error"] を返す
  - signals["prediction_error"] = (last_prediction_error / 100) * w_prediction_error
  - state に last_prediction_error が無いとき 0 扱いで落ちない
  - 既存 signal (entropy/surprise/...) が壊れてない (退行ゼロ確認)

main.py の state["last_prediction_error"] 書き込み経路は integration smoke で確認する
(unit test のスコープ外)。

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_prediction_error_pressure.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.entropy import ENTROPY_PARAMS, calc_pressure_signals


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# ENTROPY_PARAMS
# ============================================================

def test_w_prediction_error_in_params():
    print("== ENTROPY_PARAMS: w_prediction_error 追加 ==")
    return all([
        _assert("w_prediction_error" in ENTROPY_PARAMS, "w_prediction_error key 存在"),
        _assert(ENTROPY_PARAMS["w_prediction_error"] == 0.3, "Tune 1 後の値 0.3 (段階10.5、smoke で pe fire 90% 占有判定)"),
    ])


# ============================================================
# calc_pressure_signals: prediction_error signal
# ============================================================

def test_signal_key_present():
    print("== calc_pressure_signals: prediction_error key を返す ==")
    state = {"last_prediction_error": 50}
    signals = calc_pressure_signals(state)
    return _assert("prediction_error" in signals, "signals dict に 'prediction_error' キー存在")


def test_signal_value_formula():
    print("== calc_pressure_signals: 値 = (last_pe / 100) * w_prediction_error ==")
    # last_pe=50, w=1.0 → 0.5
    state = {"last_prediction_error": 50}
    signals = calc_pressure_signals(state)
    expected = 0.5 * ENTROPY_PARAMS["w_prediction_error"]
    return _assert(
        abs(signals["prediction_error"] - expected) < 1e-9,
        f"last_pe=50 → {expected} (actual: {signals['prediction_error']})",
    )


def test_signal_value_zero_when_no_error():
    print("== calc_pressure_signals: last_prediction_error 無 → 0 扱い、落ちない ==")
    state = {}  # last_prediction_error キーなし
    try:
        signals = calc_pressure_signals(state)
    except Exception as e:
        print(f"  [FAIL] 例外発生: {e}")
        return False
    return _assert(
        signals["prediction_error"] == 0.0,
        f"last_pe 未設定 → 0.0 (actual: {signals['prediction_error']})",
    )


def test_signal_value_max():
    print("== calc_pressure_signals: last_pe=100 → 最大値 1.0 * w ==")
    state = {"last_prediction_error": 100}
    signals = calc_pressure_signals(state)
    expected = 1.0 * ENTROPY_PARAMS["w_prediction_error"]
    return _assert(
        abs(signals["prediction_error"] - expected) < 1e-9,
        f"last_pe=100 → {expected} (actual: {signals['prediction_error']})",
    )


def test_signal_float_input():
    print("== calc_pressure_signals: last_prediction_error が float でも OK ==")
    state = {"last_prediction_error": 33.5}
    signals = calc_pressure_signals(state)
    expected = (33.5 / 100.0) * ENTROPY_PARAMS["w_prediction_error"]
    return _assert(
        abs(signals["prediction_error"] - expected) < 1e-9,
        f"last_pe=33.5 → {expected} (actual: {signals['prediction_error']})",
    )


# ============================================================
# 退行ゼロ: 既存 signal が残ってる
# ============================================================

def test_existing_signals_preserved():
    print("== 退行ゼロ: 既存 signal (entropy/surprise/...) が残ってる ==")
    state = {
        "entropy": 0.5,
        "last_e2": 0.4,
        "last_e3": 0.5,
        "last_e4": 0.3,
        "unresolved_external": 0.1,
        "pending": [],
        "last_prediction_error": 20,
    }
    signals = calc_pressure_signals(state)
    expected_keys = {
        "entropy", "surprise", "unresolved", "novelty",
        "unresolved_ext", "pending_burden", "stagnation", "custom",
        "prediction_error",
    }
    actual_keys = set(signals.keys())
    missing = expected_keys - actual_keys
    return all([
        _assert(not missing, f"期待 key 全存在 (missing: {missing})"),
        _assert(isinstance(signals["entropy"], float), "entropy float"),
        _assert(isinstance(signals["prediction_error"], float), "prediction_error float"),
    ])


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    tests = [
        test_w_prediction_error_in_params,
        test_signal_key_present,
        test_signal_value_formula,
        test_signal_value_zero_when_no_error,
        test_signal_value_max,
        test_signal_float_input,
        test_existing_signals_preserved,
    ]
    results = [t() for t in tests]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
