"""段階14 Pure FEP Refinement test (Step A-D)。

STAGE14_PURE_FEP_REFINEMENT_PLAN.md の各 Step §3-4 / §4-4 / §5-4 / §6-4
検証要件に対応。CLAUDE.md §5 識別力 (discrimination) 重視: 想定誤実装で
fail する fixture を意識して書く。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# Step A — Cumulative Information Gain (∑I_t)
# ============================================================

def test_cig_total_monotonic():
    """e2_total / ec_total は abs(error) 累積で単調増加 (5 cycle)。

    識別力: e2_total を sum じゃなく last_value にする誤実装、
    または abs() 抜けで負誤差で減算する誤実装で fail する。
    """
    print("== Step A: ∑I_t 単調増加 (5 cycle) ==")
    from core.predictor import update_cumulative_information_gain

    state = {}
    # 正と負の混合誤差で abs() 確認 (誤実装で abs() 抜けると負方向に減る)
    errors_e2 = [10.0, -8.0, 6.0, -4.0, 2.0]
    errors_ec = [0.5, -0.4, 0.3, -0.2, 0.1]
    expected_e2_totals = [10.0, 18.0, 24.0, 28.0, 30.0]
    expected_ec_totals = [0.5, 0.9, 1.2, 1.4, 1.5]

    actual_e2 = []
    actual_ec = []
    for e2, ec in zip(errors_e2, errors_ec):
        update_cumulative_information_gain(state, e2, ec)
        actual_e2.append(round(state["cumulative_information_gain"]["e2_total"], 4))
        actual_ec.append(round(state["cumulative_information_gain"]["ec_total"], 4))

    return all([
        _assert(actual_e2 == expected_e2_totals,
                f"e2_total 累積 {actual_e2} == {expected_e2_totals}"),
        _assert(actual_ec == expected_ec_totals,
                f"ec_total 累積 {actual_ec} == {expected_ec_totals}"),
        _assert(all(actual_e2[i] <= actual_e2[i+1] for i in range(4)),
                "e2_total 単調非減少"),
    ])


def test_cig_flat_signal_threshold():
    """e2_window_mean が閾値 (5.0 = 0.05 * 100) で flat_signal が切替。

    識別力: 閾値を 0.05 (× 100 抜け) で実装する誤実装、
    または >= で実装する境界誤実装で fail する。
    """
    print("== Step A: flat_signal 閾値切替 ==")
    from core.predictor import update_cumulative_information_gain

    # case 1: 閾値超 (e2 mean = 10.0、明確に flat じゃない)
    state_high = {"prediction_error_history_e2": [10.0] * 10}
    update_cumulative_information_gain(state_high, 10.0, None)
    high_flat = state_high["cumulative_information_gain"]["flat_signal"]

    # case 2: 閾値未満 (e2 mean = 1.0、flat=True)
    state_low = {"prediction_error_history_e2": [1.0] * 10}
    update_cumulative_information_gain(state_low, 1.0, None)
    low_flat = state_low["cumulative_information_gain"]["flat_signal"]

    # case 3: 閾値ぎりぎり (mean = 4.99 → flat=True、5.01 → flat=False)
    state_below = {"prediction_error_history_e2": [4.99] * 10}
    update_cumulative_information_gain(state_below, 4.99, None)
    below_flat = state_below["cumulative_information_gain"]["flat_signal"]

    state_above = {"prediction_error_history_e2": [5.01] * 10}
    update_cumulative_information_gain(state_above, 5.01, None)
    above_flat = state_above["cumulative_information_gain"]["flat_signal"]

    return all([
        _assert(high_flat is False, "mean=10.0 → flat=False"),
        _assert(low_flat is True, "mean=1.0 → flat=True"),
        _assert(below_flat is True, "mean=4.99 (閾値直下) → flat=True"),
        _assert(above_flat is False, "mean=5.01 (閾値直上) → flat=False"),
    ])


def test_cig_flat_streak_increment_and_reset():
    """flat_streak は連続 flat=True で increment、flat=False で 0 にリセット。

    識別力: リセット忘れ実装 (常に increment) で fail。
    また、increment を `+= 1` ではなく代入 (= 1 固定) する誤実装でも fail。
    """
    print("== Step A: flat_streak increment / リセット ==")
    from core.predictor import update_cumulative_information_gain

    state = {}
    # 連続 flat: error 1.0 (mean=1.0 < 5.0) を 3 回
    streaks = []
    for _ in range(3):
        state.setdefault("prediction_error_history_e2", []).append(1.0)
        update_cumulative_information_gain(state, 1.0, None)
        streaks.append(state["cumulative_information_gain"]["flat_streak"])

    # リセット: 大誤差 50.0 (mean が 1.0 から上昇、5.0 超えれば flat=False)
    state["prediction_error_history_e2"].extend([50.0] * 10)  # window 全部 50
    update_cumulative_information_gain(state, 50.0, None)
    after_reset = state["cumulative_information_gain"]["flat_streak"]
    after_reset_signal = state["cumulative_information_gain"]["flat_signal"]

    # 再度 flat に戻す
    state["prediction_error_history_e2"] = [1.0] * 10
    update_cumulative_information_gain(state, 1.0, None)
    after_recover = state["cumulative_information_gain"]["flat_streak"]

    return all([
        _assert(streaks == [1, 2, 3], f"連続 flat で increment {streaks} == [1,2,3]"),
        _assert(after_reset_signal is False, "大誤差で flat_signal=False"),
        _assert(after_reset == 0, f"flat=False で streak リセット ({after_reset} == 0)"),
        _assert(after_recover == 1, "リセット後 再度 flat で streak=1"),
    ])


# ============================================================
# main runner
# ============================================================

def main():
    tests = [
        test_cig_total_monotonic,
        test_cig_flat_signal_threshold,
        test_cig_flat_streak_increment_and_reset,
    ]
    print(f"Running {len(tests)} test groups (Step A)...\n")
    passed = 0
    for t in tests:
        if t():
            passed += 1
        print()
    print(f"=== Result: {passed}/{len(tests)} test groups passed ===")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
