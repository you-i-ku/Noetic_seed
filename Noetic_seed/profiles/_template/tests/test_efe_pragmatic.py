"""Slice 6.5 Step 3b: pragmatic_gain 3 成分 (Lv3 nat 単位) の識別力 test。

PLAN reference: WORLD_MODEL_DESIGN/INFO_GAIN_EFE_REDESIGN_PLAN.md §4.2 + §6.3 (v1.1)

各成分の想定誤実装と識別力 fixture (CLAUDE.md §5 literal + PLAN §6.3 literal):
  - effective_change (§4.2.1): log p*(o_t | C_t) - log p*(o_{t-1} | C_{t-1})
      想定誤実装: e1 diff のまま、preference C を読まない → 旧 [0,1] 値域
  - competence (§4.2.2): β·(Var[Q]_prev - Var[Q]_now)、β=1.0
      想定誤実装: max(0, ·) clamp で負値潰し → variance 拡大時 旧 0 vs 新 負値
  - tool_diversity (§4.2.3): H_now[tool] - H_prev[tool] (Shannon entropy)
      想定誤実装: count diff のまま、entropy 計算しない → 旧 [0, N] 値域
"""
import math

from core.info_gain import (
    EPS_LOG,
    _competence_gain,
    _effective_change_gain,
    _tool_diversity_gain,
)


# ========= effective_change (§4.2.1) test 4 件 =========


def test_effective_change_c_not_built_graceful():
    """case A (§6.3.5): C 未構築 (state._efe_C=None) → 0 graceful。"""
    state = {"_efe_C": None}
    prev = {}
    entries = [{"embedding": [0.1] * 1024}]
    assert _effective_change_gain(state, prev, entries) == 0.0


def test_effective_change_empty_components_graceful():
    """case B (§6.3.5): C empty components (n=0) → 0 graceful。"""
    state = {"_efe_C": {"components": []}}
    prev = {}
    entries = [{"embedding": [0.1] * 1024}]
    assert _effective_change_gain(state, prev, entries) == 0.0


def test_effective_change_no_prev_log_p_graceful():
    """case C (§6.3.5、初 cycle): prev に effective_change_log_p なし → 0 graceful。

    新 entry はあるが前 cycle 値がないので diff 計算不能、0 で graceful。
    """
    # ダミー C (1 component) を構築
    state = {
        "_efe_C": {
            "components": [
                {"key": "id", "mean": [1.0] + [0.0] * 1023,
                 "variance": 1.0, "weight": 1.0}
            ]
        },
    }
    prev = {}  # effective_change_log_p なし
    entries = [{"embedding": [1.0] + [0.0] * 1023}]
    assert _effective_change_gain(state, prev, entries) == 0.0


def test_effective_change_typical_diff_log_scale():
    """case D (§6.3.5): typical case (C 構築済 + prev 値あり) → log scale diff。

    識別力 (想定誤実装 fail): 旧 e1 [0,1] 差分 vs 新 log scale 差分、規模で識別。
    """
    state = {
        "_efe_C": {
            "components": [
                {"key": "id", "mean": [1.0] + [0.0] * 1023,
                 "variance": 1.0, "weight": 1.0}
            ]
        },
    }
    prev = {"effective_change_log_p": 100.0}  # 前 cycle の値、適当な値
    # 新 entry が mean とほぼ一致 → log_p_now が大きい
    entries = [{"embedding": [1.0] + [0.0] * 1023}]
    result = _effective_change_gain(state, prev, entries)
    # log_p_now (typical vMF) は数百 nat、prev 100、diff は数百規模
    assert abs(result) > 1.0  # 旧 [0,1] 値域と区別 (nat scale)


# ========= competence (§4.2.2) test 4 件 =========


def test_competence_variance_decrease_positive():
    """case A (§6.3.6): variance 縮小 → β·(prev - now) > 0 (旧 max(0, ·) と同符号)。"""
    state = {
        "predictor_confidence": {
            "t1": {"success": 5, "fail": 0},  # rate=1.0
            "t2": {"success": 3, "fail": 1},  # rate=0.75
            "t3": {"success": 4, "fail": 0},  # rate=1.0
        }
    }
    # var_now ≈ statistics.variance([1.0, 0.75, 1.0]) ≈ 0.0208
    prev = {"predictor_success_rate_var": 0.5}  # 大きい variance (高 uncertainty)
    result = _competence_gain(state, prev)
    # β=1.0 で result = (0.5 - 0.0208) ≈ 0.48
    assert result > 0.3  # 正、competence 上昇


def test_competence_variance_unchanged_zero():
    """case B (§6.3.6): variance 不変 → 0 nat。"""
    state = {
        "predictor_confidence": {
            "t1": {"success": 5, "fail": 5},
            "t2": {"success": 5, "fail": 5},
        }
    }
    # var_now = 0 (rates 全部 0.5)
    prev = {"predictor_success_rate_var": 0.0}
    assert abs(_competence_gain(state, prev)) < 1e-6


def test_competence_variance_increase_negative():
    """case C (§6.3.6、★ Lv3 negation 保持): variance 拡大 → 負値 (旧 0 clamp、識別)。

    想定誤実装 fail: 旧 max(0, var_prev - var_now) は variance 拡大で 0、新は負値。
    """
    state = {
        "predictor_confidence": {
            "t1": {"success": 9, "fail": 1},  # rate=0.9
            "t2": {"success": 1, "fail": 9},  # rate=0.1
            "t3": {"success": 5, "fail": 5},  # rate=0.5
        }
    }
    # var_now ≈ statistics.variance([0.9, 0.1, 0.5]) ≈ 0.16
    prev = {"predictor_success_rate_var": 0.01}  # 小さい variance
    result = _competence_gain(state, prev)
    # β=1.0 で result = (0.01 - 0.16) ≈ -0.15
    assert result < -0.05  # 負、旧 0 clamp と区別


def test_competence_predictor_insufficient_graceful():
    """case D (§6.3.6): predictor 不足 (attempts < MIN) → var_now=0 graceful。"""
    state = {
        "predictor_confidence": {
            "t1": {"success": 0, "fail": 0},  # attempts=0
        }
    }
    # var_now = 0 (success_rates 空)
    prev = {"predictor_success_rate_var": 0.0}
    result = _competence_gain(state, prev)
    assert result == 0.0


# ========= tool_diversity (§4.2.3) test 4 件 =========


def test_tool_diversity_new_tool_positive():
    """case A (§6.3.7): 新 tool 試行で entropy 増 → 正。"""
    state = {
        "action_ledger": [
            {"tool": "a"}, {"tool": "a"},
            {"tool": "b"}, {"tool": "b"}, {"tool": "c"},
        ]
    }
    # H_now = entropy of [2, 2, 1]/5 ≈ -((2/5)·log(2/5+ε) + (2/5)·log(2/5+ε) + (1/5)·log(1/5+ε))
    #       ≈ -(0.4·log(0.401) + 0.4·log(0.401) + 0.2·log(0.201)) ≈ 1.055
    prev = {"tool_entropy": 0.5}  # 前 cycle の entropy 低
    result = _tool_diversity_gain(state, prev)
    assert result > 0.3  # entropy 増、正


def test_tool_diversity_consecutive_negative():
    """case B (§6.3.7、★ Lv3 negation 保持): 同 tool 連発で entropy 減 → 負 (旧 clamp で 0)。

    想定誤実装 fail: 旧 max(0, count_diff) は連発時 0、新は負値で連発状態捕捉。
    """
    state = {
        "action_ledger": [
            {"tool": "a"}, {"tool": "a"}, {"tool": "a"}, {"tool": "a"}, {"tool": "a"},
        ]
    }
    # H_now = entropy of [5]/5 = -1.0·log(1.001) ≈ -0.001 ≈ 0
    prev = {"tool_entropy": 1.0}  # 前 cycle entropy 高
    result = _tool_diversity_gain(state, prev)
    assert result < -0.5  # 負、旧 0 clamp と区別


def test_tool_diversity_unchanged_zero():
    """case C (§6.3.7): 分布不変 → 0 nat (両実装一致)。"""
    state = {
        "action_ledger": [
            {"tool": "a"}, {"tool": "b"}, {"tool": "a"}, {"tool": "b"}, {"tool": "a"},
        ]
    }
    # H_now = entropy of [3, 2]/5 ≈ 0.673
    # prev も同じ entropy
    H_expected = -(
        (3 / 5) * math.log(3 / 5 + EPS_LOG) + (2 / 5) * math.log(2 / 5 + EPS_LOG)
    )
    prev = {"tool_entropy": H_expected}
    result = _tool_diversity_gain(state, prev)
    assert abs(result) < 1e-3  # 不変、識別力で 0 近傍


def test_tool_diversity_empty_ledger_graceful():
    """case D (§6.3.7): 空 ledger → 0 graceful (両実装一致)。"""
    state = {"action_ledger": []}
    prev = {"tool_entropy": 0.0}
    result = _tool_diversity_gain(state, prev)
    assert result == 0.0
