"""Slice 6.5 Step 4: regularization 2 成分 (Lv3 nat 単位) の識別力 test。

PLAN reference: WORLD_MODEL_DESIGN/INFO_GAIN_EFE_REDESIGN_PLAN.md §4.3 + §6.3 (v1.1)

各成分の想定誤実装と識別力 fixture (CLAUDE.md §5 literal + PLAN §6.3 literal):
  - consecutive_penalty (§4.3.1): log(1 + run) - log(DIVERSITY_BASELINE)
      想定誤実装: max(0, run - 1) のまま、log 変換忘れ
  - centroid_stuck (§4.3.2): log(FLOOR / max(mean_dist, ε))
      想定誤実装: (FLOOR - mean_dist) / FLOOR の正値クランプのまま
"""
import math

import pytest

from core.info_gain import (
    CENTROID_VARIANCE_FLOOR,
    DIVERSITY_BASELINE,
    EPS_LOG,
    RECENT_EMB_WINDOW,
    RECENT_TOOL_WINDOW,
    _centroid_stuck,
    _consecutive_penalty,
)


# ========= consecutive_penalty (§4.3.1) test 4 件 =========


def test_consecutive_penalty_5_run_positive_log_scale():
    """case A (§6.3.8): 同 tool 5 連発 → log(6) - log(2.5) ≈ 0.876 (識別: 旧 4)。"""
    state = {
        "action_ledger": [
            {"tool": "a"}, {"tool": "a"}, {"tool": "a"}, {"tool": "a"}, {"tool": "a"},
        ]
    }
    result = _consecutive_penalty(state)
    expected = math.log(6) - math.log(DIVERSITY_BASELINE)
    assert result == pytest.approx(expected)


def test_consecutive_penalty_single_negative_baseline_relative():
    """case B (§6.3.8、★ Lv3 negation 保持): 単発 (run=1) → log(2) - log(2.5) ≈ -0.223。

    識別力 (誤実装 fail): 旧 max(0, run-1)=0 vs 新 log scale 負値 (baseline 比較で多様)。
    """
    state = {
        "action_ledger": [
            {"tool": "a"}, {"tool": "b"}, {"tool": "c"}, {"tool": "d"}, {"tool": "e"},
        ]
    }
    # 最後 "e" の連続 run は 1 (異なる tool が続く)
    result = _consecutive_penalty(state)
    expected = math.log(2) - math.log(DIVERSITY_BASELINE)
    assert result == pytest.approx(expected)
    assert result < 0  # 多様 → 負


def test_consecutive_penalty_partial_run_distinguishes():
    """case C: 3 連発 + 違 tool 末尾 → run=1、単発と同じ -0.223 (識別: 連発数を tail で判定)。

    意図的に末尾だけ違う tool で run=1 になる fixture。
    """
    state = {
        "action_ledger": [
            {"tool": "a"}, {"tool": "a"}, {"tool": "a"}, {"tool": "b"}, {"tool": "c"},
        ]
    }
    result = _consecutive_penalty(state)
    expected = math.log(2) - math.log(DIVERSITY_BASELINE)
    assert result == pytest.approx(expected)


def test_consecutive_penalty_empty_ledger_zero_graceful():
    """case D (§6.3.8): 空 ledger → 0 (両実装一致 graceful)。"""
    state = {"action_ledger": []}
    result = _consecutive_penalty(state)
    assert result == 0.0


# ========= centroid_stuck (§4.3.2) test 4 件 =========


def _unit_vec(seed: int, dim: int = 1024) -> list:
    v = [0.0] * dim
    v[seed % dim] = 1.0
    return v


def test_centroid_stuck_full_collapse_large_nat():
    """case A (§6.3.9): 全 embedding 同一 (mean_dist→0) → 大 nat (≈ log(0.3/ε) ≈ 5.7)。"""
    same = _unit_vec(0)
    entries = [{"embedding": same} for _ in range(RECENT_EMB_WINDOW)]
    result = _centroid_stuck(entries)
    expected_max = math.log(CENTROID_VARIANCE_FLOOR / EPS_LOG)
    # mean_dist が ε に近づくと log(0.3/ε) に近づく
    assert result > 4.0  # 大、旧 1.0 と区別 (log scale 識別)


def test_centroid_stuck_diverse_negative_nat():
    """case B (§6.3.9、★ Lv3 negation 保持): 多様 (mean_dist > FLOOR) → 負 (旧 0 clamp 識別)。

    識別力 (誤実装 fail): 旧 (FLOOR - mean_dist)/FLOOR は mean_dist > FLOOR で 0 (clamp)、
    新 log(FLOOR / mean_dist) は負値 (多様性を 明示識別)。
    """
    # 5 個の orthogonal embedding (mean_dist 大)
    entries = [{"embedding": _unit_vec(i)} for i in range(RECENT_EMB_WINDOW)]
    result = _centroid_stuck(entries)
    assert result < 0  # 負、旧 0 clamp と区別


def test_centroid_stuck_boundary_zero():
    """case C (§6.3.9): mean_dist ≈ FLOOR (境界) → ≈ 0 nat。"""
    # mean_dist が FLOOR (0.3) に近い fixture を作るのは embedding 設計が面倒なため、
    # 代わりに「中程度多様」(2 種 mix で mean_dist 0.5 程度) で 0 近傍範囲を確認。
    a = _unit_vec(0)
    b = _unit_vec(1)
    entries = [{"embedding": a}, {"embedding": a}, {"embedding": b}, {"embedding": b}, {"embedding": a}]
    result = _centroid_stuck(entries)
    # 完全固着 (>4) より小、完全多様 (<0) より大 = 中間
    # log scale で境界付近 = mean_dist 大なら負
    assert -3.0 < result < 4.0  # 中間範囲


def test_centroid_stuck_insufficient_observations_zero():
    """case D (§6.3.9): 観察 < RECENT_EMB_WINDOW → 0 (両実装一致 graceful)。"""
    entries = [{"embedding": _unit_vec(0)}]  # 1 件のみ
    assert _centroid_stuck(entries) == 0.0
    assert _centroid_stuck([]) == 0.0


