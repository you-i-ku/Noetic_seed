"""Slice 6.5 Step 3a: epistemic_gain 4 成分 (Lv3 nat 単位) の識別力 test。

PLAN reference: WORLD_MODEL_DESIGN/INFO_GAIN_EFE_REDESIGN_PLAN.md §4.1 + §6.3 (v1.1)

各成分の想定誤実装と識別力 fixture (CLAUDE.md §5 literal + PLAN §6.3 literal):
  - novelty (§4.1.1): -log(max_cosine + ε)
      想定誤実装: `1 - max_cosine` のまま、log 変換忘れ → 値が 1/2/3 倍以上違う
  - pe_drop (§4.1.2): log((prev_pe + ε) / (now_pe + ε))
      想定誤実装: max(0, ·) クランプで負値潰し → 旧実装 0 vs 新 負値
  - density (§4.1.3): log((now_density + ε) / (prev_density + ε))
      想定誤実装: raw diff or 正値クランプ → 旧 0 vs 新 負値
  - memory_link (§4.1.4): log((1 + strength_now) / (1 + strength_prev))
      想定誤実装: raw diff のまま → 旧 4.0 vs 新 1.099 (例)
"""
import math

import pytest

from core.info_gain import (
    EPS_LOG,
    _density_gain,
    _memory_link_gain,
    _novelty_gain,
    _pe_drop_gain,
)


# ========= novelty (§4.1.1) test 4 件 =========


def _make_emb(seed: int, dim: int = 1024) -> list:
    """deterministic onehot 単位 vector (test 用)。"""
    v = [0.0] * dim
    v[seed % dim] = 1.0
    return v


def test_novelty_empty_input_returns_zero():
    """case A (§6.3.1): 新 entry 0 件 → 0.0 nat。"""
    assert _novelty_gain([], prev_snapshot={"entries_count": 0}) == 0.0


def test_novelty_identical_existing_near_zero_nat(monkeypatch):
    """case B (§6.3.1): 新 entry が既存と完全一致 (cosine=1) → log(1+ε) ≈ ε 程度 nat。

    識別力 (誤実装 fail): 旧 1 - max_cos = 0 と新 -log(1+ε) ≈ ε ≈ 0.001 は近似同じだが、
    cosine=0.5 ケース (test_C) で完全に値が違うため そこで識別力確保。
    """
    same = _make_emb(42)
    entries = [
        {"embedding": same},  # 新 entry 1 件 (先頭、newest first)
        {"embedding": same},  # 既存 1 件
    ]
    result = _novelty_gain(entries, prev_snapshot={"entries_count": 1})
    # cosine=1 で max_sim=1、-log(1+ε) ≈ -log(1.001) ≈ -0.001、abs ≈ ε
    assert abs(result) < 0.01


def test_novelty_partial_match_nat_scale(monkeypatch):
    """case C (§6.3.1): cosine 中間値 (= 0.5 想定) → 旧 0.5 vs 新 ≈ 0.693 nat (識別)。

    識別力: 旧実装 (1 - max_sim = 0.5) と新実装 (-log(0.501) ≈ 0.691) で値が異なる。
    旧と区別するため、新値が log scale 範囲 (0.6 ≤ value ≤ 0.8) にあることを assert。
    """
    # 2 つの orthogonal でない vector を構築 (cosine = 0.5)
    new_vec = [1.0, 0.0] + [0.0] * 1022
    # 既存: angle=60deg、cosine=0.5 になる単位 vector
    cos60 = 0.5
    sin60 = math.sqrt(1 - cos60**2)
    existing_vec = [cos60, sin60] + [0.0] * 1022
    # normalize 念のため
    norm = math.sqrt(sum(v * v for v in existing_vec))
    existing_vec = [v / norm for v in existing_vec]

    entries = [
        {"embedding": new_vec},
        {"embedding": existing_vec},
    ]
    result = _novelty_gain(entries, prev_snapshot={"entries_count": 1})
    # cos=0.5、-log(0.5 + 0.001) ≈ -log(0.501) ≈ 0.691 nat
    assert 0.6 < result < 0.8


def test_novelty_orthogonal_high_nat(monkeypatch):
    """case D (§6.3.1): 完全直交 (cosine=0) → 旧 1.0 vs 新 -log(ε) ≈ 6.9 nat (大識別)。

    識別力: 旧 1.0 と新 6.9 で 7 倍規模差、log scale 採用の直接証拠。
    """
    new_vec = _make_emb(0)  # axis 0
    existing_vec = _make_emb(1)  # axis 1 (orthogonal)
    entries = [
        {"embedding": new_vec},
        {"embedding": existing_vec},
    ]
    result = _novelty_gain(entries, prev_snapshot={"entries_count": 1})
    # cos=0、-log(0 + ε) = -log(0.001) ≈ 6.9 nat
    assert result > 5.0  # 旧 1.0 と区別、log scale 採用識別


# ========= pe_drop (§4.1.2) test 4 件 =========


def test_pe_drop_large_decrease_positive_nat():
    """case A (§6.3.2): prev=80, now=10 → log(80.001/10.001) ≈ 2.08 nat (識別: 旧 0.7)。"""
    state = {"last_prediction_error": 10.0}
    prev = {"last_prediction_error": 80.0}
    result = _pe_drop_gain(state, prev)
    # log(80.001/10.001) = log(7.999...) ≈ 2.079 nat
    assert 1.9 < result < 2.2
    # 旧実装 (0.7) と区別: 結果が > 1.5 (旧の 2 倍以上)


def test_pe_drop_unchanged_zero_nat():
    """case B (§6.3.2): prev=now → 0 nat (両実装一致)。"""
    state = {"last_prediction_error": 50.0}
    prev = {"last_prediction_error": 50.0}
    result = _pe_drop_gain(state, prev)
    assert abs(result) < 1e-6


def test_pe_drop_increase_negative_nat():
    """case C (§6.3.2、★ Lv3 negation 保持): prev=10, now=80 → -2.08 nat。

    識別力 (誤実装 fail): 旧 max(0, ·) クランプは 0 を返す、新は負値返す。
    """
    state = {"last_prediction_error": 80.0}
    prev = {"last_prediction_error": 10.0}
    result = _pe_drop_gain(state, prev)
    # log(10.001/80.001) ≈ -2.079 nat
    assert -2.2 < result < -1.9  # 負値、旧 0 と区別


def test_pe_drop_none_graceful():
    """case D (§6.3.2): pe_prev or pe_now None → 0 (両実装一致 graceful)。"""
    # both None
    assert _pe_drop_gain({}, {}) == 0.0
    # only one
    assert _pe_drop_gain({"last_prediction_error": 50.0}, {}) == 0.0
    assert _pe_drop_gain({}, {"last_prediction_error": 50.0}) == 0.0
    # explicit None
    assert _pe_drop_gain({"last_prediction_error": None}, {"last_prediction_error": 50.0}) == 0.0


# ========= density (§4.1.3) test 4 件 =========


def test_density_increase_positive_nat():
    """case A (§6.3.3): now=0.3, prev=0.1 → log(0.301/0.101) ≈ 1.092 nat (識別: 旧 0.2)。"""
    fog_now = {"local_density_mean": 0.3}
    prev = {"fog_local_density_mean": 0.1}
    result = _density_gain(prev, fog_now)
    # log(0.301/0.101) ≈ 1.092
    assert 1.0 < result < 1.2


def test_density_unchanged_zero_nat():
    """case B (§6.3.3): now=prev → 0 nat。"""
    fog_now = {"local_density_mean": 0.5}
    prev = {"fog_local_density_mean": 0.5}
    assert abs(_density_gain(prev, fog_now)) < 1e-6


def test_density_decrease_negative_nat():
    """case C (§6.3.3、★ Lv3 negation 保持): now=0.1, prev=0.3 → -1.092 nat。

    識別力 (誤実装 fail): 旧 max(0, ·) クランプは 0、新は負値で memory_graph edge 削除等
    の構造的情報減少を捕捉。
    """
    fog_now = {"local_density_mean": 0.1}
    prev = {"fog_local_density_mean": 0.3}
    result = _density_gain(prev, fog_now)
    assert -1.2 < result < -1.0


def test_density_none_graceful():
    """case D (§6.3.3): fog_now or prev が None / missing → 0 (graceful)。"""
    assert _density_gain({}, None) == 0.0
    assert _density_gain({"fog_local_density_mean": 0.5}, None) == 0.0
    assert _density_gain({}, {"local_density_mean": 0.5}) == 0.0


# ========= memory_link (§4.1.4) test 4 件 =========


def test_memory_link_strength_increase_log_ratio():
    """case A (§6.3.4): strength_now=5, prev=1 → log(6/2) ≈ 1.099 nat (識別: 旧 4.0)。"""
    links = [{"strength": 5.0}]
    prev = {"memory_links_strength_total": 1.0}
    result = _memory_link_gain(links, prev)
    # log(6.0/2.0) = log(3.0) ≈ 1.0986
    assert 1.0 < result < 1.2
    # 旧実装 (4.0) と区別: 結果が < 2.0


def test_memory_link_no_new_zero():
    """case B (§6.3.4): 新規 link 0 件 → 0 nat (両実装一致)。"""
    result = _memory_link_gain([], prev_snapshot={"memory_links_strength_total": 0.0})
    assert result == 0.0


def test_memory_link_decay_negative_nat():
    """case C (§6.3.4、★ Lv3 negation 保持): strength 減 → 負値 (識別: 旧 0 clamp)。"""
    links = [{"strength": 1.0}]
    prev = {"memory_links_strength_total": 5.0}
    result = _memory_link_gain(links, prev)
    # log(2/6) = log(1/3) ≈ -1.099 nat
    assert -1.2 < result < -1.0  # 負値、旧 0 clamp と区別


def test_memory_link_unchanged_zero():
    """case D (§6.3.4): strength 不変 → 0 nat。"""
    links = [{"strength": 3.0}]
    prev = {"memory_links_strength_total": 3.0}
    result = _memory_link_gain(links, prev)
    assert abs(result) < 1e-6
