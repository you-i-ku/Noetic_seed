"""Slice 6.5 Step 1: preference_distribution.py の識別力 test (12 件)。

PLAN reference: WORLD_MODEL_DESIGN/INFO_GAIN_EFE_REDESIGN_PLAN.md §5.1 + §6.4

識別力 fixture (CLAUDE.md §5 literal): 各 case で「想定誤実装」が fail する設計。
"""
import math

import pytest

from core import preference_distribution as pd_mod
from core.preference_distribution import (
    DEFAULT_CONFIDENCE,
    EPS,
    NAME_KEY,
    UNIFORM_VARIANCE,
    _gaussian_entropy,
    compute_C_from_self,
)


@pytest.fixture
def mock_embedding(monkeypatch):
    """bge-m3 を deterministic mock に差し替え。

    Codex review breakpoint ① 指摘反映 (2026-05-11): `compute_C_from_self` は
    `from core.embedding import _embed_sync, is_vector_ready` で import 時に
    binding 固定されるため、`core.embedding` 側の attribute swap では isolate
    できず flaky になる。`core.preference_distribution` 側 symbol を patch する。
    (段階11-C hotfix 5 と同根の import スナップショット罠回避)。

    text hash 由来の base 値で 1024 次元 vector、key 別の embedding 区別を保つ。
    """
    def fake_embed(texts):
        out = []
        for t in texts:
            base = (hash(t) % 1000) / 1000.0
            out.append([base] * 1024)
        return out

    original_embed = pd_mod._embed_sync
    original_ready = pd_mod.is_vector_ready
    pd_mod._embed_sync = fake_embed
    pd_mod.is_vector_ready = lambda: True
    try:
        yield
    finally:
        pd_mod._embed_sync = original_embed
        pd_mod.is_vector_ready = original_ready


def test_empty_state_self_returns_uniform_sentinel(mock_embedding):
    """case 1 (§6.4): 空 (clean profile) → 均一分布 sentinel (大 entropy)。

    想定誤実装 fail: 空 case で C_entropy=0 にすると `assert > log(N_uniform)` で fail。
    """
    C = compute_C_from_self({}, self_confidence={})
    assert C["n_components"] == 0
    assert C["components"] == []
    assert C["source_keys"] == []
    assert C["per_key_confidence"] == {}
    assert C["per_key_variance"] == {}

    # log(N_uniform) ≒ variance=1/EPS の isotropic Gaussian entropy (1024 dim)
    expected_uniform = _gaussian_entropy(UNIFORM_VARIANCE, 1024)
    assert C["C_entropy"] == pytest.approx(expected_uniform)


def test_single_key_lower_entropy_than_uniform(mock_embedding):
    """case 2 (§6.4): 単 key (conf=0.7) → 単峰、entropy < 空 case。"""
    C_empty = compute_C_from_self({}, self_confidence={})
    C1 = compute_C_from_self(
        {"identity": "I observe and connect"},
        self_confidence={"identity": 0.7},
    )
    assert C1["n_components"] == 1
    assert C1["source_keys"] == ["identity"]
    assert C1["per_key_confidence"]["identity"] == 0.7
    assert C1["C_entropy"] < C_empty["C_entropy"]


def test_multiple_keys_entropy_above_single(mock_embedding):
    """case 3 (§6.4): 複数 key 同 conf → entropy > 単 key (mixing weight 項追加)。"""
    C1 = compute_C_from_self(
        {"identity": "X"},
        self_confidence={"identity": 0.7},
    )
    C3 = compute_C_from_self(
        {"identity": "X", "preferences": "Y", "role": "Z"},
        self_confidence={"identity": 0.7, "preferences": 0.7, "role": 0.7},
    )
    assert C3["n_components"] == 3
    assert set(C3["source_keys"]) == {"identity", "preferences", "role"}
    assert C3["C_entropy"] > C1["C_entropy"]


def test_name_key_excluded_from_C_source(mock_embedding):
    """case 5 (§6.4): ★ NAME_KEY は C source から自動除外 (不変層 exception)。

    想定誤実装 fail: name を含めると n_components==2 になり assert で fail。
    """
    C = compute_C_from_self(
        {"name": "iku", "identity": "I observe"},
        self_confidence={"identity": 0.7},
    )
    assert C["n_components"] == 1
    assert C["source_keys"] == ["identity"]
    assert NAME_KEY not in C["source_keys"]
    assert NAME_KEY not in C["per_key_variance"]
    assert NAME_KEY not in C["per_key_confidence"]


def test_per_key_confidence_discriminates_variance(mock_embedding):
    """case 6 (§6.4): 高 conf → 小 variance → 鋭い peak → C_entropy 低。

    想定誤実装 fail: confidence 無視で variance 一律にすると C_high == C_low。
    """
    C_high = compute_C_from_self(
        {"identity": "X"}, self_confidence={"identity": 0.9}
    )
    C_low = compute_C_from_self(
        {"identity": "X"}, self_confidence={"identity": 0.3}
    )
    assert C_high["per_key_variance"]["identity"] < C_low["per_key_variance"]["identity"]
    assert C_high["C_entropy"] < C_low["C_entropy"]


def test_mixed_confidence_per_key_variance(mock_embedding):
    """case 7 (§6.4): 異なる conf で各 peak 独立 variance。"""
    C = compute_C_from_self(
        {"identity": "X", "role": "Y"},
        self_confidence={"identity": 0.9, "role": 0.3},
    )
    assert C["per_key_variance"]["identity"] < C["per_key_variance"]["role"]
    assert C["per_key_confidence"]["identity"] == 0.9
    assert C["per_key_confidence"]["role"] == 0.3


def test_default_confidence_when_missing(mock_embedding):
    """confidence 省略 / 部分欠損 → DEFAULT_CONFIDENCE で補完。"""
    C = compute_C_from_self(
        {"identity": "X", "role": "Y"},
        self_confidence={"identity": 0.5},
    )
    assert C["per_key_confidence"]["identity"] == 0.5
    assert C["per_key_confidence"]["role"] == DEFAULT_CONFIDENCE


def test_self_confidence_none_uses_default_for_all(mock_embedding):
    """self_confidence=None → 全 key default 0.7。"""
    C = compute_C_from_self(
        {"identity": "X", "preferences": "Y"},
        self_confidence=None,
    )
    assert C["per_key_confidence"]["identity"] == DEFAULT_CONFIDENCE
    assert C["per_key_confidence"]["preferences"] == DEFAULT_CONFIDENCE


def test_source_keys_filter_subset(mock_embedding):
    """source_keys 指定で subset filtering、NAME_KEY exception 継続。"""
    C = compute_C_from_self(
        {"identity": "X", "preferences": "Y", "role": "Z", "name": "iku"},
        self_confidence={"identity": 0.7, "preferences": 0.7, "role": 0.7},
        source_keys=["identity", "preferences", "name"],
    )
    assert C["n_components"] == 2
    assert set(C["source_keys"]) == {"identity", "preferences"}
    assert "role" not in C["per_key_variance"]
    assert NAME_KEY not in C["source_keys"]


def test_variance_formula_literal(mock_embedding):
    """variance_k = 1/(conf+ε) literal 検算 (識別力: 計算式間違い fail)。"""
    C = compute_C_from_self(
        {"identity": "X"}, self_confidence={"identity": 0.5}
    )
    expected = 1.0 / (0.5 + EPS)
    assert C["per_key_variance"]["identity"] == pytest.approx(expected)


def test_confidence_clamped_to_unit_interval(mock_embedding):
    """confidence は [0.0, 1.0] にクランプ (識別力: clamp 漏れで variance 逆向き)。"""
    C_neg = compute_C_from_self(
        {"identity": "X"}, self_confidence={"identity": -0.5}
    )
    C_over = compute_C_from_self(
        {"identity": "X"}, self_confidence={"identity": 1.5}
    )
    assert C_neg["per_key_confidence"]["identity"] == 0.0
    assert C_over["per_key_confidence"]["identity"] == 1.0
    assert C_over["per_key_variance"]["identity"] < C_neg["per_key_variance"]["identity"]


def test_graceful_when_bge_m3_unavailable():
    """bge-m3 未起動 → instantiate 失敗させず、mean=None で components 構築継続。

    識別力: 例外 raise 実装は test で fail (Noetic 哲学: 起動失敗で固まらない)。
    Codex review ① 指摘反映: `pd_mod.is_vector_ready` を patch (import 固定回避)。
    """
    original_ready = pd_mod.is_vector_ready
    pd_mod.is_vector_ready = lambda: False
    try:
        C = compute_C_from_self(
            {"identity": "X"}, self_confidence={"identity": 0.7}
        )
        assert C["n_components"] == 1
        assert C["components"][0]["mean"] is None
        assert C["components"][0]["variance"] == pytest.approx(1.0 / (0.7 + EPS))
        assert C["per_key_variance"]["identity"] == pytest.approx(1.0 / (0.7 + EPS))
        # C_entropy は mean 無関係で variance ベース計算可能
        assert math.isfinite(C["C_entropy"])
    finally:
        pd_mod.is_vector_ready = original_ready
