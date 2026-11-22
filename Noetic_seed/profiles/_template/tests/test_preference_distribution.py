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
    _log_surface_area_unit_sphere,
    compute_C_from_self,
    estimate_density,
)


@pytest.fixture
def mock_embedding(monkeypatch):
    """bge-m3 を deterministic mock に差し替え (L2-normalized 単位 vector 生成)。

    Codex review breakpoint ① 指摘反映 (2026-05-11): `compute_C_from_self` は
    `from core.embedding import _embed_sync, is_vector_ready` で import 時に
    binding 固定されるため、`core.embedding` 側の attribute swap では isolate
    できず flaky になる。`core.preference_distribution` 側 symbol を patch する。
    (段階11-C hotfix 5 と同根の import スナップショット罠回避)。

    実 bge-m3 出力は L2-normalized 単位 vector のため、mock も同じ性質を維持:
    hashlib.md5 由来の deterministic な onehot 単位 vector を生成 (異なる text で
    異なる idx → cos_ab = 0、Step 2 vMF mixture test の geometric assert 整合)。
    """
    import hashlib

    def fake_embed(texts):
        out = []
        for t in texts:
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % 1024
            vec = [0.0] * 1024
            vec[idx] = 1.0  # onehot 単位 vector (L2-normalized)
            out.append(vec)
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


# ========= Step 2 (v1.1 vMF mixture estimate_density) test 6 件 =========


def _unit_vec(seed: int, dim: int = 1024) -> list:
    """deterministic な単位 vector (test 用)、特定 component に近づける助けとして直接 mean を流用。"""
    # 単位 vector を作る最も簡単な方法は first axis 単独
    v = [0.0] * dim
    v[seed % dim] = 1.0
    return v


def test_vmf_single_component_peak_at_mean(mock_embedding):
    """case A (§5.2 識別力): 単 component で mean 方向 max、anti-pode で min。

    想定誤実装 fail: log f_vMF(x|μ,κ) の κ·μ·x 符号間違いで anti-pode が max になる実装。
    """
    C = compute_C_from_self({"identity": "X"}, self_confidence={"identity": 0.9})
    log_p = estimate_density(C["components"], method="vmf")

    mean = C["components"][0]["mean"]
    anti = [-x for x in mean]

    assert log_p(mean) > log_p(anti)
    # 識別力: 数値的に十分な差 (κ ≈ 1/(1/0.901+ε) ≈ 0.9 なので peak vs anti で 2κ ≈ 1.8 の差は出る)
    assert log_p(mean) - log_p(anti) > 0.5


def test_vmf_higher_confidence_sharper_peak(mock_embedding):
    """case B (§5.2 識別力): 高 conf → 鋭い peak → log_p 範囲大。

    想定誤実装 fail: confidence 無視で κ 一律にすると peak range が同じになる。
    """
    C_high = compute_C_from_self({"id": "X"}, self_confidence={"id": 0.9})
    C_low = compute_C_from_self({"id": "X"}, self_confidence={"id": 0.3})

    high_p = estimate_density(C_high["components"], method="vmf")
    low_p = estimate_density(C_low["components"], method="vmf")

    mean_h = C_high["components"][0]["mean"]
    anti_h = [-x for x in mean_h]
    range_high = high_p(mean_h) - high_p(anti_h)

    mean_l = C_low["components"][0]["mean"]
    anti_l = [-x for x in mean_l]
    range_low = low_p(mean_l) - low_p(anti_l)

    # 高 conf (κ=1/(1/0.901+ε)≈0.9) の peak-anti 差 > 低 conf (κ≈0.3) の peak-anti 差
    assert range_high > range_low


def test_vmf_empty_components_uniform_sentinel(mock_embedding):
    """case C (§5.2 識別力 / bootstrap 整合): 空 components (N=0) → uniform 一定 log_p。

    識別力: 異なる x で log_p が変動する実装は uniform sentinel 期待で fail。
    """
    log_p = estimate_density([], method="vmf")

    x = _unit_vec(0)
    y = _unit_vec(100)
    assert log_p(x) == log_p(y)
    # uniform on S^{d-1}: log_p = -log(surface_area)
    expected = -_log_surface_area_unit_sphere(1024)
    assert log_p(x) == pytest.approx(expected)


def test_vmf_multiple_components_each_peak(mock_embedding):
    """case D (§5.2 識別力): 複数 component → 各 mean 方向で orthogonal 方向より高い log_p。

    想定誤実装 fail: logsumexp じゃなく 1 つの component しか参照しない実装で、
    片方の mean (e.g. mean_b) で log_p が orthogonal と同じ低値になる (= 当該 component が
    寄与しない設計バグ)。

    注意: N=2 同 κ では中点 (両 component 等距離点) が mode、各 mean が local peak ではない
    (vMF mixture の正常 geometry)。orthogonal (非 component 方向) と比較するのが正しい識別力。
    """
    C = compute_C_from_self(
        {"a": "first identity statement", "b": "totally different second"},
        self_confidence={"a": 0.7, "b": 0.7},
    )
    log_p = estimate_density(C["components"], method="vmf")

    mean_a = C["components"][0]["mean"]
    mean_b = C["components"][1]["mean"]

    # mean_a と mean_b は onehot 単位 vector で異なる idx (mock_embedding の md5 hash 由来)
    cos_ab = sum(a * b for a, b in zip(mean_a, mean_b))
    assert abs(cos_ab) < 0.9999

    # orthogonal: mean_a / mean_b と違う idx の onehot 単位 vector
    idx_a = mean_a.index(1.0)
    idx_b = mean_b.index(1.0)
    ortho = [0.0] * 1024
    ortho_idx = 0
    while ortho_idx in (idx_a, idx_b):
        ortho_idx += 1
    ortho[ortho_idx] = 1.0

    # 各 mean 方向は orthogonal より高い (両 component が log_p に寄与してる識別)
    assert log_p(mean_a) > log_p(ortho)
    assert log_p(mean_b) > log_p(ortho)
    # 識別力: 片方の component を無視する実装だと、無視される側 (e.g. mean_b) で
    # log_p が ortho と同等まで下がる → assert で fail


def test_vmf_unknown_method_raises_not_implemented(mock_embedding):
    """case E (§5.2 + 論点 ε): method='gaussian'/'kde' は段階N で iku self-mod 移行予約、現在は NotImplementedError。

    識別力: silently fallback して動く実装は ε 哲学整合性失敗で fail (明示的に未実装宣言する)。
    """
    C = compute_C_from_self({"id": "X"}, self_confidence={"id": 0.7})

    with pytest.raises(NotImplementedError, match="段階N"):
        estimate_density(C["components"], method="gaussian")
    with pytest.raises(NotImplementedError, match="段階N"):
        estimate_density(C["components"], method="kde")
    with pytest.raises(NotImplementedError, match="段階N"):
        estimate_density(C["components"], method="nsf")


def test_vmf_normalization_correct_for_typical_confidence(mock_embedding):
    """case G (Codex P1 反映、2026-05-11): typical Noetic 設定 (κ≈0.7、v=511) で
    正規化定数が数千 nat 単位の誤差を出さないこと検証 (Bessel regime 判定の識別力 fixture)。

    数学的事実 (実測 ≈ 2094 nat、d=1024, v=511, κ=0.7005):
        log I_{511}(0.7) 真値 ≈ v·log(κ/2) - lgamma(v+1) ≈ 511·log(0.35) - 2686 ≈ -3222
        log C_d(κ) = v·log(κ) - (d/2)·log(2π) - log I_v(κ) ≈ -182 - 941 + 3222 ≈ +2099
        log_p(mean) ≈ 0 + 2099 + κ·1 ≈ +2094 (正、sane vMF mixture)

    旧実装 fail (large-κ asymptotic 一律):
        log I_v(0.7) ≈ 0.7 - 0.5·log(2π·0.7) ≈ -0.04
        log C_d(κ) ≈ -182 - 941 + 0.04 ≈ -1123
        log_p(mean) ≈ -1122 (誤差 ~3200 nat、絶対値全く違う)

    識別力: 符号で完全区別 (旧負 vs 新正)、上限は sane vMF 範囲。
    """
    C = compute_C_from_self({"identity": "X"}, self_confidence={"identity": 0.7})
    log_p = estimate_density(C["components"], method="vmf")
    mean = C["components"][0]["mean"]

    log_p_mean = log_p(mean)

    # 識別力: 旧 fallback は ≈ -1122 (負)、新 regime 判定は ≈ +2094 (正)
    assert log_p_mean > 0, (
        f"log_p(mean) = {log_p_mean} は旧 large-κ fallback の signature (≈ -1122)、"
        f"new regime 判定 (small-κ / high-order) が動いていない"
    )
    # sane 上限: 1024 次元 vMF 正規化定数の絶対値、桁外れ overflow 検出
    assert log_p_mean < 5000, (
        f"log_p(mean) = {log_p_mean} は typical vMF 範囲を逸脱、"
        f"regime 判定 or Bessel 計算に他の問題あり"
    )


def test_vmf_graceful_when_all_means_none():
    """case F (Step 1 graceful 継承): bge-m3 未起動で全 mean=None → uniform sentinel graceful。

    識別力: 例外 raise 実装は test で fail (Noetic 哲学: 起動失敗で固まらない)。
    """
    original_ready = pd_mod.is_vector_ready
    pd_mod.is_vector_ready = lambda: False
    try:
        C = compute_C_from_self(
            {"id": "X", "role": "Y"}, self_confidence={"id": 0.7, "role": 0.7}
        )
        # 全 component mean=None 確認 (Step 1 設計)
        assert all(c["mean"] is None for c in C["components"])

        log_p = estimate_density(C["components"], method="vmf")
        # uniform sentinel に縮退 (異なる x で同じ log_p)
        x = _unit_vec(0)
        y = _unit_vec(100)
        assert log_p(x) == log_p(y)
        # uniform 値の literal 検算
        expected = -_log_surface_area_unit_sphere(1024)
        assert log_p(x) == pytest.approx(expected)
    finally:
        pd_mod.is_vector_ready = original_ready


# ========= Step 7 (§5.3 bootstrap 遷移) test 2 件 =========


def test_C_transition_empty_to_unimodal_to_multimodal(mock_embedding):
    """case H (Step 7 §5.3 literal): 空 → 単峰 → 多峰 連続遷移 (bootstrap handling)。

    PLAN §5.3 literal:
        - state.self 空 → 均一分布 C (C_entropy = uniform sentinel、n_components=0)
        - state.self に key 追加 → Step 1 compute_C_from_self で C 構築
        - 均一 → 単峰 → 多峰 の遷移サポート

    識別力: 各 stage で n_components / C_entropy / source_keys が単調に変化。
    想定誤実装 fail: 空 case ハンドルなし → empty case で C_entropy=0 になり
    「単峰 < 空」inequality が成り立たず assert fail。
    """
    # Stage 0: clean profile (空 state.self) → 均一 sentinel
    C0 = compute_C_from_self({}, self_confidence={})
    assert C0["n_components"] == 0
    assert C0["source_keys"] == []

    # Stage 1: 1 key 追加 → 単峰
    C1 = compute_C_from_self(
        {"identity": "I observe"}, self_confidence={"identity": 0.7}
    )
    assert C1["n_components"] == 1
    assert C1["source_keys"] == ["identity"]
    assert C1["C_entropy"] < C0["C_entropy"]  # 均一 sentinel より集中 (preference 強)

    # Stage 2: 2 key 追加 → 多峰
    C2 = compute_C_from_self(
        {"identity": "I observe", "preferences": "diverse exploration"},
        self_confidence={"identity": 0.7, "preferences": 0.7},
    )
    assert C2["n_components"] == 2
    assert set(C2["source_keys"]) == {"identity", "preferences"}
    # 多峰は単峰より entropy 高い (mixing weight 項 -Σ w log w 追加)
    assert C2["C_entropy"] > C1["C_entropy"]
    # 識別力: 単調順序 C0 > C2 > C1 (uniform > multimodal > unimodal)
    assert C0["C_entropy"] > C2["C_entropy"] > C1["C_entropy"]


def test_estimate_density_transition_empty_to_vmf_mixture(mock_embedding):
    """case I (Step 7 estimate_density 拡張): 空 → 単峰 vMF → 多峰 vMF 遷移検証。

    識別力: 各 stage で log_p の挙動が「uniform 定数 → mean 方向 peak → 複数 peak」と
    遷移、bootstrap handling が estimate_density 側にも一貫している (uniform sentinel が
    estimate_density の側でも縮退正しい)。
    """
    # Stage 0: 空 → uniform sentinel (どの x でも同じ log_p)
    C0 = compute_C_from_self({}, self_confidence={})
    log_p0 = estimate_density(C0["components"], method="vmf")
    x = _unit_vec(0)
    y = _unit_vec(100)
    assert log_p0(x) == log_p0(y)
    uniform_val = log_p0(x)

    # Stage 1: 単峰 vMF → mean 方向で uniform より高い (concentration あり)
    C1 = compute_C_from_self(
        {"identity": "X"}, self_confidence={"identity": 0.7}
    )
    log_p1 = estimate_density(C1["components"], method="vmf")
    mean1 = C1["components"][0]["mean"]
    assert log_p1(mean1) > uniform_val

    # Stage 2: 多峰 vMF → 各 component mean 方向で uniform より高い
    C2 = compute_C_from_self(
        {"identity": "X", "role": "Y"},
        self_confidence={"identity": 0.7, "role": 0.7},
    )
    log_p2 = estimate_density(C2["components"], method="vmf")
    mean2_a = C2["components"][0]["mean"]
    mean2_b = C2["components"][1]["mean"]
    assert log_p2(mean2_a) > uniform_val
    assert log_p2(mean2_b) > uniform_val
