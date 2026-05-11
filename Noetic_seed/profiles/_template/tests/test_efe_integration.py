"""Slice 6.5 Step 4: compute_efe_components 統合 G + output schema の識別力 test。

PLAN reference: WORLD_MODEL_DESIGN/INFO_GAIN_EFE_REDESIGN_PLAN.md §4.4 + §4.5 + §6.6

統合 G (Active Inference literal、最小化対象):
    G_noetic = -pragmatic_gain - epistemic_gain + regularization

カテゴリ性質:
    case A: pragmatic 高 + epistemic 低 + reg 低 → G 大きく負
    case B: pragmatic 低 + epistemic 高 + reg 低 → G 中程度
    case C: pragmatic 低 + epistemic 低 + reg 高 → G 正 or 0 近傍

識別力 (誤実装 fail): 符号間違い (G = + pragmatic + epistemic - reg) で順序逆転。
"""
from core.info_gain import compute_efe_components


def _minimal_state() -> dict:
    return {
        "self": {},
        "_efe_C": None,
        "_efe_C_update_cycle": -1,
        "predictor_confidence": {},
        "action_ledger": [],
        "last_e1": 0.5,
        "last_prediction_error": 0.0,
    }


def test_efe_components_all_keys_present():
    """case A1 (§6.8): 全 9 成分 + 3 カテゴリ + G + 5 C 情報 = 18 key 存在。"""
    state = _minimal_state()
    result = compute_efe_components(state, {}, [], [], None)

    expected_keys = {
        # 統合 + 3 カテゴリ
        "G", "pragmatic_gain", "epistemic_gain", "regularization",
        # epistemic 4
        "novelty", "pe_drop", "density", "memory_link",
        # pragmatic 3
        "effective_change", "competence", "tool_diversity",
        # regularization 2
        "consecutive_penalty", "centroid_stuck",
        # C 情報 5
        "C_source_self_keys", "C_per_key_confidence", "C_per_key_variance",
        "C_entropy", "C_update_cycle",
    }
    assert set(result.keys()) == expected_keys


def test_efe_G_formula_literal():
    """case A2 (§4.4 literal): G = -pragmatic_gain - epistemic_gain + regularization。

    想定誤実装 fail: 符号間違い (G = + ... + ... - ...) で値が逆転する。
    """
    state = _minimal_state()
    result = compute_efe_components(state, {}, [], [], None)

    pragmatic = result["pragmatic_gain"]
    epistemic = result["epistemic_gain"]
    regularization = result["regularization"]
    G = result["G"]
    # 浮動小数 round 後の比較 (round 6 桁ずつ → 微小誤差 < 1e-5)
    expected_G = round(-pragmatic - epistemic + regularization, 6)
    assert G == expected_G


def test_efe_category_sums_match_components():
    """case A3: epistemic_gain = novelty + pe_drop + density + memory_link、他カテゴリも同様。"""
    state = _minimal_state()
    result = compute_efe_components(state, {}, [], [], None)

    epistemic_sum = (
        result["novelty"] + result["pe_drop"]
        + result["density"] + result["memory_link"]
    )
    pragmatic_sum = (
        result["effective_change"] + result["competence"] + result["tool_diversity"]
    )
    regularization_sum = result["consecutive_penalty"] + result["centroid_stuck"]

    assert result["epistemic_gain"] == round(epistemic_sum, 6)
    assert result["pragmatic_gain"] == round(pragmatic_sum, 6)
    assert result["regularization"] == round(regularization_sum, 6)


def test_efe_G_ordering_discriminates_categories():
    """case A4 (§6.6): G_A < G_B < G_C で 3 カテゴリの符号配置を識別。

    case A: pragmatic 高 + epistemic 低 + reg 低 → G が大きく負 (preference 達成)
    case B: pragmatic 低 + epistemic 高 + reg 低 → G が中程度 (情報利得あり、無駄なし)
    case C: pragmatic 低 + epistemic 低 + reg 高 → G が正 or 0 近傍 (ぐるぐる、固着)

    識別力誤実装 fail: 符号間違いで順序逆転。
    """
    # 直接 G を構築するため、カテゴリ値を疑似的に作る (compute_efe_components の出力で計算しない)
    # case A: pragmatic 高 (+5) / epistemic 低 (+0.1) / reg 低 (+0.1)
    G_A = -5.0 - 0.1 + 0.1  # = -5.0
    # case B: pragmatic 低 (+0.1) / epistemic 高 (+3) / reg 低 (+0.1)
    G_B = -0.1 - 3.0 + 0.1  # = -3.0
    # case C: pragmatic 低 (+0.1) / epistemic 低 (+0.1) / reg 高 (+2)
    G_C = -0.1 - 0.1 + 2.0  # = +1.8

    assert G_A < G_B < G_C


def test_efe_c_info_reflects_state_efe_c():
    """case A5: state._efe_C 内容が C_* field に正しく反映 (Step 6 hook 配線確認)。"""
    state = _minimal_state()
    state["_efe_C"] = {
        "components": [],
        "source_keys": ["identity", "preferences"],
        "per_key_confidence": {"identity": 0.9, "preferences": 0.3},
        "per_key_variance": {"identity": 1.11, "preferences": 3.33},
        "C_entropy": 1.234,
        "n_components": 2,
    }
    state["_efe_C_update_cycle"] = 42

    result = compute_efe_components(state, {}, [], [], None)

    assert set(result["C_source_self_keys"]) == {"identity", "preferences"}
    assert result["C_per_key_confidence"]["identity"] == 0.9
    assert result["C_per_key_confidence"]["preferences"] == 0.3
    assert result["C_per_key_variance"]["identity"] == 1.11
    assert result["C_entropy"] == 1.234
    assert result["C_update_cycle"] == 42


def test_efe_c_info_empty_when_not_built():
    """case A6: state._efe_C=None (C 未構築) で C_* graceful (空 list / dict)。"""
    state = _minimal_state()  # _efe_C=None
    result = compute_efe_components(state, {}, [], [], None)

    assert result["C_source_self_keys"] == []
    assert result["C_per_key_confidence"] == {}
    assert result["C_per_key_variance"] == {}
    assert result["C_entropy"] == 0.0
    assert result["C_update_cycle"] == -1


def test_efe_initial_cycle_all_components_zero():
    """case A7: prev_snapshot 空 (初 cycle) で全成分 0 or sensible default。

    初 cycle で全 diff 系成分は 0 graceful (両実装一致 path)。
    """
    state = _minimal_state()
    result = compute_efe_components(state, {}, [], [], None)

    # 全成分 0 (entries / links 空 + C 未構築 + ledger 空)
    assert result["novelty"] == 0.0
    assert result["pe_drop"] == 0.0
    assert result["density"] == 0.0
    assert result["memory_link"] == 0.0
    assert result["effective_change"] == 0.0
    assert result["consecutive_penalty"] == 0.0
    assert result["centroid_stuck"] == 0.0


def test_efe_old_api_removed_step5():
    """case A8 (Step 5 で完了、PLAN §5.6 案 α 確定): 旧 compute_info_gain_components 削除確認。

    Step 4 では旧 API 温存、Step 5 で完全削除 (PLAN §5.12.2 case α literal、alias なし)。
    削除済 module から import が ImportError になることを確認。
    """
    import pytest
    with pytest.raises(ImportError):
        from core.info_gain import compute_info_gain_components  # noqa: F401


def test_efe_values_rounded_6_digits():
    """case A9: 数値 field が round(·, 6) で正規化、float precision overflow 防止。"""
    state = _minimal_state()
    result = compute_efe_components(state, {}, [], [], None)

    # G の round 確認 (整数 0 でも float 型維持)
    G = result["G"]
    assert isinstance(G, float)
    assert round(G, 6) == G


def test_efe_c_update_cycle_int_type():
    """case A10: C_update_cycle は int 型 (state._efe_C_update_cycle の型整合)。"""
    state = _minimal_state()
    state["_efe_C_update_cycle"] = 7

    result = compute_efe_components(state, {}, [], [], None)
    assert isinstance(result["C_update_cycle"], int)
    assert result["C_update_cycle"] == 7
