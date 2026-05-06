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

    # 閾値同値 (Codex review P2-1 fix): mean=5.0 で flat=False を強制
    # (誤実装 `<=` だと pass せず、識別力 up)
    state_equal = {"prediction_error_history_e2": [5.0] * 10}
    update_cumulative_information_gain(state_equal, 5.0, None)
    equal_flat = state_equal["cumulative_information_gain"]["flat_signal"]

    return all([
        _assert(high_flat is False, "mean=10.0 → flat=False"),
        _assert(low_flat is True, "mean=1.0 → flat=True"),
        _assert(below_flat is True, "mean=4.99 (閾値直下) → flat=True"),
        _assert(equal_flat is False, "mean=5.0 (閾値同値、strict <) → flat=False"),
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


def test_cig_end_to_end_via_update_predictor_confidence():
    """update_predictor_confidence() 経由で CIG が history append 後に走る順序を検証。

    Codex review P2-2 fix: production wiring (history append → CIG update) の
    ordering 整合性を end-to-end で確認。直接 helper 呼出し test は history を
    bypass してたため、ordering regression を検出できなかった。

    識別力: CIG を history append 前に呼ぶ誤実装 (= 最新 error が window_mean
    に含まれない) で window_mean が想定値と乖離 → fail する。
    """
    print("== Step A: end-to-end (update_predictor_confidence → CIG) ==")
    from core.predictor import update_predictor_confidence

    state = {}
    # 1 cycle 目: error=4.0、history は空から append、window_mean=4.0 想定
    update_predictor_confidence(state, "reflect", 4.0)
    cig_after_first = dict(state["cumulative_information_gain"])

    # 2 cycle 目: error=2.0、history=[4.0, 2.0]、window_mean=3.0 想定
    update_predictor_confidence(state, "reflect", 2.0)
    cig_after_second = dict(state["cumulative_information_gain"])

    # 3 cycle 目: error=10.0、window_mean=(4+2+10)/3≒5.33 で flat=False 切替想定
    update_predictor_confidence(state, "reflect", 10.0)
    cig_after_third = dict(state["cumulative_information_gain"])

    return all([
        _assert(abs(cig_after_first["e2_window_mean"] - 4.0) < 1e-9,
                f"1 cycle 後 mean=4.0 ({cig_after_first['e2_window_mean']})"),
        _assert(cig_after_first["flat_signal"] is True,
                "1 cycle 後 mean=4.0 < 5.0 で flat=True"),
        _assert(abs(cig_after_second["e2_window_mean"] - 3.0) < 1e-9,
                f"2 cycle 後 mean=3.0 ({cig_after_second['e2_window_mean']})"),
        _assert(cig_after_second["flat_streak"] == 2,
                f"2 cycle 連続 flat で streak=2 ({cig_after_second['flat_streak']})"),
        _assert(cig_after_third["flat_signal"] is False,
                f"3 cycle 後 mean≒5.33 で flat=False ({cig_after_third['e2_window_mean']:.2f})"),
        _assert(cig_after_third["flat_streak"] == 0,
                "flat=False で streak リセット"),
        _assert(abs(cig_after_third["e2_total"] - 16.0) < 1e-9,
                f"e2_total 累積 16.0 ({cig_after_third['e2_total']})"),
    ])


# ============================================================
# Step B — Attractor Cosine Redundancy Detection
# ============================================================
# PLAN §4-4 検証要件 + CLAUDE.md §5 識別力 4 fixture (空 / 全一致 / 部分一致 /
# 完全不一致)。embedding は controller モジュール内の _embed_sync を mock、
# cosine_similarity は real 流用 (deterministic 直交ベクトル)。

def _patch_embed_sync(controller_mod, mock_fn):
    """controller モジュール内の _embed_sync / is_vector_ready を patch。"""
    original_ready = controller_mod.is_vector_ready
    original_embed = controller_mod._embed_sync
    controller_mod.is_vector_ready = lambda: True
    controller_mod._embed_sync = mock_fn
    return original_ready, original_embed


def _restore_embed_sync(controller_mod, originals):
    controller_mod.is_vector_ready = originals[0]
    controller_mod._embed_sync = originals[1]


def test_attractor_redundancy_all_orthogonal():
    """全 5 候補が直交 (異 tool / 異 reason) → cluster_count=5, diversity=1.0。

    識別力: cluster 計算が常に 1 を返す誤実装、または cosine 閾値判定逆実装で
    fail (cluster_count=5 と 1 で区別)。
    """
    print("== Step B: 全 5 候補直交 → cluster_count=5 ==")
    import core.controller as cm

    def mock_embed(texts):
        n = len(texts)
        # 各 text に互いに直交する単位ベクトル (cosine=0)
        return [[(1.0 if j == i else 0.0) for j in range(max(n, 5))]
                for i in range(n)]

    candidates = [
        {"tool": "reflect", "reason": "A"},
        {"tool": "memory_store", "reason": "B"},
        {"tool": "read_file", "reason": "C"},
        {"tool": "update_self", "reason": "D"},
        {"tool": "bash", "reason": "E"},
    ]

    orig = _patch_embed_sync(cm, mock_embed)
    try:
        result = cm._detect_attractor_redundancy(candidates)
    finally:
        _restore_embed_sync(cm, orig)

    return all([
        _assert(result["cluster_count"] == 5,
                f"cluster_count=5 ({result['cluster_count']})"),
        _assert(abs(result["diversity_score"] - 1.0) < 1e-9,
                f"diversity_score=1.0 ({result['diversity_score']})"),
        _assert(len(result["redundancy_pairs"]) == 0,
                f"redundancy_pairs 空 ({len(result['redundancy_pairs'])})"),
        _assert(len(result["redundant_tool_set"]) == 0,
                "redundant_tool_set 空"),
    ])


def test_attractor_redundancy_all_same():
    """全 5 候補が同 tool / 同 embedding → cluster_count=1, diversity=0.2,
    redundant_tool_set={reflect}。

    識別力: redundant_tool_set で同 tool 判定漏れの誤実装、または
    cluster_count を pair 数で計算する誤実装 (10 vs 1) で fail。
    """
    print("== Step B: 全 5 候補同 tool 同 embedding → cluster_count=1 ==")
    import core.controller as cm

    def mock_embed(texts):
        return [[1.0, 0.0, 0.0, 0.0, 0.0]] * len(texts)

    candidates = [
        {"tool": "reflect", "reason": f"reason_{i}"} for i in range(5)
    ]

    orig = _patch_embed_sync(cm, mock_embed)
    try:
        result = cm._detect_attractor_redundancy(candidates)
    finally:
        _restore_embed_sync(cm, orig)

    return all([
        _assert(result["cluster_count"] == 1,
                f"cluster_count=1 ({result['cluster_count']})"),
        _assert(abs(result["diversity_score"] - 0.2) < 1e-9,
                f"diversity_score=0.2 ({result['diversity_score']})"),
        _assert("reflect" in result["redundant_tool_set"],
                "redundant_tool_set に reflect 含む"),
        _assert(len(result["redundancy_pairs"]) == 10,
                f"redundancy_pairs=C(5,2)=10 ({len(result['redundancy_pairs'])})"),
    ])


def test_attractor_redundancy_partial():
    """前 2 候補が redundant (同 tool 同 embedding)、残り 3 候補は独立
    (異 tool 直交) → cluster_count=4, diversity=0.8, redundant={reflect} のみ。

    識別力: union-find が動かず pair 数=cluster とする誤実装、または
    redundant_tool_set が異 tool 同 embedding を含めてしまう誤実装で fail。
    """
    print("== Step B: 部分 redundant (2 同/3 独立) → cluster_count=4 ==")
    import core.controller as cm

    def mock_embed(texts):
        # 前 2 件は同 vector、後 3 件は直交
        return [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]

    candidates = [
        {"tool": "reflect", "reason": "shared reason"},
        {"tool": "reflect", "reason": "shared reason"},
        {"tool": "memory_store", "reason": "B"},
        {"tool": "read_file", "reason": "C"},
        {"tool": "bash", "reason": "D"},
    ]

    orig = _patch_embed_sync(cm, mock_embed)
    try:
        result = cm._detect_attractor_redundancy(candidates)
    finally:
        _restore_embed_sync(cm, orig)

    return all([
        _assert(result["cluster_count"] == 4,
                f"cluster_count=4 ({result['cluster_count']})"),
        _assert(abs(result["diversity_score"] - 0.8) < 1e-9,
                f"diversity_score=0.8 ({result['diversity_score']})"),
        _assert("reflect" in result["redundant_tool_set"],
                "reflect 縮退検出"),
        _assert("memory_store" not in result["redundant_tool_set"],
                "memory_store 非縮退 (異 tool 別 cluster)"),
        _assert(len(result["redundancy_pairs"]) == 1,
                f"redundancy_pairs=1 (前 2 件のみ)"),
    ])


def test_attractor_redundancy_state_persist_json_safe():
    """state["last_redundancy"] が json.dumps を通る contract 検証。

    Codex review BLOCKER fix: redundant_tool_set を set のまま state に保存
    する誤実装は json.dumps で TypeError raise → fail。controller_select 内
    の state 保存ブロックと同じ shape を再現して contract test 化。

    識別力: state["last_redundancy"] に set / tuple を直接保存する誤実装で
    json.dumps が TypeError raise → fail。
    """
    print("== Step B: state[last_redundancy] が JSON-safe (BLOCKER fix) ==")
    import json
    import core.controller as cm

    def mock_embed(texts):
        return [[1.0, 0.0, 0.0, 0.0, 0.0]] * len(texts)

    candidates = [{"tool": "reflect", "reason": f"r_{i}"} for i in range(5)]

    orig = _patch_embed_sync(cm, mock_embed)
    try:
        # controller_select の state 保存ブロックと同じ projection
        redundancy = cm._detect_attractor_redundancy(candidates)
        state = {}
        state["last_redundancy"] = {
            "redundancy_pairs": [list(p) for p in redundancy["redundancy_pairs"]],
            "cluster_count": redundancy["cluster_count"],
            "redundant_tool_set": sorted(redundancy["redundant_tool_set"]),
            "diversity_score": redundancy["diversity_score"],
        }
        try:
            json.dumps(state)
            json_ok = True
            json_err = None
        except TypeError as e:
            json_ok = False
            json_err = str(e)
    finally:
        _restore_embed_sync(cm, orig)

    lr = state.get("last_redundancy", {})
    return all([
        _assert(json_ok, f"json.dumps(state) 成功 (err={json_err})"),
        _assert(isinstance(lr.get("redundant_tool_set"), list),
                "redundant_tool_set が list 型 (set じゃない)"),
        _assert(lr.get("redundant_tool_set") == ["reflect"],
                f"set 内容保持 ({lr.get('redundant_tool_set')})"),
        _assert(isinstance(lr.get("redundancy_pairs"), list),
                "redundancy_pairs list 型"),
        _assert(all(isinstance(p, list) for p in lr.get("redundancy_pairs", [])),
                "redundancy_pairs 各要素 list (tuple じゃない)"),
        _assert(lr.get("cluster_count") == 1,
                f"cluster_count 保持 ({lr.get('cluster_count')})"),
        _assert(abs(lr.get("diversity_score", -1) - 0.2) < 1e-9,
                "diversity_score 保持"),
    ])


def test_attractor_redundancy_malformed_vecs_fallback():
    """_embed_sync が malformed (length 不一致) vector list 返却時、neutral
    fallback (cluster_count=N, diversity=1.0) で動作。

    Codex review P2-b fix: vecs is None だけ check の誤実装は len 不一致で
    indexing error → fail (本テストは neutral fallback で正常動作確認)。
    """
    print("== Step B: malformed vecs (length 不一致) の防御 fallback ==")
    import core.controller as cm

    candidates = [{"tool": "reflect", "reason": f"r_{i}"} for i in range(5)]

    # case A: 空 list 返却 (`not vecs` で発火)
    def mock_empty(texts):
        return []
    orig = _patch_embed_sync(cm, mock_empty)
    try:
        empty_result = cm._detect_attractor_redundancy(candidates)
    finally:
        _restore_embed_sync(cm, orig)

    # case B: length 不一致 (n=5 候補に 2 件しか返さない)
    def mock_short(texts):
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    orig = _patch_embed_sync(cm, mock_short)
    try:
        short_result = cm._detect_attractor_redundancy(candidates)
    finally:
        _restore_embed_sync(cm, orig)

    return all([
        _assert(empty_result["cluster_count"] == 5,
                f"empty vecs neutral fallback cluster_count=N=5 ({empty_result['cluster_count']})"),
        _assert(abs(empty_result["diversity_score"] - 1.0) < 1e-9,
                "empty vecs diversity=1.0 (圧縮発火しない)"),
        _assert(short_result["cluster_count"] == 5,
                f"length 不一致 neutral fallback cluster_count=N=5 ({short_result['cluster_count']})"),
        _assert(abs(short_result["diversity_score"] - 1.0) < 1e-9,
                "length 不一致 diversity=1.0"),
        _assert(len(short_result["redundancy_pairs"]) == 0,
                "length 不一致 で indexing error せず空 pairs"),
    ])


def test_attractor_redundancy_empty_and_fallback():
    """空入力 / vector 未起動 / _embed_sync=None の fallback 動作。

    識別力: 空入力で IndexError raise の誤実装、fallback で diversity_score=0.0
    を返す誤実装 (cluster_count=N + diversity=1.0 が正、補正発火しないため)。
    """
    print("== Step B: 空入力 / 未起動 / None fallback ==")
    import core.controller as cm

    # 空入力
    empty_result = cm._detect_attractor_redundancy([])

    candidates = [{"tool": "reflect", "reason": str(i)} for i in range(3)]
    original_ready = cm.is_vector_ready
    original_embed = cm._embed_sync

    # vector 未起動
    cm.is_vector_ready = lambda: False
    try:
        not_ready_result = cm._detect_attractor_redundancy(candidates)
    finally:
        cm.is_vector_ready = original_ready

    # _embed_sync=None 返却
    cm.is_vector_ready = lambda: True
    cm._embed_sync = lambda texts: None
    try:
        none_result = cm._detect_attractor_redundancy(candidates)
    finally:
        cm._embed_sync = original_embed
        cm.is_vector_ready = original_ready

    return all([
        _assert(empty_result["cluster_count"] == 0,
                f"空入力 cluster_count=0 ({empty_result['cluster_count']})"),
        _assert(empty_result["diversity_score"] == 0.0,
                "空入力 diversity_score=0.0"),
        _assert(not_ready_result["cluster_count"] == 3,
                f"vector 未起動 cluster_count=N=3 ({not_ready_result['cluster_count']})"),
        _assert(abs(not_ready_result["diversity_score"] - 1.0) < 1e-9,
                "vector 未起動 diversity=1.0 (補正発火しない)"),
        _assert(none_result["cluster_count"] == 3,
                "_embed_sync=None cluster_count=N=3"),
        _assert(abs(none_result["diversity_score"] - 1.0) < 1e-9,
                "_embed_sync=None diversity=1.0"),
    ])


# ============================================================
# Step C — Dynamic β Lower Bound (EFE-Internal)
# ============================================================
# PLAN §5-4 検証要件 + ゆう要件「厚い identifying fixture」(条件 1):
# lower bound 違反 / upper bound / runaway / リセット / floor 経路を 7 case
# で deterministic 検証。

def test_dynamic_beta_base_below_trigger():
    """flat_streak < BETA_TRIGGER_STREAK で β = BETA_BASE (現挙動維持、
    対症療法回避の核心)。

    識別力: flat_streak >= 1 で β を上げる誤実装 (trigger 閾値ズレ) で fail。
    cig field 空 / flat_streak=0 / flat_streak=1 の 3 case 同時検証。
    """
    print("== Step C: flat_streak<2 で β=BETA_BASE (現挙動維持) ==")
    from core.predictor import _compute_dynamic_beta, BETA_BASE

    state_empty = {}
    beta_a = _compute_dynamic_beta(state_empty, {"tool": "reflect"})

    state_0 = {"cumulative_information_gain": {
        "flat_streak": 0, "e2_window_mean": 1.0, "e2_total": 100.0}}
    beta_b = _compute_dynamic_beta(state_0, {"tool": "reflect"})

    state_1 = {"cumulative_information_gain": {
        "flat_streak": 1, "e2_window_mean": 1.0, "e2_total": 100.0}}
    beta_c = _compute_dynamic_beta(state_1, {"tool": "reflect"})

    return all([
        _assert(beta_a == BETA_BASE, f"cig 空 β=BETA_BASE ({beta_a})"),
        _assert(beta_b == BETA_BASE, f"flat_streak=0 β=BETA_BASE ({beta_b})"),
        _assert(beta_c == BETA_BASE,
                f"flat_streak=1 (閾値直下) β=BETA_BASE ({beta_c})"),
    ])


def test_dynamic_beta_trigger_at_streak_2():
    """flat_streak >= BETA_TRIGGER_STREAK (=2) で β 動的増 (lower bound
    不等式 β >= e_h/i_avg を直訳)。

    識別力: BETA_TRIGGER_STREAK を 3 等ずらした誤実装、または flat_streak=2
    で β=BETA_BASE 維持の誤実装で fail。numerical e_h/i_avg 計算ミスでも fail。
    """
    print("== Step C: flat_streak>=2 で β 動的増 (lower bound trigger) ==")
    from core.predictor import _compute_dynamic_beta, BETA_BASE, BETA_CAP

    # window_mean=50, e2_total=500, N=10 → e_h=0.5, i_avg=(500/10)/100=0.5
    # β_required = 0.5/0.5 = 1.0
    state = {
        "cumulative_information_gain": {
            "flat_streak": 2, "e2_window_mean": 50.0, "e2_total": 500.0,
        },
        "prediction_error_history_e2": [50.0] * 10,
    }
    beta = _compute_dynamic_beta(state, {"tool": "reflect"})

    return all([
        _assert(beta > BETA_BASE, f"β > BETA_BASE ({beta} > {BETA_BASE})"),
        _assert(beta <= BETA_CAP, f"β <= BETA_CAP ({beta} <= {BETA_CAP})"),
        _assert(abs(beta - 1.0) < 1e-6,
                f"β = e_h/i_avg_norm = 1.0 ({beta})"),
    ])


def test_dynamic_beta_cap_runaway():
    """e_h 高 / i_avg 低の極端 case で β = BETA_CAP (runaway 抑制)。

    識別力: cap 制約を実装し忘れた誤実装で β >> 2.0 → fail。
    """
    print("== Step C: runaway pattern で β=BETA_CAP cap 発火 ==")
    from core.predictor import _compute_dynamic_beta, BETA_CAP

    # window_mean=100 (max), e2_total=1, N=100
    # e_h = max(0.01, 100/100) = 1.0
    # i_avg = max(0.01, 1/100) = 0.01, i_avg_norm = 0.01/100 = 0.0001
    # β_required = 1.0 / max(0.01, 0.0001) = 1.0/0.01 = 100 → cap=2.0
    # (Codex P3 fix: 旧コメント「10000」は誤、max(0.01, ...) clamp で 100)
    state = {
        "cumulative_information_gain": {
            "flat_streak": 5, "e2_window_mean": 100.0, "e2_total": 1.0,
        },
        "prediction_error_history_e2": [0.01] * 100,
    }
    beta = _compute_dynamic_beta(state, {"tool": "reflect"})

    return all([
        _assert(beta == BETA_CAP, f"β = BETA_CAP ({beta} == {BETA_CAP})"),
    ])


def test_dynamic_beta_lower_bound_clamped():
    """e_h < i_avg で β_required < BETA_BASE のとき max(BETA_BASE, ...) で
    BETA_BASE に clamped (情報利得が高い時に β 過剰減算を防ぐ)。

    識別力: max(BETA_BASE, ...) を抜いた誤実装で β < 0.5 → fail。
    """
    print("== Step C: β_required < BETA_BASE で BETA_BASE に clamped ==")
    from core.predictor import _compute_dynamic_beta, BETA_BASE

    # window_mean=10 (e_h=0.1), e2_total=10000, N=10 → i_avg=(10000/10)/100=10
    # β_required = 0.1/10 = 0.01 < 0.5 → max(0.5, 0.01) = 0.5
    state = {
        "cumulative_information_gain": {
            "flat_streak": 3, "e2_window_mean": 10.0, "e2_total": 10000.0,
        },
        "prediction_error_history_e2": [1000.0] * 10,
    }
    beta = _compute_dynamic_beta(state, {"tool": "reflect"})

    return all([
        _assert(beta == BETA_BASE,
                f"β = BETA_BASE clamped ({beta} == {BETA_BASE})"),
    ])


def test_predicted_outcome_multiplier_with_beta():
    """_predicted_outcome_multiplier が combined / beta で計算する end-to-end。
    flat_streak=0 (β=0.5) と flat_streak=2/β=1.0 で mult 抑制を比較。

    識別力: β を combined に乗算する誤実装 (× beta)、または beta 計算 skip
    で fail。numerical 値 1.6 / 0.8 で deterministic 検証。
    """
    print("== Step C: _predicted_outcome_multiplier の β 反映 ==")
    from core.controller import _predicted_outcome_multiplier

    cfg = {"predicted_e2_floor": 0.05}
    prediction = {"predicted_e2": 80}  # pe2_ratio=0.8、ec=None で combined=0.8
    candidate_a = {"tool": "reflect"}

    # case A: flat_streak=0 (β=0.5) → mult = max(0.05, 0.8/0.5) = 1.6
    state_a = {"predictor_confidence": {
        "reflect": {"e2_conf": 0.7, "ec_conf": 0.7}}}
    mult_a = _predicted_outcome_multiplier(prediction, candidate_a, state_a, cfg)

    # case B: flat_streak=2/β=1.0 → mult = max(0.05, 0.8/1.0) = 0.8
    candidate_b = {"tool": "reflect"}
    state_b = {
        "predictor_confidence": {"reflect": {"e2_conf": 0.7, "ec_conf": 0.7}},
        "cumulative_information_gain": {
            "flat_streak": 2, "e2_window_mean": 50.0, "e2_total": 500.0,
        },
        "prediction_error_history_e2": [50.0] * 10,
    }
    mult_b = _predicted_outcome_multiplier(prediction, candidate_b, state_b, cfg)

    return all([
        _assert(abs(mult_a - 1.6) < 1e-6,
                f"flat_streak=0: mult=1.6 ({mult_a})"),
        _assert(abs(mult_b - 0.8) < 1e-6,
                f"flat_streak=2/β=1.0: mult=0.8 ({mult_b})"),
        _assert(mult_b < mult_a,
                f"β 上昇で mult 抑制 ({mult_b} < {mult_a})"),
    ])


def test_predicted_outcome_multiplier_floor_with_beta():
    """β 動的化下でも floor が機能 (PLAN literal「predicted_e2_floor 触らない」
    整合)。combined / beta が floor 未満なら mult = floor で clamp。

    識別力: floor を抜いた誤実装、または β を floor 計算に巻き込む誤実装で fail。
    """
    print("== Step C: floor が β 下でも機能 ==")
    from core.controller import _predicted_outcome_multiplier

    cfg = {"predicted_e2_floor": 0.05}
    prediction = {"predicted_e2": 5}  # pe2_ratio=0.05
    candidate = {"tool": "reflect"}

    # β=BETA_CAP=2.0 で combined/β = 0.05/2.0 = 0.025 < floor → mult=0.05
    state = {
        "predictor_confidence": {"reflect": {"e2_conf": 0.7, "ec_conf": 0.7}},
        "cumulative_information_gain": {
            "flat_streak": 5, "e2_window_mean": 100.0, "e2_total": 1.0,
        },
        "prediction_error_history_e2": [0.01] * 100,
    }
    mult = _predicted_outcome_multiplier(prediction, candidate, state, cfg)

    return all([
        _assert(abs(mult - 0.05) < 1e-6,
                f"floor 0.05 適用 ({mult})"),
    ])


def test_dynamic_beta_penalty_includes_beta_value():
    """combined < 0.4 で penalty メッセージに β 値が含まれる (PLAN §5-2 literal)。

    Codex review P2-1 fix: 文字列 'beta=' のみ assertion から、
    具体 value 'beta=0.5' (基底) と 'beta=1.0' (動的 case) literal 強化。
    識別力: rounding 誤差 / 値計算ミス / β 計算 skip の誤実装で fail。
    """
    print("== Step C: penalty に β 値 literal 含む (基底 + 動的) ==")
    from core.controller import _predicted_outcome_multiplier

    cfg = {"predicted_e2_floor": 0.05}

    # case A: flat_streak=0 (β=BETA_BASE=0.5)
    candidate_a = {"tool": "reflect"}
    state_a = {
        "predictor_confidence": {"reflect": {"e2_conf": 0.7, "ec_conf": 0.7}},
    }
    # combined / β = 0.3 / 0.5 = 0.6、ただし combined=0.3 < 0.4 で penalty 追記
    _predicted_outcome_multiplier({"predicted_e2": 30}, candidate_a, state_a, cfg)
    penalties_a = candidate_a.get("penalties", [])

    # case B: flat_streak=2 (β=1.0、test_dynamic_beta_trigger_at_streak_2 同設定)
    candidate_b = {"tool": "reflect"}
    state_b = {
        "predictor_confidence": {"reflect": {"e2_conf": 0.7, "ec_conf": 0.7}},
        "cumulative_information_gain": {
            "flat_streak": 2, "e2_window_mean": 50.0, "e2_total": 500.0,
        },
        "prediction_error_history_e2": [50.0] * 10,
    }
    _predicted_outcome_multiplier({"predicted_e2": 30}, candidate_b, state_b, cfg)
    penalties_b = candidate_b.get("penalties", [])

    return all([
        _assert(len(penalties_a) > 0 and len(penalties_b) > 0,
                "両 case で penalty 追記される"),
        _assert(any("beta=0.5" in p for p in penalties_a),
                f"case A: penalty に 'beta=0.5' 含む ({penalties_a})"),
        _assert(any("beta=1.0" in p for p in penalties_b),
                f"case B: penalty に 'beta=1.0' 含む ({penalties_b})"),
    ])


def test_predicted_outcome_multiplier_ec_present_with_dynamic_beta():
    """predicted_ec あり + 不均等 conf + 動的 β の組合せで combined / β 計算
    が正しく走る end-to-end (Codex review P2-3 fix)。

    識別力: predicted_ec 経路で β を skip / combined 計算ミス / conf 重み
    無視の誤実装で fail。numerical literal で deterministic 検証。
    """
    print("== Step C: EC-present 動的 β 経路 end-to-end ==")
    from core.predictor import BETA_BASE
    from core.controller import _predicted_outcome_multiplier

    cfg = {"predicted_e2_floor": 0.05}
    candidate = {"tool": "reflect"}

    # 不均等 conf: e2_conf=0.9, ec_conf=0.3
    # pe2=80, pec=0.4 → combined = (0.8*0.9 + 0.4*0.3) / 1.2 = (0.72 + 0.12)/1.2 = 0.7
    # flat_streak=2, e_h=0.5, i_avg=0.5 → β = 1.0
    # mult = max(0.05, 0.7 / 1.0) = 0.7
    state = {
        "predictor_confidence": {"reflect": {"e2_conf": 0.9, "ec_conf": 0.3}},
        "cumulative_information_gain": {
            "flat_streak": 2, "e2_window_mean": 50.0, "e2_total": 500.0,
        },
        "prediction_error_history_e2": [50.0] * 10,
    }
    prediction = {"predicted_e2": 80, "predicted_ec": 0.4}
    mult_dynamic = _predicted_outcome_multiplier(prediction, candidate, state, cfg)

    # 比較対照: 同じ state から cig 抜いた case (flat_streak=0 → β=BETA_BASE=0.5)
    state_base = {
        "predictor_confidence": {"reflect": {"e2_conf": 0.9, "ec_conf": 0.3}},
    }
    candidate_base = {"tool": "reflect"}
    mult_base = _predicted_outcome_multiplier(prediction, candidate_base, state_base, cfg)

    # 数値: combined ≈ 0.7
    # base: 0.7 / 0.5 = 1.4
    # dynamic: 0.7 / 1.0 = 0.7
    return all([
        _assert(abs(mult_base - 0.7 / BETA_BASE) < 1e-9,
                f"base β=0.5 で mult=1.4 ({mult_base})"),
        _assert(abs(mult_dynamic - 0.7) < 1e-9,
                f"dynamic β=1.0 で mult=0.7 ({mult_dynamic})"),
        _assert(mult_dynamic < mult_base,
                f"動的 β で抑制 ({mult_dynamic} < {mult_base})"),
    ])


# ============================================================
# main runner
# ============================================================

def main():
    tests = [
        test_cig_total_monotonic,
        test_cig_flat_signal_threshold,
        test_cig_flat_streak_increment_and_reset,
        test_cig_end_to_end_via_update_predictor_confidence,
        test_attractor_redundancy_all_orthogonal,
        test_attractor_redundancy_all_same,
        test_attractor_redundancy_partial,
        test_attractor_redundancy_state_persist_json_safe,
        test_attractor_redundancy_malformed_vecs_fallback,
        test_attractor_redundancy_empty_and_fallback,
        test_dynamic_beta_base_below_trigger,
        test_dynamic_beta_trigger_at_streak_2,
        test_dynamic_beta_cap_runaway,
        test_dynamic_beta_lower_bound_clamped,
        test_predicted_outcome_multiplier_with_beta,
        test_predicted_outcome_multiplier_floor_with_beta,
        test_dynamic_beta_penalty_includes_beta_value,
        test_predicted_outcome_multiplier_ec_present_with_dynamic_beta,
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
