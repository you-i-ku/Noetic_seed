"""段階13 Phase 4 commit 1 — graph_maturity 関数 + 5 sub-score 識別力 test。

CLAUDE.md §5 識別力 + §6 docstring 同期 literal 適用。

想定誤実装 D1-D7 cover:
    D1: sigmoid clamp 抜け (overflow で NaN/Inf)
    D2: WEIGHT_X swap — uniform 0.2 で symbol swap は数学的に invisible のため、
        constants を一時 patch で non-uniform 化 + 一意 sub-score で contract 検証
    D3: frontier に degree=0 (isolated) 含む (B-1 誤実装) —
        load_all_memories monkey patch で isolated memory 仕込み、結果不変 assert
    D4: link_type="none" sanitize 漏れ — **avg_strength / _frontier_count 側のみ適用**
        (density は意図的に raw count、PLAN §3-2 "link 量の純粋指標" literal 仕様)
    D5: anomaly_rate を sum 計算 (mean じゃなく)
    D6: small_world_sigma に baseline 引き忘れ
    D7: avg_strength の _link_strength helper 不使用 (confidence fallback ない)

Graceful skip 統一 (PLAN §3-3 整合、ゆう gut 確定 2026-05-03):
    全 5 sub-score で「データなし → 0.0 寄与ゼロ」pattern。空 graph で
    graph_maturity = 0.0 = pressure 軸完全支配 (PLAN §3-3 literal 整合)。

fixture 4 種 (識別力):
    1. 空 graph: 全 sub-score=0.0 graceful skip (PLAN §3-3 整合)
    2. leaf chain: A→B→C で leaf=2、isolated 除外確認
    3. triangle: 全 degree=2 で leaf=0 (well-connected)
    4. high state: 飽和 graph (maturity > 0.8)

使い方:
    cd Noetic_seed/profiles/_template
    python tests/test_graph_maturity.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

# standalone 直接実行時 (Windows cp932 console) でも Unicode 文字 (≈, →, etc.)
# を出力できるよう UTF-8 reconfigure (test_predictor_jepa.py / run_tests.py と同 pattern)。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, OSError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.memory_links as ml
import core.memory as cm
import core.tag_emergence_monitor as tem
from core.dynamic_composition import (
    _sigmoid,
    _density_score,
    _structure_score,
    _anomaly_score,
    _frontier_count,
    _frontier_score,
    _avg_strength_score,
    compute_graph_maturity,
    N_NORMALIZE_LINK,
    N_NORMALIZE_FRONTIER,
    SMALL_WORLD_BASELINE,
    ANOMALY_HISTORY_WINDOW,
    WEIGHT_DENSITY,
    WEIGHT_STRUCTURE,
    WEIGHT_ANOMALY,
    WEIGHT_FRONTIER,
    WEIGHT_AVG_STRENGTH,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup_tmp_memory():
    """tmp dir を memory_links.MEMORY_DIR に差し替え。返り値で原値を保持。"""
    tmp = Path(tempfile.mkdtemp(prefix="noetic_graph_maturity_"))
    orig_ml = ml.MEMORY_DIR
    ml.MEMORY_DIR = tmp
    return tmp, orig_ml


def _restore_tmp_memory(tmp, orig_ml):
    ml.MEMORY_DIR = orig_ml
    shutil.rmtree(tmp, ignore_errors=True)


def _write_links(tmp, links):
    """memory_links.jsonl を tmp dir に直接 write。"""
    fpath = tmp / ml.LINK_FILE_NAME
    with open(fpath, "w", encoding="utf-8") as f:
        for link in links:
            f.write(json.dumps(link, ensure_ascii=False) + "\n")


def _patch_helpers(memory_count: int, sigma: float):
    """load_all_memories + _compute_small_world_metrics の monkey patch。

    Returns:
        (orig_lam, orig_sw): 復元用 tuple
    """
    orig_lam = cm.load_all_memories
    orig_sw = tem._compute_small_world_metrics
    cm.load_all_memories = lambda: [{"id": f"m{i}"} for i in range(memory_count)]
    tem._compute_small_world_metrics = lambda links, n: {
        "avg_path_length": 1.0,
        "small_world_sigma": sigma,
        "small_world_C_random": 0.0,
        "small_world_L_random": 0.0,
    }
    return orig_lam, orig_sw


def _restore_helpers(orig_lam, orig_sw):
    cm.load_all_memories = orig_lam
    tem._compute_small_world_metrics = orig_sw


# ============================================================
# D1: sigmoid clamp & 範囲 [0, 1]
# ============================================================

def test_sigmoid_clamp_overflow():
    """D1: 大きな正/負 input でも overflow せず [0, 1] 範囲、NaN/Inf なし。"""
    print("== D1: _sigmoid clamp overflow ==")
    return all([
        _assert(_sigmoid(0.0) == 0.5, "sigmoid(0)=0.5"),
        _assert(_sigmoid(1000.0) == 1.0, "sigmoid(+inf-like)=1.0 (clamp)"),
        _assert(_sigmoid(-1000.0) == 0.0, "sigmoid(-inf-like)=0.0 (clamp)"),
        _assert(0.0 <= _sigmoid(50) <= 1.0, "sigmoid(50)=clamp 境界"),
        _assert(0.0 <= _sigmoid(-50) <= 1.0, "sigmoid(-50)=clamp 境界"),
    ])


# ============================================================
# D5: anomaly_score = sigmoid(mean(history))、sum じゃない
# ============================================================

def test_anomaly_score_empty_history():
    """history 空 → 0.0 graceful skip (PLAN §3-3 整合、寄与ゼロ)。"""
    print("== anomaly_score: empty history ==")
    state = {"jepa_prediction_error_history": []}
    score = _anomaly_score(state)
    return _assert(score == 0.0, f"empty → 0.0 (actual {score})")


def test_anomaly_score_state_missing():
    """state field 欠落でも graceful → 0.0 (PLAN §3-3 整合)。"""
    print("== anomaly_score: state field missing ==")
    score = _anomaly_score({})
    return _assert(score == 0.0, f"missing field → 0.0 (actual {score})")


def test_anomaly_score_mean_not_sum():
    """D5 識別: mean 計算 (sum じゃない)。

    history=[0.1, 0.2, 0.3] → mean=0.2 → sigmoid(0.2)≈0.5498
    sum 誤実装なら sum=0.6 → sigmoid(0.6)≈0.6457 で別値、識別可能。
    """
    print("== anomaly_score: mean (D5 識別) ==")
    state = {"jepa_prediction_error_history": [0.1, 0.2, 0.3]}
    score = _anomaly_score(state)
    expected_mean = _sigmoid(0.2)
    expected_sum = _sigmoid(0.6)
    return all([
        _assert(abs(score - expected_mean) < 1e-9,
                f"score≈sigmoid(mean=0.2)={expected_mean:.6f} (actual {score:.6f})"),
        _assert(abs(score - expected_sum) > 1e-3,
                "sum 誤実装と識別可能"),
    ])


def test_anomaly_score_window_truncation():
    """ANOMALY_HISTORY_WINDOW より長い history は直近 N のみ使う。"""
    print("== anomaly_score: window truncation ==")
    history = [0.0] * 100 + [1.0] * ANOMALY_HISTORY_WINDOW
    state = {"jepa_prediction_error_history": history}
    score = _anomaly_score(state)
    expected = _sigmoid(1.0)
    return _assert(abs(score - expected) < 1e-9,
                   f"直近 {ANOMALY_HISTORY_WINDOW} のみ → sigmoid(1.0)≈{expected:.6f} (actual {score:.6f})")


# ============================================================
# D3 + D4: frontier_count = leaf (degree=1)、isolated/none link 除外
# ============================================================

def test_frontier_count_empty_graph():
    """空 graph → frontier=0。"""
    print("== frontier_count: empty graph ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        _write_links(tmp, [])
        return _assert(_frontier_count({}) == 0, "空 graph → 0")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_frontier_count_leaf_chain():
    """A→B→C 連鎖 → A,C が leaf (degree=1)、B は中間 (degree=2)、frontier=2。"""
    print("== frontier_count: A→B→C 連鎖 ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "A", "to_id": "B", "link_type": "supporting"},
            {"from_id": "B", "to_id": "C", "link_type": "supporting"},
        ]
        _write_links(tmp, links)
        result = _frontier_count({})
        return _assert(result == 2, f"A,C が leaf → 2 (actual {result})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_frontier_count_excludes_isolated():
    """D3 識別 (強化版): isolated memory が fixture にあっても結果不変。

    A→B link + C は memory として存在するが link なし (isolated)。
    現実装は link 経路のみで degree dict 構築 = isolated C は自然除外、結果=2。
    将来「load_all_memories から isolated memory を degree=0 で frontier に含める」
    誤実装が混入した場合、C も count されて 3 になる → 本 test が catch する。

    fixture に isolated memory を materialize する強化により、Codex review
    指摘 (D3 が actually tested じゃない) を fix。
    """
    print("== frontier_count: isolated memory 仕込み + 除外 (D3 識別、強化版) ==")
    tmp, orig_ml = _setup_tmp_memory()
    # isolated memory C を含む 3 memory 仕込み (load_all_memories 経由で見える)
    orig_lam = cm.load_all_memories
    cm.load_all_memories = lambda: [{"id": "A"}, {"id": "B"}, {"id": "C"}]
    try:
        links = [{"from_id": "A", "to_id": "B", "link_type": "supporting"}]
        _write_links(tmp, links)
        result = _frontier_count({})
        return _assert(result == 2,
                       f"A,B leaf、isolated C 除外 → 2 (actual {result})")
    finally:
        cm.load_all_memories = orig_lam
        _restore_tmp_memory(tmp, orig_ml)


def test_frontier_count_excludes_none_links():
    """D4 識別: link_type='none' は frontier 算出から除外。"""
    print("== frontier_count: link_type='none' 除外 (D4 識別) ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "A", "to_id": "B", "link_type": "none"},
            {"from_id": "C", "to_id": "D", "link_type": "supporting"},
        ]
        _write_links(tmp, links)
        result = _frontier_count({})
        return _assert(result == 2, f"none 除外で C,D の 2 (actual {result})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_frontier_count_self_loop_excluded():
    """from==to (self loop) は除外 (small_world と同 sanitize 規則)。"""
    print("== frontier_count: self loop 除外 ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "A", "to_id": "A", "link_type": "supporting"},
            {"from_id": "B", "to_id": "C", "link_type": "supporting"},
        ]
        _write_links(tmp, links)
        result = _frontier_count({})
        return _assert(result == 2, f"self loop 除外で B,C の 2 (actual {result})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_frontier_count_duplicate_and_reciprocal_links():
    """P2-2 識別: 重複 link / 相互 link でも adjacency-set でユニーク化 → leaf 正確。

    Codex review (2 周目) 指摘 fix: link entry ごとに degree++ する旧実装は
    A→B 2 回 / 相互 link で degree=2 となり leaf 検出失敗、small_world と divergent。
    adjacency-set pattern (set でユニーク化) に揃えると、A の隣接 set={B}、
    B の隣接 set={A}、両方 degree=1 = leaf 正確。
    """
    print("== frontier_count: 重複/相互 link でも leaf 正確 (P2-2 識別) ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            # 重複: A→B が 2 回
            {"from_id": "A", "to_id": "B", "link_type": "supporting"},
            {"from_id": "A", "to_id": "B", "link_type": "semantic"},
            # 相互: C↔D
            {"from_id": "C", "to_id": "D", "link_type": "supporting"},
            {"from_id": "D", "to_id": "C", "link_type": "supporting"},
        ]
        _write_links(tmp, links)
        result = _frontier_count({})
        # 正実装 (adj set): A,B,C,D 全部 degree=1 (相手 1 つだけ) → 4
        # 旧実装 (link entry++): A=2, B=2, C=2, D=2 → leaf 0
        return all([
            _assert(result == 4,
                    f"adj-set pattern: A,B,C,D 全部 leaf → 4 (actual {result})"),
            _assert(result != 0, "旧 entry++ 実装の 0 と識別 (P2-2)"),
        ])
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_frontier_count_triangle_no_leaf():
    """A↔B↔C↔A の三角形 → 全 node degree=2、leaf 0。"""
    print("== frontier_count: triangle (no leaf) ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "A", "to_id": "B", "link_type": "supporting"},
            {"from_id": "B", "to_id": "C", "link_type": "supporting"},
            {"from_id": "C", "to_id": "A", "link_type": "supporting"},
        ]
        _write_links(tmp, links)
        result = _frontier_count({})
        return _assert(result == 0, f"triangle → leaf 0 (actual {result})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


# ============================================================
# density_score
# ============================================================

def test_density_score_empty():
    """link 0 → 0.0 graceful skip (PLAN §3-3 整合、寄与ゼロ)。"""
    print("== density_score: empty ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        _write_links(tmp, [])
        score = _density_score({})
        return _assert(score == 0.0, f"empty → 0.0 (actual {score})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_density_score_at_normalize():
    """link N_NORMALIZE_LINK 本 → sigmoid(1.0)≈0.731 (raw count、'none' も含む)。"""
    print("== density_score: at N_NORMALIZE_LINK ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = (
            [{"from_id": f"a{i}", "to_id": f"b{i}", "link_type": "supporting"}
             for i in range(50)]
            + [{"from_id": f"a{i}", "to_id": f"b{i}", "link_type": "none"}
               for i in range(50)]
        )
        _write_links(tmp, links)
        score = _density_score({})
        expected = _sigmoid(N_NORMALIZE_LINK / N_NORMALIZE_LINK)
        return _assert(abs(score - expected) < 1e-9,
                       f"100 link → sigmoid(1)≈{expected:.6f} (actual {score:.6f})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


# ============================================================
# avg_strength_score (D7 識別)
# ============================================================

def test_avg_strength_score_empty():
    """link 0 → 0.0 graceful skip (PLAN §3-3 整合、寄与ゼロ)。"""
    print("== avg_strength_score: empty ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        _write_links(tmp, [])
        score = _avg_strength_score({})
        return _assert(score == 0.0, f"empty → 0.0 (actual {score})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_avg_strength_score_confidence_fallback():
    """D7 識別: strength field なし、confidence のみ link で fallback。

    confidence=0.7 のみ link → mean=0.7 → sigmoid(0.7)≈0.668。
    raw "strength" 直読 (fallback なし) 誤実装だと 0.0 → sigmoid(0)=0.5、識別可能。
    """
    print("== avg_strength_score: confidence fallback (D7 識別) ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "A", "to_id": "B", "link_type": "supporting", "confidence": 0.7},
        ]
        _write_links(tmp, links)
        score = _avg_strength_score({})
        expected = _sigmoid(0.7)
        return all([
            _assert(abs(score - expected) < 1e-9,
                    f"confidence fallback → sigmoid(0.7)≈{expected:.6f} (actual {score:.6f})"),
            _assert(abs(score - 0.5) > 0.01, "raw strength 誤実装の 0.5 と識別"),
        ])
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_avg_strength_score_excludes_none():
    """link_type='none' は集計外。"""
    print("== avg_strength_score: 'none' 除外 ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "A", "to_id": "B", "link_type": "supporting", "strength": 0.4},
            {"from_id": "C", "to_id": "D", "link_type": "none", "strength": 100.0},
        ]
        _write_links(tmp, links)
        score = _avg_strength_score({})
        expected = _sigmoid(0.4)
        return _assert(abs(score - expected) < 1e-9,
                       f"'none' 除外で mean=0.4 → sigmoid(0.4)≈{expected:.6f} (actual {score:.6f})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


# ============================================================
# structure_score (D6 識別)
# ============================================================

def test_structure_score_baseline_subtraction():
    """D6 識別: small_world_sigma に baseline 引く動作。

    sigma=baseline=1.0 → structure = sigmoid(0) = 0.5。
    引き忘れ誤実装なら sigmoid(1.0)≈0.731、識別可能。

    note: graceful skip 経路 (links 0 or memory<2) を回避するため
    fixture に link 1 本仕込み (_compute_small_world_metrics は monkey patch
    で sigma=1.0 固定、links の実値は metrics 計算に影響しない)。
    """
    print("== structure_score: baseline subtraction (D6 識別) ==")
    tmp, orig_ml = _setup_tmp_memory()
    orig_lam, orig_sw = _patch_helpers(memory_count=5, sigma=1.0)
    try:
        _write_links(tmp, [
            {"from_id": "A", "to_id": "B", "link_type": "supporting"},
        ])
        score = _structure_score({})
        miss_baseline = _sigmoid(1.0)
        return all([
            _assert(abs(score - 0.5) < 1e-9,
                    f"sigma=baseline → 0.5 (actual {score:.6f})"),
            _assert(abs(score - miss_baseline) > 0.01,
                    f"baseline 引き忘れの sigmoid(1)≈{miss_baseline:.6f} と識別"),
        ])
    finally:
        _restore_helpers(orig_lam, orig_sw)
        _restore_tmp_memory(tmp, orig_ml)


def test_structure_score_above_baseline():
    """sigma=2.0 → sigmoid(2.0 - 1.0) = sigmoid(1)≈0.731。"""
    print("== structure_score: sigma > baseline ==")
    tmp, orig_ml = _setup_tmp_memory()
    orig_lam, orig_sw = _patch_helpers(memory_count=5, sigma=2.0)
    try:
        _write_links(tmp, [
            {"from_id": "A", "to_id": "B", "link_type": "supporting"},
        ])
        score = _structure_score({})
        expected = _sigmoid(1.0)
        return _assert(abs(score - expected) < 1e-9,
                       f"sigma=2 → sigmoid(1)≈{expected:.6f} (actual {score:.6f})")
    finally:
        _restore_helpers(orig_lam, orig_sw)
        _restore_tmp_memory(tmp, orig_ml)


def test_structure_score_sentinel_graceful_skip():
    """既存 _compute_small_world_metrics の sentinel 検出時は 0.0 graceful skip。

    PLAN §3-3 整合: 「空 graph で graph_maturity ≈ 0.0」literal を実装で純粋表現。
    sigmoid(-1.0) ≈ 0.27 (random 以下の構造度) として誤読しない構造的ガード。
    """
    print("== structure_score: sentinel → 0.0 graceful skip ==")
    tmp, orig_ml = _setup_tmp_memory()
    # memory_count=0 + links 空 = 既存 _compute_small_world_metrics の sentinel 条件
    orig_lam, orig_sw = _patch_helpers(memory_count=0, sigma=0.0)
    try:
        _write_links(tmp, [])
        score = _structure_score({})
        miss_sentinel = _sigmoid(-1.0)  # sentinel 誤読時の値
        return all([
            _assert(score == 0.0,
                    f"sentinel → 0.0 graceful skip (actual {score})"),
            _assert(abs(score - miss_sentinel) > 0.01,
                    f"sentinel 誤読の sigmoid(-1)≈{miss_sentinel:.6f} と識別"),
        ])
    finally:
        _restore_helpers(orig_lam, orig_sw)
        _restore_tmp_memory(tmp, orig_ml)


def test_structure_score_all_sanitized_out():
    """P2-1 識別: links はあるが全部 sanitize-out (link_type='none' / self-loop)
    のとき、helper が sigma=0.0 を return → structure_score 側で 0.0 graceful skip。

    Codex review (2 周目) 指摘 fix: 早期 (i) graceful skip は通過してしまう
    けど、`sigma == 0.0` sentinel 検出で (ii)(iii) sanitize-out 経路も catch。
    sigmoid(-1) ≈ 0.27 になる誤実装と識別。
    """
    print("== structure_score: 全 sanitize-out → 0.0 (P2-1 識別) ==")
    tmp, orig_ml = _setup_tmp_memory()
    # memory はあるが links は全 sanitize-out: 早期 (i) check 通過 →
    # _compute_small_world_metrics 内 sanitize で valid_link_count=0 → sigma=0.0
    orig_lam, orig_sw = _patch_helpers(memory_count=5, sigma=0.0)
    try:
        # 全部 sanitize-out 対象 ('none' / self-loop / endpoint 欠落)
        links = [
            {"from_id": "A", "to_id": "B", "link_type": "none"},
            {"from_id": "C", "to_id": "C", "link_type": "supporting"},  # self-loop
            {"from_id": "", "to_id": "D", "link_type": "supporting"},   # endpoint 欠落
        ]
        _write_links(tmp, links)
        score = _structure_score({})
        miss_sentinel = _sigmoid(-1.0)
        return all([
            _assert(score == 0.0,
                    f"全 sanitize-out → 0.0 graceful skip (actual {score})"),
            _assert(abs(score - miss_sentinel) > 0.01,
                    f"sentinel 誤読の sigmoid(-1)≈{miss_sentinel:.6f} と識別 (P2-1)"),
        ])
    finally:
        _restore_helpers(orig_lam, orig_sw)
        _restore_tmp_memory(tmp, orig_ml)


# ============================================================
# D2: weight 合計 == 1.0 + uniform 検証
# ============================================================

def test_weights_sum_to_one():
    """D2 識別: 5 weight 合計が 1.0、maturity の上限 1.0 保証。"""
    print("== weights: sum == 1.0 ==")
    total = (WEIGHT_DENSITY + WEIGHT_STRUCTURE + WEIGHT_ANOMALY +
             WEIGHT_FRONTIER + WEIGHT_AVG_STRENGTH)
    return _assert(abs(total - 1.0) < 1e-9, f"sum={total} (expected 1.0)")


def test_weights_uniform():
    """D2 補強: 全 weight 0.2 で uniform (PLAN §3-2 weighted_avg literal)。"""
    print("== weights: uniform (0.2 each) ==")
    return all([
        _assert(WEIGHT_DENSITY == 0.2, "density=0.2"),
        _assert(WEIGHT_STRUCTURE == 0.2, "structure=0.2"),
        _assert(WEIGHT_ANOMALY == 0.2, "anomaly=0.2"),
        _assert(WEIGHT_FRONTIER == 0.2, "frontier=0.2"),
        _assert(WEIGHT_AVG_STRENGTH == 0.2, "avg_strength=0.2"),
    ])


def test_weight_assignment_contract():
    """D2 識別 (構造契約検証): 各 sub-score に正しい weight が紐付くこと。

    uniform 0.2 spec のもとでは symbol swap が数学的に invisible のため、
    constants を一時 patch で non-uniform 化 + 各 sub-score を一意の固定値
    monkey patch、graph_maturity 出力を literal 計算と比較する形で contract 検証。
    WEIGHT_DENSITY と WEIGHT_FRONTIER を swap した実装は結果が一致しなくなる。

    PLAN §3-2 uniform 0.2 spec は test_weights_uniform で別途検証、本 test は
    swap 識別力の構造的ガード (Codex review 指摘 D2 fix)。
    """
    print("== compute_graph_maturity: weight assignment contract (D2 識別) ==")
    import core.dynamic_composition as dc

    # backup
    orig_d_w = dc.WEIGHT_DENSITY
    orig_s_w = dc.WEIGHT_STRUCTURE
    orig_a_w = dc.WEIGHT_ANOMALY
    orig_f_w = dc.WEIGHT_FRONTIER
    orig_as_w = dc.WEIGHT_AVG_STRENGTH
    orig_d = dc._density_score
    orig_s = dc._structure_score
    orig_a = dc._anomaly_score
    orig_fs = dc._frontier_score
    orig_as = dc._avg_strength_score

    try:
        # non-uniform weights (test 中のみ、各 weight 一意の値)
        dc.WEIGHT_DENSITY = 0.5
        dc.WEIGHT_STRUCTURE = 0.1
        dc.WEIGHT_ANOMALY = 0.1
        dc.WEIGHT_FRONTIER = 0.2
        dc.WEIGHT_AVG_STRENGTH = 0.1

        # 一意 sub-score 固定値 (density のみ 1.0、他 0)
        dc._density_score = lambda s: 1.0
        dc._structure_score = lambda s: 0.0
        dc._anomaly_score = lambda s: 0.0
        dc._frontier_score = lambda s: 0.0
        dc._avg_strength_score = lambda s: 0.0

        result = dc.compute_graph_maturity({})
        # 期待: 0.5 * 1.0 + 0.1 * 0.0 + 0.1 * 0.0 + 0.2 * 0.0 + 0.1 * 0.0 = 0.5
        # WEIGHT_DENSITY と WEIGHT_FRONTIER swap 誤実装なら 0.2 * 1.0 = 0.2
        expected_correct = 0.5
        miss_swap = 0.2
        return all([
            _assert(abs(result - expected_correct) < 1e-9,
                    f"WEIGHT_DENSITY と _density_score 紐付け契約 → "
                    f"{expected_correct} (actual {result:.6f})"),
            _assert(abs(result - miss_swap) > 0.01,
                    f"DENSITY/FRONTIER swap 誤実装の {miss_swap} と識別 (D2)"),
        ])
    finally:
        dc.WEIGHT_DENSITY = orig_d_w
        dc.WEIGHT_STRUCTURE = orig_s_w
        dc.WEIGHT_ANOMALY = orig_a_w
        dc.WEIGHT_FRONTIER = orig_f_w
        dc.WEIGHT_AVG_STRENGTH = orig_as_w
        dc._density_score = orig_d
        dc._structure_score = orig_s
        dc._anomaly_score = orig_a
        dc._frontier_score = orig_fs
        dc._avg_strength_score = orig_as


def test_frontier_score_at_normalize():
    """W2: frontier_score の sigmoid 正規化を direct 検証。

    leaf=N_NORMALIZE_FRONTIER=20 → sigmoid(1.0)≈0.731 を直接検証。
    `_frontier_count` 経路は test_frontier_count_* で詳細覆ってる、
    本 test は sigmoid 通過部分の direct 検証 (W2 識別力補強)。
    """
    print("== frontier_score: at N_NORMALIZE_FRONTIER (W2 direct) ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        # hub-spoke で leaf 20 個ちょうど作る (hub 1 が degree=20、leaf{0..19} が degree=1)
        links = [
            {"from_id": "hub", "to_id": f"leaf{i}", "link_type": "supporting"}
            for i in range(20)
        ]
        _write_links(tmp, links)
        score = _frontier_score({})
        expected = _sigmoid(1.0)
        return _assert(abs(score - expected) < 1e-9,
                       f"leaf=20 → sigmoid(1)≈{expected:.6f} (actual {score:.6f})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


# ============================================================
# compute_graph_maturity 統合 (5 sub-score 既知値で literal 計算)
# ============================================================

def test_graph_maturity_empty_state():
    """空 graph + 空 history → 全 sub-score=0.0 graceful skip → maturity=0.0。

    PLAN §3-3 literal「空 graph で graph_maturity ≈ 0.0、pressure 軸のみ動く」
    完全整合。空 graph で graph 軸寄与ゼロ = pressure 軸完全支配。

    全 sub-score graceful skip:
        density   = 0.0  (links 空)
        structure = 0.0  (sentinel: not links or memory<2)
        anomaly   = 0.0  (history 空)
        frontier  = 0.0  (frontier_count == 0)
        avg_str   = 0.0  (valid_strengths 空)
        maturity  = 0.2 * 0 * 5 = 0.0
    """
    print("== graph_maturity: empty state → 0.0 (PLAN §3-3 literal) ==")
    tmp, orig_ml = _setup_tmp_memory()
    orig_lam, orig_sw = _patch_helpers(memory_count=0, sigma=0.0)
    try:
        _write_links(tmp, [])
        maturity = compute_graph_maturity({})
        return all([
            _assert(0.0 <= maturity <= 1.0, f"範囲 [0,1]: {maturity:.6f}"),
            _assert(maturity == 0.0,
                    f"empty → 0.0 (PLAN §3-3 整合、actual {maturity:.6f})"),
        ])
    finally:
        _restore_helpers(orig_lam, orig_sw)
        _restore_tmp_memory(tmp, orig_ml)


def test_graph_maturity_high_state():
    """飽和 graph (link 多 / leaf 多 / strength 高 / sigma 高 / anomaly 高)
    → maturity > 0.8。
    """
    print("== graph_maturity: high state ==")
    tmp, orig_ml = _setup_tmp_memory()
    orig_lam, orig_sw = _patch_helpers(memory_count=50, sigma=5.0)
    try:
        # hub-spoke 構造: hub から 50 leaf
        links = [
            {"from_id": "hub", "to_id": f"leaf{i}",
             "link_type": "supporting", "strength": 0.9}
            for i in range(50)
        ]
        _write_links(tmp, links)
        state = {"jepa_prediction_error_history": [3.0] * 30}
        maturity = compute_graph_maturity(state)
        return _assert(maturity > 0.8,
                       f"high state → maturity > 0.8 (actual {maturity:.6f})")
    finally:
        _restore_helpers(orig_lam, orig_sw)
        _restore_tmp_memory(tmp, orig_ml)


def test_graph_maturity_range_invariant():
    """compute_graph_maturity は常に [0, 1] 範囲 (D1 補強)。"""
    print("== graph_maturity: range invariant ==")
    tmp, orig_ml = _setup_tmp_memory()
    orig_lam, orig_sw = _patch_helpers(memory_count=100, sigma=1000.0)
    try:
        links = [
            {"from_id": f"a{i}", "to_id": f"b{i}",
             "link_type": "supporting", "strength": 100.0}
            for i in range(10000)
        ]
        _write_links(tmp, links)
        state = {"jepa_prediction_error_history": [1000.0] * 100}
        maturity = compute_graph_maturity(state)
        return _assert(0.0 <= maturity <= 1.0,
                       f"極端 input でも [0,1]: {maturity:.6f}")
    finally:
        _restore_helpers(orig_lam, orig_sw)
        _restore_tmp_memory(tmp, orig_ml)


# ============================================================
# main
# ============================================================

def main():
    tests = [
        test_sigmoid_clamp_overflow,
        test_anomaly_score_empty_history,
        test_anomaly_score_state_missing,
        test_anomaly_score_mean_not_sum,
        test_anomaly_score_window_truncation,
        test_frontier_count_empty_graph,
        test_frontier_count_leaf_chain,
        test_frontier_count_excludes_isolated,
        test_frontier_count_excludes_none_links,
        test_frontier_count_self_loop_excluded,
        test_frontier_count_duplicate_and_reciprocal_links,
        test_frontier_count_triangle_no_leaf,
        test_density_score_empty,
        test_density_score_at_normalize,
        test_avg_strength_score_empty,
        test_avg_strength_score_confidence_fallback,
        test_avg_strength_score_excludes_none,
        test_structure_score_baseline_subtraction,
        test_structure_score_above_baseline,
        test_structure_score_sentinel_graceful_skip,
        test_structure_score_all_sanitized_out,
        test_frontier_score_at_normalize,
        test_weights_sum_to_one,
        test_weights_uniform,
        test_weight_assignment_contract,
        test_graph_maturity_empty_state,
        test_graph_maturity_high_state,
        test_graph_maturity_range_invariant,
    ]
    results = []
    for t in tests:
        try:
            result = t()
            results.append(bool(result))
        except Exception as e:
            print(f"  [FAIL] {t.__name__} raised: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()
    passed = sum(results)
    total = len(results)
    print(f"=== {passed}/{total} groups passed ===")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
