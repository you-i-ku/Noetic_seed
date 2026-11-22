"""段階13 Phase 4 commit 3 — graph 軸 fire 候補 4 種 + 統合 entry 識別力 test。

CLAUDE.md §5 識別力 + §6 docstring 同期 literal 適用。

検証対象 (PLAN §3-4 (a)-(d) + §3-1 重み付け):
    _anomaly_candidates              : 予測誤差大の状態シグナル (1 candidate)
    _structural_tension_candidates   : cluster 間 link 不均衡 (phase6_metrics 流用)
    _relational_salience_candidates  : hub 性 (degree / avg_degree 相対比)
    _predictive_frontier_candidates  : leaf node (degree=1、commit 1 流用)
    compute_graph_axis_candidates    : 4 種統合 + w_graph multiplicative 重み付け

想定誤実装 (識別力 fixture):
    GAC1: anomaly が history 全体じゃなく直近 1 entry のみ参照
    GAC2: structural_tension の inter_ratio 反転 (高 ratio で緊張高い誤認)
    GAC3: relational_salience が absolute degree (相対比じゃなく直値) で判定
    GAC4: predictive_frontier が isolated (degree=0) も含める
    GAC5: 統合 entry が w_graph 未適用 (raw score そのまま return)
    GAC6: 統合 entry が w_graph=0 でも空じゃない list 返す
    GAC7: kind="graph" field 抜け / name format 崩れ

fixture 4 種 (識別力):
    1. 空 graph: 全 candidate 種で空 list (PLAN §3-3 整合)
    2. hub-spoke: relational_salience + predictive_frontier 同時発火
    3. triangle: 4 種すべて発火条件外
    4. high anomaly: anomaly 単独発火

使い方:
    cd Noetic_seed/profiles/_template
    python tests/test_graph_axis_candidates.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

# standalone 直接実行時 (Windows cp932 console) でも Unicode 文字を出力
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, OSError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.memory_links as ml
import core.memory as cm
from core.dynamic_composition import (
    _sigmoid,
    _anomaly_candidates,
    _structural_tension_candidates,
    _relational_salience_candidates,
    _predictive_frontier_candidates,
    compute_graph_axis_candidates,
    ANOMALY_HISTORY_WINDOW,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup_tmp_memory():
    """tmp dir を memory_links.MEMORY_DIR に差し替え、原値を保持。"""
    tmp = Path(tempfile.mkdtemp(prefix="noetic_graph_axis_"))
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


# ============================================================
# (a) anomaly_candidates: GAC1 識別 (history 全体 mean、1 entry のみじゃない)
# ============================================================

def test_anomaly_candidates_empty_history():
    """history 空 → 空 list graceful skip (PLAN §3-3 整合)。"""
    print("== anomaly_candidates: empty history ==")
    state = {"jepa_prediction_error_history": []}
    result = _anomaly_candidates(state)
    return _assert(result == [], f"empty history → [] (actual {result})")


def test_anomaly_candidates_state_missing():
    """state field 欠落 → 空 list。"""
    print("== anomaly_candidates: state field missing ==")
    result = _anomaly_candidates({})
    return _assert(result == [], f"missing field → [] (actual {result})")


def test_anomaly_candidates_window_mean_format():
    """history がある → 1 candidate、name/score/kind 正しい format。

    GAC1 識別: window mean (直近 ANOMALY_HISTORY_WINDOW) 使用、1 entry のみ参照
    の誤実装と区別。
    """
    print("== anomaly_candidates: window mean format ==")
    # 0.0 を 5 個 + 1.0 を ANOMALY_HISTORY_WINDOW 個 (window 内 mean=1.0)
    history = [0.0] * 5 + [1.0] * ANOMALY_HISTORY_WINDOW
    state = {"jepa_prediction_error_history": history}
    result = _anomaly_candidates(state)
    expected_score = _sigmoid(1.0)
    miss_first_only = _sigmoid(history[0])  # 1 entry のみ誤実装
    return all([
        _assert(len(result) == 1, f"1 candidate (actual {len(result)})"),
        _assert(result[0]["name"] == "anomaly:recent_history",
                f"name='anomaly:recent_history' (actual {result[0].get('name')})"),
        _assert(result[0]["kind"] == "graph", f"kind='graph'"),
        _assert(abs(result[0]["score"] - expected_score) < 1e-9,
                f"score=sigmoid(window_mean=1.0)≈{expected_score:.6f} "
                f"(actual {result[0]['score']:.6f})"),
        _assert(abs(result[0]["score"] - miss_first_only) > 0.01,
                "1 entry のみ参照誤実装と識別 (GAC1)"),
    ])


# ============================================================
# (b) structural_tension_candidates: GAC2 識別 (inter_ratio 低 = 緊張高い)
# ============================================================

def test_structural_tension_no_phase6_metrics():
    """phase6_metrics 未設定 → 空 list (reflect 未走行)。"""
    print("== structural_tension: no phase6_metrics ==")
    result = _structural_tension_candidates({})
    return _assert(result == [], f"no phase6_metrics → [] (actual {result})")


def test_structural_tension_zero_inter_ratio():
    """inter_ratio = 0 (cluster なし or 全孤立) → 空 list。"""
    print("== structural_tension: inter_ratio=0 ==")
    state = {"phase6_metrics": {"cluster_inter_ratio": 0.0}}
    result = _structural_tension_candidates(state)
    return _assert(result == [], f"inter_ratio=0 → [] (actual {result})")


def test_structural_tension_low_inter_ratio_high_tension():
    """GAC2 識別: inter_ratio 低 (cluster 間 link 少) → tension 高、score 大。

    inter_ratio=0.1 → tension=0.9 → sigmoid(0.9)≈0.711。
    反転誤実装 (inter_ratio 直 sigmoid) なら sigmoid(0.1)≈0.525、識別可能。
    """
    print("== structural_tension: low inter_ratio = high tension (GAC2 識別) ==")
    state = {"phase6_metrics": {"cluster_inter_ratio": 0.1}}
    result = _structural_tension_candidates(state)
    expected = _sigmoid(0.9)
    miss_inverted = _sigmoid(0.1)
    return all([
        _assert(len(result) == 1, f"1 candidate (actual {len(result)})"),
        _assert(result[0]["name"] == "structural_tension:cluster_inter_ratio",
                "name 正しい"),
        _assert(result[0]["kind"] == "graph", "kind='graph'"),
        _assert(abs(result[0]["score"] - expected) < 1e-9,
                f"score=sigmoid(1-0.1)≈{expected:.6f} (actual {result[0]['score']:.6f})"),
        _assert(abs(result[0]["score"] - miss_inverted) > 0.01,
                "反転誤実装 sigmoid(0.1) と識別 (GAC2)"),
    ])


def test_structural_tension_high_inter_ratio_low_tension():
    """inter_ratio 高 (cluster 間 link 多) → tension 低。"""
    print("== structural_tension: high inter_ratio = low tension ==")
    state = {"phase6_metrics": {"cluster_inter_ratio": 0.9}}
    result = _structural_tension_candidates(state)
    expected = _sigmoid(0.1)  # tension=0.1
    return all([
        _assert(len(result) == 1, "1 candidate"),
        _assert(abs(result[0]["score"] - expected) < 1e-9,
                f"score=sigmoid(0.1)≈{expected:.6f} (actual {result[0]['score']:.6f})"),
    ])


# ============================================================
# (c) relational_salience_candidates: GAC3 識別 (相対比じゃなく絶対 degree 誤実装)
# ============================================================

def test_relational_salience_empty_graph():
    """空 graph → 空 list。"""
    print("== relational_salience: empty graph ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        _write_links(tmp, [])
        result = _relational_salience_candidates({})
        return _assert(result == [], f"empty → [] (actual {result})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_relational_salience_uniform_degree_no_candidate():
    """全 node の degree が均等 → relative=1.0 で hub なし → 空 list。

    triangle (A↔B↔C↔A): 全 node degree=2、avg=2、relative=1.0 = candidate なし。
    """
    print("== relational_salience: uniform degree (triangle) ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "A", "to_id": "B", "link_type": "supporting"},
            {"from_id": "B", "to_id": "C", "link_type": "supporting"},
            {"from_id": "C", "to_id": "A", "link_type": "supporting"},
        ]
        _write_links(tmp, links)
        result = _relational_salience_candidates({})
        return _assert(result == [],
                       f"uniform degree → no candidate (actual {result})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_relational_salience_hub_spoke():
    """GAC3 識別: hub-spoke で hub のみ candidate (相対比 > 1.0)。

    hub から 4 leaf: hub degree=4、leaf degree=1、avg=(4+1*4)/5=1.6
    hub relative = 4/1.6 = 2.5 (> 1) → candidate
    leaf relative = 1/1.6 = 0.625 (<= 1) → 除外

    GAC3 (絶対 degree で判定) 誤実装なら閾値次第で leaf も含むかも、
    本 fixture は **hub のみ candidate = 1 件** で識別可。
    """
    print("== relational_salience: hub-spoke (GAC3 識別) ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "hub", "to_id": f"leaf{i}", "link_type": "supporting"}
            for i in range(4)
        ]
        _write_links(tmp, links)
        result = _relational_salience_candidates({})
        names = [c["name"] for c in result]
        return all([
            _assert(len(result) == 1,
                    f"hub のみ candidate → 1 件 (actual {len(result)})"),
            _assert(result[0]["name"] == "relational_salience:hub",
                    f"name='relational_salience:hub' (actual {result[0].get('name')})"),
            _assert(result[0]["kind"] == "graph", "kind='graph'"),
            _assert(0.0 <= result[0]["score"] <= 1.0,
                    f"score 範囲 [0,1] (actual {result[0]['score']:.6f})"),
            _assert(not any("leaf" in n for n in names),
                    "leaf は candidate に含まれない (相対比 < 1)"),
        ])
    finally:
        _restore_tmp_memory(tmp, orig_ml)


# ============================================================
# (d) predictive_frontier_candidates: GAC4 識別 (isolated 含めない)
# ============================================================

def test_predictive_frontier_empty_graph():
    """空 graph → 空 list。"""
    print("== predictive_frontier: empty graph ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        _write_links(tmp, [])
        result = _predictive_frontier_candidates({})
        return _assert(result == [], f"empty → [] (actual {result})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_predictive_frontier_triangle_no_leaf():
    """triangle → leaf 0 → 空 list (commit 1 _frontier_count と整合)。"""
    print("== predictive_frontier: triangle (no leaf) ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "A", "to_id": "B", "link_type": "supporting"},
            {"from_id": "B", "to_id": "C", "link_type": "supporting"},
            {"from_id": "C", "to_id": "A", "link_type": "supporting"},
        ]
        _write_links(tmp, links)
        result = _predictive_frontier_candidates({})
        return _assert(result == [], f"triangle → [] (actual {result})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_predictive_frontier_leaf_chain_with_strength():
    """A→B→C 連鎖 + strength で leaf candidate に score 反映。

    A,C が leaf (degree=1)、B は中間 (degree=2)。candidate 2 件、
    score = sigmoid(strength)。
    """
    print("== predictive_frontier: leaf chain with strength ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "A", "to_id": "B", "link_type": "supporting", "strength": 0.5},
            {"from_id": "B", "to_id": "C", "link_type": "supporting", "strength": 0.3},
        ]
        _write_links(tmp, links)
        result = _predictive_frontier_candidates({})
        names = sorted([c["name"] for c in result])
        score_a = next(c["score"] for c in result if c["name"] == "predictive_frontier:A")
        score_c = next(c["score"] for c in result if c["name"] == "predictive_frontier:C")
        return all([
            _assert(len(result) == 2, f"A,C leaf → 2 件 (actual {len(result)})"),
            _assert(names == ["predictive_frontier:A", "predictive_frontier:C"],
                    "name 集合"),
            _assert(all(c["kind"] == "graph" for c in result), "全 kind='graph'"),
            _assert(abs(score_a - _sigmoid(0.5)) < 1e-9,
                    f"A score=sigmoid(0.5)≈{_sigmoid(0.5):.6f} (actual {score_a:.6f})"),
            _assert(abs(score_c - _sigmoid(0.3)) < 1e-9,
                    f"C score=sigmoid(0.3)≈{_sigmoid(0.3):.6f} (actual {score_c:.6f})"),
        ])
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_predictive_frontier_excludes_isolated():
    """GAC4 識別: isolated (degree=0、link 経路に登場しない node) は除外。

    A→B のみ link、C は load_all_memories 経由で見えるが link なし。
    candidate は A,B のみ (isolated C は frontier じゃない、commit 1 確定)。
    """
    print("== predictive_frontier: isolated 除外 (GAC4 識別) ==")
    tmp, orig_ml = _setup_tmp_memory()
    orig_lam = cm.load_all_memories
    cm.load_all_memories = lambda: [{"id": "A"}, {"id": "B"}, {"id": "C"}]
    try:
        links = [{"from_id": "A", "to_id": "B", "link_type": "supporting", "strength": 0.4}]
        _write_links(tmp, links)
        result = _predictive_frontier_candidates({})
        names = sorted([c["name"] for c in result])
        return all([
            _assert(len(result) == 2,
                    f"A,B leaf、C isolated 除外 → 2 (actual {len(result)})"),
            _assert(names == ["predictive_frontier:A", "predictive_frontier:B"],
                    "isolated C 含まれない"),
        ])
    finally:
        cm.load_all_memories = orig_lam
        _restore_tmp_memory(tmp, orig_ml)


# ============================================================
# 統合 entry: GAC5/GAC6/GAC7 識別 (w_graph 適用 / 空 / format)
# ============================================================

def test_compute_graph_axis_w_graph_zero_empty():
    """GAC6 識別: w_graph<=0 (空 graph) → 空 list。

    PLAN §3-3「空で graph 軸寄与ゼロ」literal 整合の構造的ガード。
    """
    print("== compute_graph_axis: w_graph=0 → empty (GAC6 識別) ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        # graph データはあっても w_graph=0 なら空 return
        links = [
            {"from_id": "hub", "to_id": f"leaf{i}", "link_type": "supporting", "strength": 0.5}
            for i in range(4)
        ]
        _write_links(tmp, links)
        state = {"jepa_prediction_error_history": [1.0] * ANOMALY_HISTORY_WINDOW}
        result = compute_graph_axis_candidates(state, w_graph=0.0)
        return _assert(result == [], f"w_graph=0 → [] (actual len {len(result)})")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_compute_graph_axis_w_graph_full_no_scaling():
    """w_graph=1.0 → raw score そのまま (multiplicative 1 = 不変)。"""
    print("== compute_graph_axis: w_graph=1.0 → raw score 不変 ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "A", "to_id": "B", "link_type": "supporting", "strength": 0.5},
        ]
        _write_links(tmp, links)
        result = compute_graph_axis_candidates({}, w_graph=1.0)
        # leaf A,B に対して score = sigmoid(0.5) * 1.0
        expected = _sigmoid(0.5)
        scores = [c["score"] for c in result if c["name"].startswith("predictive_frontier")]
        return all([
            _assert(len(scores) == 2, f"frontier 2 candidate (actual {len(scores)})"),
            _assert(all(abs(s - expected) < 1e-9 for s in scores),
                    f"全 score=sigmoid(0.5)≈{expected:.6f} (w_graph=1 不変)"),
        ])
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_compute_graph_axis_w_graph_half_scaling():
    """GAC5 識別: w_graph=0.5 → score 半分 (multiplicative 適用)。

    raw score sigmoid(0.5)≈0.622、w_graph=0.5 適用後 ≈0.311。
    GAC5 (w_graph 未適用) 誤実装と識別。
    """
    print("== compute_graph_axis: w_graph=0.5 → score 半分 (GAC5 識別) ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "A", "to_id": "B", "link_type": "supporting", "strength": 0.5},
        ]
        _write_links(tmp, links)
        result = compute_graph_axis_candidates({}, w_graph=0.5)
        raw_score = _sigmoid(0.5)
        expected = raw_score * 0.5
        scores = [c["score"] for c in result if c["name"].startswith("predictive_frontier")]
        return all([
            _assert(len(scores) == 2, "frontier 2 candidate"),
            _assert(all(abs(s - expected) < 1e-9 for s in scores),
                    f"全 score=raw*0.5={expected:.6f} (actual first {scores[0]:.6f})"),
            _assert(all(abs(s - raw_score) > 0.01 for s in scores),
                    f"未適用 raw score={raw_score:.6f} と識別 (GAC5)"),
        ])
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_compute_graph_axis_format_consistency():
    """GAC7 識別: 全 candidate に "name" / "score" / "kind"='graph' field 必須。"""
    print("== compute_graph_axis: format consistency (GAC7 識別) ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "hub", "to_id": f"leaf{i}", "link_type": "supporting", "strength": 0.5}
            for i in range(4)
        ]
        _write_links(tmp, links)
        state = {
            "jepa_prediction_error_history": [1.0] * ANOMALY_HISTORY_WINDOW,
            "phase6_metrics": {"cluster_inter_ratio": 0.3},
        }
        result = compute_graph_axis_candidates(state, w_graph=1.0)
        return all([
            _assert(len(result) > 0, f"4 種混在 fixture で candidate あり"),
            _assert(all("name" in c and isinstance(c["name"], str) for c in result),
                    "全 entry に name field (str)"),
            _assert(all("score" in c and isinstance(c["score"], float) for c in result),
                    "全 entry に score field (float)"),
            _assert(all(c.get("kind") == "graph" for c in result),
                    "全 entry に kind='graph'"),
            _assert(all(0.0 <= c["score"] <= 1.0 for c in result),
                    "全 score [0,1] 範囲 (sigmoid * w_graph)"),
        ])
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_compute_graph_axis_4_kinds_mix():
    """4 種候補が同時発火する fixture で list 構造検証 (識別力)。

    fixture: hub-spoke (relational + frontier) + history (anomaly) +
             phase6_metrics (structural_tension) で 4 種すべて発火。
    """
    print("== compute_graph_axis: 4 種混在発火 ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        links = [
            {"from_id": "hub", "to_id": f"leaf{i}", "link_type": "supporting", "strength": 0.5}
            for i in range(4)
        ]
        _write_links(tmp, links)
        state = {
            "jepa_prediction_error_history": [0.7] * ANOMALY_HISTORY_WINDOW,
            "phase6_metrics": {"cluster_inter_ratio": 0.2},
        }
        result = compute_graph_axis_candidates(state, w_graph=1.0)
        kinds_present = set()
        for c in result:
            name = c["name"]
            if name.startswith("anomaly:"):
                kinds_present.add("anomaly")
            elif name.startswith("structural_tension:"):
                kinds_present.add("structural_tension")
            elif name.startswith("relational_salience:"):
                kinds_present.add("relational_salience")
            elif name.startswith("predictive_frontier:"):
                kinds_present.add("predictive_frontier")
        return all([
            _assert("anomaly" in kinds_present, "anomaly 発火"),
            _assert("structural_tension" in kinds_present, "structural_tension 発火"),
            _assert("relational_salience" in kinds_present, "relational_salience 発火"),
            _assert("predictive_frontier" in kinds_present, "predictive_frontier 発火"),
        ])
    finally:
        _restore_tmp_memory(tmp, orig_ml)


def test_compute_graph_axis_empty_fixture():
    """空 fixture (link なし、history なし、phase6_metrics なし) → 空 list。

    PLAN §3-3 整合: graph 育ってない状態で graph 軸寄与完全ゼロ。
    """
    print("== compute_graph_axis: 完全空 fixture ==")
    tmp, orig_ml = _setup_tmp_memory()
    try:
        _write_links(tmp, [])
        result = compute_graph_axis_candidates({}, w_graph=1.0)
        return _assert(result == [],
                       f"完全空 → [] (actual {len(result)} items)")
    finally:
        _restore_tmp_memory(tmp, orig_ml)


# ============================================================
# main
# ============================================================

def main():
    tests = [
        test_anomaly_candidates_empty_history,
        test_anomaly_candidates_state_missing,
        test_anomaly_candidates_window_mean_format,
        test_structural_tension_no_phase6_metrics,
        test_structural_tension_zero_inter_ratio,
        test_structural_tension_low_inter_ratio_high_tension,
        test_structural_tension_high_inter_ratio_low_tension,
        test_relational_salience_empty_graph,
        test_relational_salience_uniform_degree_no_candidate,
        test_relational_salience_hub_spoke,
        test_predictive_frontier_empty_graph,
        test_predictive_frontier_triangle_no_leaf,
        test_predictive_frontier_leaf_chain_with_strength,
        test_predictive_frontier_excludes_isolated,
        test_compute_graph_axis_w_graph_zero_empty,
        test_compute_graph_axis_w_graph_full_no_scaling,
        test_compute_graph_axis_w_graph_half_scaling,
        test_compute_graph_axis_format_consistency,
        test_compute_graph_axis_4_kinds_mix,
        test_compute_graph_axis_empty_fixture,
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
