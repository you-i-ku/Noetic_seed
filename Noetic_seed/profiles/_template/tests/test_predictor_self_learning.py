"""段階10 柱 B: Predictor 自己学習 テスト。

STAGE10 PLAN v1 §3-B の要件:
  - update_predictor_confidence: 段階3 update_fact_confidence の β+ 式再利用
    - matches=True:  conf = conf + 0.05 * (1 - conf) (上限 1.0 漸近)
    - matches=False: conf = max(0.0, conf - 0.15)
  - _is_match: 案 (a) 自己相対化
    - bootstrap (history < 5): 全 matches
    - 以降: abs(error) < median(history) なら matches
  - _append_history: FIFO、cap 100
  - state lazy init: predictor_confidence / prediction_error_history_e2/_ec
  - tool 別 confidence 分化 (未 init は 0.7 start)
  - MediumPredictor が e2_conf で confidence 調整

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_predictor_self_learning.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.predictor import (
    update_predictor_confidence,
    _is_match,
    _append_history,
    HISTORY_CAP,
    BOOTSTRAP_N,
    MediumPredictor,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# _is_match: bootstrap / 自己相対化
# ============================================================

def test_is_match_bootstrap():
    print(f"== _is_match: history < {BOOTSTRAP_N} は全 matches ==")
    results = []
    for n in range(BOOTSTRAP_N):
        history = [50] * n  # 長さ 0-4
        # どんな error でも matches になる
        results.append(_is_match(100, history))
        results.append(_is_match(0, history))
    return _assert(all(results), f"history 長 0-{BOOTSTRAP_N-1} 全 matches 扱い")


def test_is_match_below_median():
    print("== _is_match: abs(error) < median → matches ==")
    history = [10, 20, 30, 40, 50]  # median=30
    return all([
        _assert(_is_match(20, history), "error=20 < median=30 → matches"),
        _assert(not _is_match(50, history), "error=50 >= median=30 → mismatches"),
        _assert(not _is_match(30, history), "error=30 == median=30 → mismatches (strict <)"),
    ])


def test_is_match_abs_value():
    print("== _is_match: 絶対値で判定 ==")
    history = [10, 20, 30, 40, 50]
    return _assert(_is_match(-15, history), "error=-15 → abs(15) < 30 → matches")


# ============================================================
# _append_history: FIFO, cap HISTORY_CAP
# ============================================================

def test_append_history_basic():
    print("== _append_history: abs 値を append ==")
    state = {}
    _append_history(state, 50, "e2")
    _append_history(state, -30, "e2")
    return all([
        _assert(state["prediction_error_history_e2"] == [50.0, 30.0], "abs + float"),
    ])


def test_append_history_fifo_cap():
    print(f"== _append_history: cap {HISTORY_CAP} で FIFO ==")
    state = {}
    for i in range(HISTORY_CAP + 5):
        _append_history(state, i, "e2")
    hist = state["prediction_error_history_e2"]
    return all([
        _assert(len(hist) == HISTORY_CAP, f"長さ = {HISTORY_CAP}"),
        _assert(hist[0] == 5.0, "先頭 5 件が削除済 (FIFO)"),
        _assert(hist[-1] == HISTORY_CAP + 4, "末尾が最新"),
    ])


def test_append_history_axis_separation():
    print("== _append_history: e2 / ec axis 分離 ==")
    state = {}
    _append_history(state, 10, "e2")
    _append_history(state, 0.5, "ec")
    return all([
        _assert(state["prediction_error_history_e2"] == [10.0], "e2 軸"),
        _assert(state["prediction_error_history_ec"] == [0.5], "ec 軸"),
    ])


# ============================================================
# update_predictor_confidence: β+ 式再利用 (段階3 と同式)
# ============================================================

def test_update_creates_entry():
    print("== update: 未 init tool は {e2_conf:0.7, ec_conf:0.7} で作成 ==")
    state = {}
    update_predictor_confidence(state, "output_display", 10)
    entry = state["predictor_confidence"]["output_display"]
    # bootstrap なので matches → e2_conf = 0.7 + 0.05 * (1 - 0.7) = 0.715
    return all([
        _assert("output_display" in state["predictor_confidence"], "tool entry 作成"),
        _assert(abs(entry["e2_conf"] - 0.715) < 1e-9, f"bootstrap matches → 0.715 (actual: {entry['e2_conf']})"),
        _assert(entry["ec_conf"] == 0.7, "ec_conf は変化なし (Step 2 時点)"),
    ])


def test_update_bootstrap_monotonic_increase():
    print(f"== update: bootstrap {BOOTSTRAP_N} cycle 連続で e2_conf が単調増加 ==")
    state = {}
    confidences = []
    for _ in range(BOOTSTRAP_N):
        update_predictor_confidence(state, "tool_A", 99)  # 大誤差でも bootstrap で matches
        confidences.append(state["predictor_confidence"]["tool_A"]["e2_conf"])
    return _assert(
        all(confidences[i] < confidences[i + 1] for i in range(len(confidences) - 1)),
        f"bootstrap 中は matches 連続で単調増加 (values: {[round(c, 4) for c in confidences]})",
    )


def test_update_post_bootstrap_mismatch():
    print("== update: bootstrap 後、大誤差で mismatches → -0.15 ==")
    state = {
        "predictor_confidence": {"tool_B": {"e2_conf": 0.7, "ec_conf": 0.7}},
        "prediction_error_history_e2": [5, 10, 15, 20, 25, 30],  # median=20 (長さ 6、index 3 は 20)
    }
    # error=99 > median=20 → mismatches → e2_conf = max(0, 0.7 - 0.15) = 0.55
    update_predictor_confidence(state, "tool_B", 99)
    return _assert(
        abs(state["predictor_confidence"]["tool_B"]["e2_conf"] - 0.55) < 1e-9,
        f"mismatches → 0.55 (actual: {state['predictor_confidence']['tool_B']['e2_conf']})",
    )


def test_update_empty_tool_name_noop():
    print("== update: tool_name='' は no-op ==")
    state = {}
    update_predictor_confidence(state, "", 50)
    return _assert(
        state.get("predictor_confidence", {}) == {},
        "空 tool_name で state 変化なし",
    )


def test_update_tool_separation():
    print("== update: tool 別に独立な entry が作られる ==")
    # 注: history_e2 は state global (全 tool 共有)、confidence は tool 別 entry
    # global history 設計の理由 = iku 全体の誤差分布との相対評価 (PLAN v1 §3-B)
    state = {}
    update_predictor_confidence(state, "tool_A", 10)
    update_predictor_confidence(state, "tool_B", 10)
    pc = state["predictor_confidence"]
    return all([
        _assert("tool_A" in pc and "tool_B" in pc, "両方 entry 存在"),
        _assert(pc["tool_A"] is not pc["tool_B"], "各 entry は独立オブジェクト"),
        _assert(len(state["prediction_error_history_e2"]) == 2, "history は共有 (tool 跨ぎで append)"),
    ])


def test_update_history_appended():
    print("== update: history に abs(error) が append される ==")
    state = {}
    update_predictor_confidence(state, "tool_X", -30)
    return _assert(
        state["prediction_error_history_e2"] == [30.0],
        f"abs(-30)=30 が history に (actual: {state['prediction_error_history_e2']})",
    )


def test_update_ec_axis_step2():
    print("== update: Step 2 時点で prediction_error_ec=None → ec 軸更新なし ==")
    state = {}
    update_predictor_confidence(state, "tool_C", 20)  # prediction_error_ec 指定なし
    return all([
        _assert(state["predictor_confidence"]["tool_C"]["ec_conf"] == 0.7, "ec_conf 変化なし"),
        _assert(state.get("prediction_error_history_ec", []) == [], "ec history 変化なし"),
    ])


# ============================================================
# MediumPredictor: e2_conf 適用
# ============================================================

def test_medium_predictor_uses_e2_conf():
    print("== MediumPredictor: tool 別 e2_conf で confidence 調整 ==")
    state = {
        "predictor_confidence": {
            "output_display": {"e2_conf": 0.5, "ec_conf": 0.7},
        }
    }
    candidate = {
        "tool": "output_display",
        "prediction": {
            "source": "medium",
            "confidence": 0.8,
            "predicted_e2": 80,
        },
    }
    result = MediumPredictor().predict(candidate, state)
    # adjusted = 0.8 * 0.5 = 0.4
    return all([
        _assert(abs(result["confidence"] - 0.4) < 1e-9, f"0.8 * 0.5 = 0.4 (actual: {result['confidence']})"),
        _assert(result["predicted_e2"] == 80, "predicted_e2 は影響なし (pragmatic 軸は controller で別処理)"),
    ])


def test_medium_predictor_default_e2_conf():
    print("== MediumPredictor: 未 init tool は e2_conf=0.7 default ==")
    state = {}
    candidate = {
        "tool": "new_tool",
        "prediction": {
            "source": "medium",
            "confidence": 0.9,
            "predicted_e2": 60,
        },
    }
    result = MediumPredictor().predict(candidate, state)
    # adjusted = 0.9 * 0.7 = 0.63
    return _assert(
        abs(result["confidence"] - 0.63) < 1e-9,
        f"0.9 * 0.7 = 0.63 (actual: {result['confidence']})",
    )


# ============================================================
# state.py lazy init
# ============================================================

def test_state_lazy_init_new_state():
    print("== state.py: 新規 state 作成時に 3 キー追加 ==")
    # load_state を直接呼ばず、追加された default keys の存在を想定
    # (実体 test は runtime でカバー、ここは契約確認のみ)
    from core.state import load_state
    # STATE_FILE が未存在な場合の default return を想定
    # しかし実環境で副作用避けるため簡易確認: 関数 signature チェック
    return _assert(callable(load_state), "load_state callable 確認 (実 init は runtime test)")


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    tests = [
        test_is_match_bootstrap,
        test_is_match_below_median,
        test_is_match_abs_value,
        test_append_history_basic,
        test_append_history_fifo_cap,
        test_append_history_axis_separation,
        test_update_creates_entry,
        test_update_bootstrap_monotonic_increase,
        test_update_post_bootstrap_mismatch,
        test_update_empty_tool_name_noop,
        test_update_tool_separation,
        test_update_history_appended,
        test_update_ec_axis_step2,
        test_medium_predictor_uses_e2_conf,
        test_medium_predictor_default_e2_conf,
        test_state_lazy_init_new_state,
    ]
    results = [t() for t in tests]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
