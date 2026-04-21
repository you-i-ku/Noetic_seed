"""controller × Predictor 統合乗算 (段階9 Step 2) 統合テスト。

STAGE9 §4-3 / §7-4 の要件:
  - novelty × predicted_e2 の統合乗算が正しく効く
  - predicted_e2 が低い候補は強く抑制される
  - Medium 予測が candidate.prediction に入ってる場合、それが乗算に使われる
  - 候補間の相対重み差が predicted_e2 で変動する

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_controller_predictor_integration.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.controller import (
    _predicted_outcome_multiplier,
    controller_select,
)
from core.config import WORLD_MODEL_CFG
from core.world_model import init_world_model


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# 統合乗算: predicted_e2 が controller_select を通して反映される
# ============================================================

def test_medium_prediction_propagates_through_controller():
    print("== Medium prediction (candidate.prediction) が controller の乗算に反映 ==")
    wm = init_world_model()
    state = {
        "energy": 50, "entropy": 0.65, "log": [], "pending": [],
        "world_model": wm,
    }
    ctrl = {"tool_rank": {"output_display": 60, "wait": 40}}
    # Medium 予測あり (LLM① が出したつもり): output_display=80 (高達成)
    # Light fallback: wait="待機 → 変化なし" → no_response → 30
    candidates = [
        {"tool": "output_display", "reason": "挨拶応答",
         "expected": "挨拶が返る",
         "prediction": {"predicted_e2": 80, "confidence": 0.7, "source": "medium"}},
        {"tool": "wait", "reason": "待機", "expected": "応答なしの予想"},
    ]
    controller_select(candidates, ctrl, state)
    return all([
        _assert(candidates[0].get("_predicted_e2") == 80,
                f"Medium 予測 80 が controller に伝播 (actual: {candidates[0].get('_predicted_e2')})"),
        _assert(candidates[1].get("_predicted_e2") == 30,
                f"Light fallback → no_response → 30 (actual: {candidates[1].get('_predicted_e2')})"),
        _assert(candidates[0]["predicted_outcome"].get("detail") == "medium",
                "output_display は Medium 由来 (detail=medium)"),
        _assert(candidates[1]["predicted_outcome"].get("detail") == "light",
                "wait は Light fallback (detail=light)"),
    ])


def test_low_predicted_e2_suppresses_weight():
    print("== pe2 が低い候補は multiplier 側で強く抑制される ==")
    # 段階10 柱 C: signature 拡張 (prediction, candidate, state, cfg)
    # predicted_ec 欠如時は段階9 挙動 (pe2_ratio のみ) 維持、比 1:9 相当変わらず
    # penalty ラベルは "low_outcome=..." に変更 (pe2/ec 統合指標として)
    cand_low = {"tool": "x"}
    cand_high = {"tool": "x"}
    m_low = _predicted_outcome_multiplier(
        {"predicted_e2": 10}, cand_low, {}, WORLD_MODEL_CFG)
    m_high = _predicted_outcome_multiplier(
        {"predicted_e2": 90}, cand_high, {}, WORLD_MODEL_CFG)
    return all([
        _assert(abs(m_low - 0.1) < 1e-9, f"pe2=10 → 0.1 (actual: {m_low})"),
        _assert(abs(m_high - 0.9) < 1e-9, f"pe2=90 → 0.9 (actual: {m_high})"),
        _assert(m_high / m_low > 8.0,
                f"pe2 高低の比 9 倍 (actual ratio: {m_high/m_low:.1f}x)"),
        _assert(any("low_outcome" in p for p in cand_low.get("penalties", [])),
                "低 pe2 candidate に penalty 記録 (ラベル low_outcome)"),
        _assert("penalties" not in cand_high or not cand_high["penalties"],
                "高 pe2 candidate に penalty なし"),
    ])


def test_high_pe2_candidate_has_higher_weight_share():
    """pe2 の差が controller_select の重み配分に影響することを確認。
    cumul-sum 重み計算の性質上、確率的選択になるので比だけ確認。
    """
    print("== 高 pe2 候補は低 pe2 候補より重み share が大きい ==")
    wm = init_world_model()
    state = {
        "energy": 50, "entropy": 0.65, "log": [], "pending": [],
        "world_model": wm,
    }
    ctrl = {"tool_rank": {"tool_a": 50, "tool_b": 50}}
    candidates = [
        {"tool": "tool_a", "reason": "候補 A",
         "prediction": {"predicted_e2": 80, "confidence": 0.7, "source": "medium"}},
        {"tool": "tool_b", "reason": "候補 B",
         "prediction": {"predicted_e2": 20, "confidence": 0.7, "source": "medium"}},
    ]
    # 複数回試行して選択比率を確認 (統計的)。controller_select は random.random()
    # ベースなので seed 固定せず、100 試行で A が明らかに多いことを期待。
    import random
    random.seed(42)
    count_a = 0
    for _ in range(100):
        # candidates は参照渡しで副作用あり。コピーして試行。
        cands_copy = [dict(c) for c in candidates]
        chosen = controller_select(cands_copy, ctrl, state)
        if chosen["tool"] == "tool_a":
            count_a += 1
    # 理論: A の multiplier は 0.8、B は 0.2、比 4:1 → A 選択率 ~80%
    # 実際は novelty 等の他因子あり、70% 以上を緩い閾値とする
    return _assert(count_a >= 70,
                   f"tool_a (pe2=80) を 100 試行中 {count_a} 回選択 (>= 70 期待)")


def test_missing_prediction_uses_light_fallback():
    print("== candidate.prediction なし → Light fallback で pe2 補填 ==")
    wm = init_world_model()
    state = {
        "energy": 50, "entropy": 0.65, "log": [], "pending": [],
        "world_model": wm,
    }
    ctrl = {"tool_rank": {"output_display": 60}}
    candidates = [
        {"tool": "output_display", "reason": "応答する", "expected": "応答が返る"},
        # prediction なし
    ]
    controller_select(candidates, ctrl, state)
    # Light fallback: "応答" keyword → positive_reply → 70
    return all([
        _assert(candidates[0].get("_predicted_e2") == 70,
                f"Light fallback → 70 (actual: {candidates[0].get('_predicted_e2')})"),
        _assert(candidates[0]["predicted_outcome"].get("detail") == "light",
                "detail=light (fallback 由来)"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("Medium prediction の controller 伝播", test_medium_prediction_propagates_through_controller),
        ("低 pe2 の抑制効果 (単独 multiplier)", test_low_predicted_e2_suppresses_weight),
        ("高 pe2 候補が重み share 多い (統計)", test_high_pe2_candidate_has_higher_weight_share),
        ("prediction なし → Light fallback", test_missing_prediction_uses_light_fallback),
    ]
    results = []
    for _label, fn in groups:
        print()
        ok = fn()
        results.append((_label, ok))
    print()
    print("=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for _label, ok in results:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {_label}")
    print(f"\n  {passed}/{total} groups passed")
    sys.exit(0 if passed == total else 1)
