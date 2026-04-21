"""MediumPredictor + parse_candidates 段階9 テスト。

STAGE9 §4-2 / §7-3 の要件:
  - make_prediction: predicted_e2 追加 (デフォルト 50、clamp [0, 100])
  - LightPredictor: category に応じた暫定 predicted_e2 を返す
  - MediumPredictor: candidate['prediction'] の predicted_e2 を読む
    (source=="medium" の時のみ)、それ以外は Light fallback
  - parse_candidates: 行末尾 "/ predicted_e2: XX" を抽出、candidate に格納
    抽出失敗行は prediction フィールドなし (Light fallback 経路)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_medium_predictor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.predictor import (
    LightPredictor,
    MediumPredictor,
    make_prediction,
    CATEGORY_POSITIVE_REPLY,
    CATEGORY_ERROR,
    CATEGORY_NO_RESPONSE,
    CATEGORY_OTHER,
)
from core.parser import parse_candidates


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# make_prediction: predicted_e2 追加
# ============================================================

def test_make_prediction_has_predicted_e2_field():
    print("== make_prediction: predicted_e2 field 追加、デフォルト 50 ==")
    r = make_prediction()
    return all([
        _assert("predicted_e2" in r, "predicted_e2 field 存在"),
        _assert(r["predicted_e2"] == 50, f"default=50 (actual: {r['predicted_e2']})"),
        _assert(isinstance(r["predicted_e2"], int), "int 型"),
    ])


def test_make_prediction_clamps_predicted_e2():
    print("== make_prediction: predicted_e2 clamp [0, 100] ==")
    r_high = make_prediction(predicted_e2=150)
    r_low = make_prediction(predicted_e2=-30)
    r_str = make_prediction(predicted_e2="not a number")  # type: ignore
    return all([
        _assert(r_high["predicted_e2"] == 100, "150 → clamp 100"),
        _assert(r_low["predicted_e2"] == 0, "-30 → clamp 0"),
        _assert(r_str["predicted_e2"] == 50, "不正値 → default 50"),
    ])


def test_make_prediction_backward_compat():
    print("== make_prediction: 既存 field (category/confidence/detail) 後方互換 ==")
    r = make_prediction(CATEGORY_POSITIVE_REPLY, 0.8, "test", 85)
    return all([
        _assert(r["category"] == CATEGORY_POSITIVE_REPLY, "category 維持"),
        _assert(abs(r["confidence"] - 0.8) < 1e-9, "confidence 維持"),
        _assert(r["detail"] == "test", "detail 維持"),
        _assert(r["predicted_e2"] == 85, "predicted_e2 設定"),
    ])


# ============================================================
# LightPredictor: category → predicted_e2 マップ
# ============================================================

def test_light_predictor_predicted_e2_by_category():
    print("== LightPredictor: category ごとに暫定 predicted_e2 を返す ==")
    p = LightPredictor()
    r_pos = p.predict({"expected": "応答があるはず"}, {})
    r_err = p.predict({"expected": "エラーになる"}, {})
    r_no = p.predict({"expected": "応答なしの可能性"}, {})
    r_other = p.predict({"expected": "なにかする"}, {})
    return all([
        _assert(r_pos["predicted_e2"] == 70, f"positive_reply → 70 (actual: {r_pos['predicted_e2']})"),
        _assert(r_err["predicted_e2"] == 20, f"error → 20 (actual: {r_err['predicted_e2']})"),
        _assert(r_no["predicted_e2"] == 30, f"no_response → 30 (actual: {r_no['predicted_e2']})"),
        _assert(r_other["predicted_e2"] == 50, f"other → 50 (actual: {r_other['predicted_e2']})"),
    ])


# ============================================================
# MediumPredictor: candidate.prediction から predicted_e2 を読む
# ============================================================

def test_medium_predictor_reads_prediction_from_candidate():
    print("== MediumPredictor: candidate.prediction から predicted_e2 を読む ==")
    p = MediumPredictor()
    cand = {
        "tool": "output_display",
        "reason": "応答",
        "prediction": {
            "predicted_e2": 72,
            "confidence": 0.7,
            "source": "medium",
        },
    }
    r = p.predict(cand, {})
    # 段階10 Step 3 (案 イ revert): MediumPredictor は純粋な素通し役、
    # confidence は prediction から直接返す。selection 接続は controller に一本化。
    return all([
        _assert(r["predicted_e2"] == 72, f"72 (actual: {r['predicted_e2']})"),
        _assert(r["detail"] == "medium", "detail=medium (LLM① 併合由来)"),
        _assert(abs(r["confidence"] - 0.7) < 1e-9, "confidence=0.7 (素通し)"),
    ])


def test_medium_predictor_falls_back_without_prediction():
    print("== MediumPredictor: prediction なしで Light fallback ==")
    p = MediumPredictor()
    cand = {"tool": "output_display", "expected": "応答があるはず"}
    r = p.predict(cand, {})
    # Light fallback → keyword "応答" → positive_reply (70)
    return all([
        _assert(r["predicted_e2"] == 70, "Light fallback → 70"),
        _assert(r["detail"] == "light", "detail=light (fallback 由来)"),
        _assert(r["category"] == CATEGORY_POSITIVE_REPLY, "Light の category 判定"),
    ])


def test_medium_predictor_falls_back_with_non_medium_source():
    print("== MediumPredictor: source!=medium で Light fallback (安全網) ==")
    p = MediumPredictor()
    cand = {
        "tool": "output_display",
        "expected": "エラー",
        "prediction": {
            "predicted_e2": 99,
            "confidence": 0.9,
            "source": "other",  # medium ではない
        },
    }
    r = p.predict(cand, {})
    return all([
        _assert(r["predicted_e2"] == 20, "Light fallback → error=20 (prediction 無視)"),
        _assert(r["detail"] == "light", "detail=light"),
    ])


def test_medium_predictor_boundary_values():
    print("== MediumPredictor: predicted_e2 境界値 0 / 100 ==")
    p = MediumPredictor()
    cand_0 = {"tool": "x", "prediction": {"predicted_e2": 0, "confidence": 0.7, "source": "medium"}}
    cand_100 = {"tool": "x", "prediction": {"predicted_e2": 100, "confidence": 0.7, "source": "medium"}}
    r_0 = p.predict(cand_0, {})
    r_100 = p.predict(cand_100, {})
    return all([
        _assert(r_0["predicted_e2"] == 0, "境界値 0"),
        _assert(r_100["predicted_e2"] == 100, "境界値 100"),
    ])


# ============================================================
# parse_candidates: predicted_e2 抽出
# ============================================================

def test_parse_candidates_extracts_predicted_e2():
    print("== parse_candidates: 「/ predicted_e2: XX」抽出 → prediction 格納 ==")
    text = """1. [応答を送る] → output_display / predicted_e2: 72
2. [memory 整理] → memory_store / predicted_e2: 55
3. [内省] → reflect / predicted_e2: 85"""
    cands = parse_candidates(text, {"output_display", "memory_store", "reflect"})
    if len(cands) != 3:
        return _assert(False, f"3 candidate 抽出 (actual: {len(cands)})")
    return all([
        _assert(cands[0]["tool"] == "output_display", "cand 0 tool"),
        _assert(cands[0].get("prediction", {}).get("predicted_e2") == 72, "cand 0 pe2=72"),
        _assert(cands[0].get("prediction", {}).get("source") == "medium", "cand 0 source=medium"),
        _assert(cands[1]["tool"] == "memory_store", "cand 1 tool"),
        _assert(cands[1].get("prediction", {}).get("predicted_e2") == 55, "cand 1 pe2=55"),
        _assert(cands[2]["tool"] == "reflect", "cand 2 tool"),
        _assert(cands[2].get("prediction", {}).get("predicted_e2") == 85, "cand 2 pe2=85"),
    ])


def test_parse_candidates_no_predicted_e2_has_no_prediction():
    print("== parse_candidates: predicted_e2 なし行 → prediction フィールドなし (Light fallback 経路) ==")
    text = """1. [応答] → output_display
2. [内省] → reflect"""
    cands = parse_candidates(text, {"output_display", "reflect"})
    if len(cands) != 2:
        return _assert(False, f"2 candidate (actual: {len(cands)})")
    return all([
        _assert("prediction" not in cands[0], "cand 0 prediction フィールドなし"),
        _assert("prediction" not in cands[1], "cand 1 prediction フィールドなし"),
    ])


def test_parse_candidates_tool_extraction_unharmed_by_pe2_suffix():
    print("== parse_candidates: 「/ predicted_e2: XX」が tool 名抽出を汚染しない ==")
    # 既存のロジックは ASCII 英数字+_ 以外を削るので、ノイズ除去が機能しないと
    # tool 名が「output_displaypredicted_e2_72」になってしまう (回帰リスク)
    text = "1. [応答送信] → output_display / predicted_e2: 72"
    cands = parse_candidates(text, {"output_display"})
    if not cands:
        return _assert(False, "candidate 抽出失敗")
    return all([
        _assert(cands[0]["tool"] == "output_display",
                f"tool='output_display' (actual: '{cands[0]['tool']}')"),
        _assert(cands[0].get("prediction", {}).get("predicted_e2") == 72, "pe2=72"),
    ])


def test_parse_candidates_predicted_e2_clamp_in_parser():
    print("== parse_candidates: LLM が範囲外の数字を出しても clamp される ==")
    text = """1. [A] → tool_a / predicted_e2: 150
2. [B] → tool_b / predicted_e2: 0"""
    cands = parse_candidates(text, {"tool_a", "tool_b"})
    if len(cands) != 2:
        return _assert(False, f"2 candidate (actual: {len(cands)})")
    return all([
        _assert(cands[0]["prediction"]["predicted_e2"] == 100, "150 → clamp 100"),
        _assert(cands[1]["prediction"]["predicted_e2"] == 0, "0 → 0 (境界値)"),
    ])


def test_parse_candidates_mixed_with_chain_tools():
    print("== parse_candidates: chain (tool+tool) 形式 + predicted_e2 混在 ==")
    text = "1. [調査] → read_file+update_self / predicted_e2: 65"
    cands = parse_candidates(text, {"read_file", "update_self"})
    if not cands:
        return _assert(False, "candidate 抽出失敗")
    return all([
        _assert(cands[0]["tools"] == ["read_file", "update_self"], "chain tools 抽出"),
        _assert(cands[0].get("prediction", {}).get("predicted_e2") == 65, "pe2=65"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("make_prediction: predicted_e2 field", test_make_prediction_has_predicted_e2_field),
        ("make_prediction: clamp [0,100]", test_make_prediction_clamps_predicted_e2),
        ("make_prediction: 後方互換", test_make_prediction_backward_compat),
        ("Light: category → predicted_e2", test_light_predictor_predicted_e2_by_category),
        ("Medium: prediction から読む", test_medium_predictor_reads_prediction_from_candidate),
        ("Medium: prediction なしで Light fallback", test_medium_predictor_falls_back_without_prediction),
        ("Medium: source!=medium で fallback", test_medium_predictor_falls_back_with_non_medium_source),
        ("Medium: 境界値 0 / 100", test_medium_predictor_boundary_values),
        ("parse: predicted_e2 抽出", test_parse_candidates_extracts_predicted_e2),
        ("parse: pe2 なし行 → prediction なし", test_parse_candidates_no_predicted_e2_has_no_prediction),
        ("parse: tool 名抽出の純度保持", test_parse_candidates_tool_extraction_unharmed_by_pe2_suffix),
        ("parse: 範囲外 → clamp", test_parse_candidates_predicted_e2_clamp_in_parser),
        ("parse: chain + predicted_e2", test_parse_candidates_mixed_with_chain_tools),
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
