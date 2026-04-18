"""predictor.py テスト (WM 段階5)。

成功条件:
  - BasePredictor / LightPredictor / Medium/Heavy/Mode2 Stub が所定の
    {category, confidence, detail} 形式を返す
  - LightPredictor の keyword マッチが 4 カテゴリを区別
  - 「応答なし」が「応答」より優先判定される (順序 no_response → error → positive_reply)
  - get_predictor が mode 文字列から正しいインスタンスを返し、
    不明 mode は Light に fallback

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_predictor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.predictor import (
    BasePredictor,
    LightPredictor,
    MediumPredictor,
    HeavyPredictor,
    Mode2Predictor,
    get_predictor,
    make_prediction,
    CATEGORY_POSITIVE_REPLY,
    CATEGORY_ERROR,
    CATEGORY_NO_RESPONSE,
    CATEGORY_OTHER,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _has_shape(pred: dict) -> bool:
    return (
        isinstance(pred, dict)
        and "category" in pred
        and "confidence" in pred
        and "detail" in pred
        and isinstance(pred["confidence"], float)
    )


# ============================================================
# make_prediction + Base
# ============================================================

def test_base_predictor_default_category():
    print("== BasePredictor: default predict → other / 0.3 ==")
    p = BasePredictor()
    r = p.predict({}, {}, None)
    return all([
        _assert(_has_shape(r), "{category, confidence, detail} 形式"),
        _assert(r["category"] == CATEGORY_OTHER, "category=other"),
        _assert(abs(r["confidence"] - 0.3) < 1e-9, "confidence=0.3"),
    ])


def test_make_prediction_clamps_and_validates():
    print("== make_prediction: 不正 category → other、confidence clamp ==")
    r1 = make_prediction("nonsense", 1.5, "x")
    r2 = make_prediction(CATEGORY_ERROR, -0.4, "y")
    return all([
        _assert(r1["category"] == CATEGORY_OTHER, "不明 category → other"),
        _assert(r1["confidence"] == 1.0, "1.5 → clamp 1.0"),
        _assert(r2["confidence"] == 0.0, "-0.4 → clamp 0.0"),
    ])


# ============================================================
# LightPredictor keyword マッチ
# ============================================================

def test_light_predictor_positive_reply():
    print("== Light: '応答' / 'reply' keyword → positive_reply ==")
    p = LightPredictor()
    r1 = p.predict({"expected": "応答があるはず"}, {})
    r2 = p.predict({"intent": "send reply"}, {})
    return all([
        _assert(r1["category"] == CATEGORY_POSITIVE_REPLY, "応答 → positive_reply"),
        _assert(r2["category"] == CATEGORY_POSITIVE_REPLY, "reply → positive_reply"),
        _assert(r1["detail"] == "light", "detail=light"),
    ])


def test_light_predictor_error():
    print("== Light: 'エラー' / 'fail' keyword → error ==")
    p = LightPredictor()
    r1 = p.predict({"expected": "エラーになる可能性"}, {})
    r2 = p.predict({"reason": "this may fail"}, {})
    return all([
        _assert(r1["category"] == CATEGORY_ERROR, "エラー → error"),
        _assert(r2["category"] == CATEGORY_ERROR, "fail → error"),
    ])


def test_light_predictor_no_response():
    print("== Light: '応答なし' / '無視' / 'silent' → no_response ==")
    p = LightPredictor()
    r1 = p.predict({"expected": "応答なしの可能性"}, {})
    r2 = p.predict({"intent": "無視されるかも"}, {})
    r3 = p.predict({"reason": "user may be silent"}, {})
    return all([
        _assert(r1["category"] == CATEGORY_NO_RESPONSE, "応答なし → no_response"),
        _assert(r2["category"] == CATEGORY_NO_RESPONSE, "無視 → no_response"),
        _assert(r3["category"] == CATEGORY_NO_RESPONSE, "silent → no_response"),
    ])


def test_light_predictor_no_response_priority_over_positive():
    print("== Light: 「応答なし」は「応答」より優先 (順序ガード) ==")
    p = LightPredictor()
    # 「応答なし」は substring として「応答」を含むので、順序が間違うと誤判定
    r = p.predict({"expected": "応答なしになる可能性"}, {})
    return _assert(r["category"] == CATEGORY_NO_RESPONSE,
                   f"応答なし → no_response (actual: {r['category']})")


def test_light_predictor_other():
    print("== Light: keyword なし → other / 0.3 ==")
    p = LightPredictor()
    r = p.predict({"expected": "なにかする", "intent": "行動"}, {})
    return all([
        _assert(r["category"] == CATEGORY_OTHER, "other"),
        _assert(abs(r["confidence"] - 0.3) < 1e-9, "confidence=0.3"),
    ])


# ============================================================
# Stub classes (Medium / Heavy / Mode2) は Light と同じ動作
# ============================================================

def test_stubs_fall_back_to_light():
    print("== Medium/Heavy/Mode2 stub: Light と同じ動作 ==")
    cand = {"expected": "応答があれば成功"}
    base = LightPredictor().predict(cand, {})
    return all([
        _assert(MediumPredictor().predict(cand, {})["category"] == base["category"],
                "Medium → Light と同じ category"),
        _assert(HeavyPredictor().predict(cand, {})["category"] == base["category"],
                "Heavy → Light と同じ category"),
        _assert(Mode2Predictor().predict(cand, {})["category"] == base["category"],
                "Mode2 → Light と同じ category"),
    ])


# ============================================================
# get_predictor ファクトリ
# ============================================================

def test_get_predictor_by_mode():
    print("== get_predictor: 文字列から正しいクラス取得 ==")
    return all([
        _assert(isinstance(get_predictor("light"), LightPredictor), "light"),
        _assert(isinstance(get_predictor("medium"), MediumPredictor), "medium"),
        _assert(isinstance(get_predictor("heavy"), HeavyPredictor), "heavy"),
        _assert(isinstance(get_predictor("mode2"), Mode2Predictor), "mode2"),
    ])


def test_get_predictor_unknown_mode_defaults_to_light():
    print("== get_predictor: 不明 mode → Light fallback (特権化しない) ==")
    p1 = get_predictor("nonsense")
    p2 = get_predictor("")
    return all([
        _assert(isinstance(p1, LightPredictor), "nonsense → Light"),
        _assert(isinstance(p2, LightPredictor), "'' → Light"),
        _assert(p1.mode == "light", "mode=light"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("Base: default other", test_base_predictor_default_category),
        ("make_prediction: clamp + validate", test_make_prediction_clamps_and_validates),
        ("Light: positive_reply", test_light_predictor_positive_reply),
        ("Light: error", test_light_predictor_error),
        ("Light: no_response", test_light_predictor_no_response),
        ("Light: no_response 優先 (順序ガード)", test_light_predictor_no_response_priority_over_positive),
        ("Light: other", test_light_predictor_other),
        ("Stubs: Medium/Heavy/Mode2 fallback", test_stubs_fall_back_to_light),
        ("get_predictor: by mode", test_get_predictor_by_mode),
        ("get_predictor: unknown → Light", test_get_predictor_unknown_mode_defaults_to_light),
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
