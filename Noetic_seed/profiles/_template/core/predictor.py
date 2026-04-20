"""Predictor プラグイン — 候補行動の結果を
{category, confidence, detail, predicted_e2} 形式で予測。

WORLD_MODEL.md §6 段階5 / STAGE5_IMPLEMENTATION_PLAN.md /
STAGE9_MEDIUM_PREDICTOR_AND_HOTFIX_PLAN.md §4-2 の実装。

設計指針 (PLAN §1 より継承):
- ミニマリズム: Light 本実装、Medium は段階9 で本実装、Heavy/Mode2 はスタブ
- 依存注入: controller 側で mode 指定 → get_predictor(mode) で取得
- LLM as brain: predictor は構造判定、LLM にプロンプト誘導を書かない
- 特権化しない: 予測失敗時は Light fallback、controller 側は継続

段階9 の設計 (in-context world model, NAACL 2025 の direct instance):
- MediumPredictor は LLM① propose prompt に predicted_e2 (0-100) 出力指示を
  併合し、parse_candidates が抽出した candidate['prediction'] を返すだけ。
  追加 LLM 呼出しゼロ。parse 失敗時は LightPredictor に fallback (安全網)。
- predicted_e2 ≒ Active Inference の pragmatic value of Expected Free Energy。
- controller では novelty (epistemic) と predicted_e2 (pragmatic) を同形式で
  乗算 (max(0.05, x) で下限ガード)、統合 EFE 最小化に近似する。

段階6+ への契約:
- predict(candidate, state, world_model=None) シグネチャは安定
- {category, confidence, detail, predicted_e2} フォーマット (predicted_e2 は
  段階9 追加、デフォルト 50 で後方互換維持)
"""
from typing import Optional


# ============================================================
# カテゴリ定数 (WORLD_MODEL.md §6 段階5)
# ============================================================

CATEGORY_POSITIVE_REPLY = "positive_reply"
CATEGORY_ERROR = "error"
CATEGORY_NO_RESPONSE = "no_response"
CATEGORY_OTHER = "other"

_VALID_CATEGORIES = {
    CATEGORY_POSITIVE_REPLY,
    CATEGORY_ERROR,
    CATEGORY_NO_RESPONSE,
    CATEGORY_OTHER,
}


# ============================================================
# 予測結果フォーマット
# ============================================================

def make_prediction(category: str = CATEGORY_OTHER,
                    confidence: float = 0.3,
                    detail: str = "",
                    predicted_e2: int = 50) -> dict:
    """{category, confidence, detail, predicted_e2} 形式の予測 dict を返す。

    不正な category は OTHER に fallback、confidence は [0.0, 1.0] にクランプ、
    predicted_e2 は [0, 100] にクランプ (段階9 追加、デフォルト 50=neutral)。
    """
    if category not in _VALID_CATEGORIES:
        category = CATEGORY_OTHER
    conf = max(0.0, min(1.0, float(confidence)))
    try:
        pe2 = max(0, min(100, int(predicted_e2)))
    except (TypeError, ValueError):
        pe2 = 50
    return {
        "category": category,
        "confidence": conf,
        "detail": str(detail),
        "predicted_e2": pe2,
    }


# category → 暫定 predicted_e2 マップ (LightPredictor 用、段階9)。
# pragmatic value 近似: positive は高、error/no_response は低、other は neutral。
_CATEGORY_PREDICTED_E2 = {
    CATEGORY_POSITIVE_REPLY: 70,
    CATEGORY_ERROR: 20,
    CATEGORY_NO_RESPONSE: 30,
    CATEGORY_OTHER: 50,
}


# ============================================================
# Predictor クラス群
# ============================================================

class BasePredictor:
    """Predictor 抽象クラス。

    predict() のデフォルト実装は {"other", 0.3, ""}。
    サブクラスは predict() を override する。
    """
    mode = "base"

    def predict(self, candidate: dict, state: dict,
                world_model: Optional[dict] = None) -> dict:
        return make_prediction()


class LightPredictor(BasePredictor):
    """Keyword マッチベース。追加 LLM 呼び出しなし。

    candidate の expected / intent / reason 文字列から category を推定。
    順序: no_response → error → positive_reply → other
    (「応答なし」が「応答」より先に hit するよう no_response を最優先で判定)
    """
    mode = "light"

    # 日本語 keyword は lower() 後も不変、英語は小文字化で吸収
    _NO_RESPONSE = ("応答なし", "無視", "無反応", "no response", "silent")
    _ERROR = ("エラー", "失敗", "error", "fail", "exception")
    _POSITIVE = ("応答", "返事", "reply", "success", " ok")

    def predict(self, candidate: dict, state: dict,
                world_model: Optional[dict] = None) -> dict:
        src = " ".join([
            str(candidate.get("expected", "")),
            str(candidate.get("intent", "")),
            str(candidate.get("reason", "")),
        ]).lower()
        if any(k in src for k in self._NO_RESPONSE):
            return make_prediction(CATEGORY_NO_RESPONSE, 0.5, "light",
                                   _CATEGORY_PREDICTED_E2[CATEGORY_NO_RESPONSE])
        if any(k in src for k in self._ERROR):
            return make_prediction(CATEGORY_ERROR, 0.6, "light",
                                   _CATEGORY_PREDICTED_E2[CATEGORY_ERROR])
        if any(k in src for k in self._POSITIVE):
            return make_prediction(CATEGORY_POSITIVE_REPLY, 0.6, "light",
                                   _CATEGORY_PREDICTED_E2[CATEGORY_POSITIVE_REPLY])
        return make_prediction(CATEGORY_OTHER, 0.3, "light",
                               _CATEGORY_PREDICTED_E2[CATEGORY_OTHER])


class MediumPredictor(BasePredictor):
    """LLM① プロンプト併合による予測 (段階9 本実装)。

    LLM① の propose prompt に「各候補に predicted_e2 (0-100) を付けて」と
    併合指示、parse_candidates が抽出した candidate['prediction'] を返す。
    追加 LLM 呼出しゼロ (in-context world model, NAACL 2025 の direct instance)。
    parse 失敗時 (prediction が無い / source が medium でない) は LightPredictor
    に fallback (安全網、特権化しない原則)。
    """
    mode = "medium"

    def predict(self, candidate: dict, state: dict,
                world_model: Optional[dict] = None) -> dict:
        prediction = candidate.get("prediction")
        if isinstance(prediction, dict) and prediction.get("source") == "medium":
            return make_prediction(
                category=CATEGORY_OTHER,  # category は補助情報、multiplier は predicted_e2 主
                confidence=prediction.get("confidence", 0.7),
                detail="medium",
                predicted_e2=prediction.get("predicted_e2", 50),
            )
        return LightPredictor().predict(candidate, state, world_model)


class HeavyPredictor(BasePredictor):
    """独立 LLM 呼び出しで候補ごとに詳細予測。段階6+ で実装予定。"""
    mode = "heavy"

    def predict(self, candidate: dict, state: dict,
                world_model: Optional[dict] = None) -> dict:
        return LightPredictor().predict(candidate, state, world_model)


class Mode2Predictor(BasePredictor):
    """Mode-2 反実仮想予測。将来実装。"""
    mode = "mode2"

    def predict(self, candidate: dict, state: dict,
                world_model: Optional[dict] = None) -> dict:
        return LightPredictor().predict(candidate, state, world_model)


# ============================================================
# ファクトリ
# ============================================================

_PREDICTOR_REGISTRY = {
    "light": LightPredictor,
    "medium": MediumPredictor,
    "heavy": HeavyPredictor,
    "mode2": Mode2Predictor,
}


def get_predictor(mode: str = "light") -> BasePredictor:
    """mode 文字列から Predictor インスタンスを取得。

    不明 mode は LightPredictor に fallback (特権化しない方針)。
    """
    cls = _PREDICTOR_REGISTRY.get(mode, LightPredictor)
    return cls()
