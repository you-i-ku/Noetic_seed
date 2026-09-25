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
# 段階10 柱 B: Predictor 自己学習 (β+ 式再利用)
# ============================================================
#
# STAGE10 PLAN v1 §3-B の実装。段階3 update_fact_confidence の β+ 式を
# そのまま再利用し、tool 別 {e2_conf, ec_conf} を予測誤差で自己学習する。
#
# matches 判定 = 案 (a) 自己相対化:
#   - bootstrap (history < 5): 全 matches 扱い (初期不安定期の救済)
#   - 以降: abs(error) <= median(history) なら matches (偶数件は上側の中央値)
#
# history は軸別・全 tool 共有の直近 100 件 FIFO。今回の誤差を追加する前に判定する。

HISTORY_CAP = 100
BOOTSTRAP_N = 5


def _is_match(error: float, history: list) -> bool:
    """履歴5件未満は全一致、以降は絶対誤差が履歴の上側中央値以下なら一致。

    history は今回の誤差を追加する前の絶対誤差列。
    中央値0なら誤差0だけが一致する。絶対的な正確さではなく自己相対化の判定。
    """
    if len(history) < BOOTSTRAP_N:
        return True
    sorted_h = sorted(history)
    median = sorted_h[len(sorted_h) // 2]
    return abs(float(error)) <= median


def _append_history(state: dict, error: float, axis: str = "e2") -> None:
    """history に abs(error) を FIFO append、cap HISTORY_CAP。"""
    key = f"prediction_error_history_{axis}"
    history = state.setdefault(key, [])
    history.append(abs(float(error)))
    if len(history) > HISTORY_CAP:
        del history[0]


def update_predictor_confidence(state: dict, tool_name: str,
                                 prediction_error: float,
                                 prediction_error_ec: Optional[float] = None) -> None:
    """段階3 update_fact_confidence の β+ 式を再利用して tool 別 confidence 更新。

    Args:
        state: state dict
        tool_name: 実行された tool 名 (空文字なら no-op)
        prediction_error: E2 軸の予測誤差 (abs 値、0-100 scale)
        prediction_error_ec: EC 軸の予測誤差 (abs 値、0-1 scale)。Step 3 で有効化
    """
    if not tool_name:
        return
    from core.world_model import update_fact_confidence

    pc = state.setdefault("predictor_confidence", {})
    entry = pc.setdefault(tool_name, {"e2_conf": 0.7, "ec_conf": 0.7})

    # E2 軸: 現在の history で matches 判定 → β+ 更新 → history append (順序重要)
    history_e2 = state.setdefault("prediction_error_history_e2", [])
    matches_e2 = _is_match(prediction_error, history_e2)
    fake_fact = {"confidence": entry["e2_conf"]}
    update_fact_confidence(fake_fact, matches_e2)
    entry["e2_conf"] = fake_fact["confidence"]
    _append_history(state, prediction_error, "e2")

    # EC 軸 (Step 3 で有効化、Step 2 時点では prediction_error_ec=None)
    if prediction_error_ec is not None:
        history_ec = state.setdefault("prediction_error_history_ec", [])
        matches_ec = _is_match(prediction_error_ec, history_ec)
        fake_fact = {"confidence": entry["ec_conf"]}
        update_fact_confidence(fake_fact, matches_ec)
        entry["ec_conf"] = fake_fact["confidence"]
        _append_history(state, prediction_error_ec, "ec")

    # 段階14 Step A: cumulative information gain を同時 update (PLAN §3-2)
    # 段階14.6: tool_name を渡して per_tool cig も更新 (PLAN §10-4 literal、F-006 fix)
    update_cumulative_information_gain(
        state, prediction_error, prediction_error_ec, tool_name=tool_name
    )


# ============================================================
# 段階14 Step A: Cumulative Information Gain Observable (∑I_t)
# ============================================================
#
# STAGE14 PLAN §3 の実装。predicted_e2 / predicted_ec の prediction
# error history から mutual information の累積近似を state field 化、
# reflect 連続中の「情報利得 flat」を第一級の観測量として表現する。
#
# flat_signal: 直近 N=CIG_WINDOW cycle の |error| 平均が閾値
# (e2 軸 0-100 scale 上の CIG_THRESHOLD_RAW * 100 = 5.0、PLAN §10-7
# threshold=0.05) 未満なら True。
# flat_streak: flat_signal=True が連続した cycle 数。basin transition
# (Step D) の trigger 入力。

CIG_WINDOW = 10
CIG_THRESHOLD_RAW = 0.05  # PLAN §10-7、e2 軸 0-100 scale で × 100 適用


def _new_cig_entry() -> dict:
    """cig entry (global / per-tool) の初期スキーマ。"""
    return {
        "e2_total": 0.0, "ec_total": 0.0,
        "e2_window_mean": 0.0, "ec_window_mean": 0.0,
        "flat_signal": False, "flat_streak": 0,
    }


def update_cumulative_information_gain(
    state: dict,
    prediction_error: float,
    prediction_error_ec: Optional[float] = None,
    tool_name: Optional[str] = None,
) -> None:
    """予測誤差から ∑I_t / window mean / flat_signal / flat_streak を更新。

    state["cumulative_information_gain"] dict を破壊的に更新する。
    呼出し点は update_predictor_confidence() 末尾 (history append 後)。
    Step C (β 動的化) / Step D (basin transition) の入力源。

    段階14.6 (PLAN §10-4 literal「tool 別 (predictor_confidence 同型)」、
    F-006 fix): tool_name 指定時は cig["per_tool"][tool_name] にも同 logic
    で update。tool 別 mini history (FIFO CIG_WINDOW) を持って window_mean /
    flat_signal / flat_streak を計算、global cig と独立して動く。Step C tool
    別 β scope の入力源。tool_name 未指定 (None / 空文字) なら global のみ更新
    (backward compat、Step D / 既存 test 維持)。

    Args:
        state: state dict
        prediction_error: E2 軸の予測誤差 (abs 値、0-100 scale)
        prediction_error_ec: EC 軸の予測誤差 (abs 値、0-1 scale)。None なら
            ec_total は不変、ec_window_mean は history があれば再計算
        tool_name: 段階14.6 tool 別 cig 入力源。指定なしで global のみ
    """
    cig = state.setdefault("cumulative_information_gain", _new_cig_entry())
    threshold = CIG_THRESHOLD_RAW * 100

    # global: 既存 logic (history は state global、Step D / 既存 test 互換)
    cig["e2_total"] += abs(float(prediction_error))
    if prediction_error_ec is not None:
        cig["ec_total"] += abs(float(prediction_error_ec))
    h_e2 = state.get("prediction_error_history_e2", [])[-CIG_WINDOW:]
    cig["e2_window_mean"] = sum(h_e2) / len(h_e2) if h_e2 else 0.0
    h_ec = state.get("prediction_error_history_ec", [])[-CIG_WINDOW:]
    cig["ec_window_mean"] = sum(h_ec) / len(h_ec) if h_ec else 0.0
    flat = cig["e2_window_mean"] < threshold
    cig["flat_signal"] = flat
    cig["flat_streak"] = cig["flat_streak"] + 1 if flat else 0

    # 段階14.6: tool 別 cig を per_tool に同 logic で更新 (PLAN §10-4 literal)
    if tool_name:
        per_tool = cig.setdefault("per_tool", {})
        tcig = per_tool.setdefault(tool_name, {
            **_new_cig_entry(),
            "history_e2": [], "history_ec": [],
            # call_count: 累積 update 回数 (Codex P2 fix、PLAN §5-2 literal
            # 「cycle 数」直訳)。history_e2 は FIFO CIG_WINDOW で cap されるが
            # call_count は all-time、i_avg = e2_total / call_count で
            # 分子分母の時間窓を整合させる。
            "call_count": 0,
        })
        tcig["call_count"] += 1
        tcig["e2_total"] += abs(float(prediction_error))
        tcig["history_e2"].append(abs(float(prediction_error)))
        if len(tcig["history_e2"]) > CIG_WINDOW:
            del tcig["history_e2"][0]
        tcig["e2_window_mean"] = (
            sum(tcig["history_e2"]) / len(tcig["history_e2"])
            if tcig["history_e2"] else 0.0
        )
        if prediction_error_ec is not None:
            tcig["ec_total"] += abs(float(prediction_error_ec))
            tcig["history_ec"].append(abs(float(prediction_error_ec)))
            if len(tcig["history_ec"]) > CIG_WINDOW:
                del tcig["history_ec"][0]
            tcig["ec_window_mean"] = (
                sum(tcig["history_ec"]) / len(tcig["history_ec"])
                if tcig["history_ec"] else 0.0
            )
        flat_tool = tcig["e2_window_mean"] < threshold
        tcig["flat_signal"] = flat_tool
        tcig["flat_streak"] = tcig["flat_streak"] + 1 if flat_tool else 0


# ============================================================
# 段階14 Step C: Dynamic β Lower Bound (EFE-Internal)
# ============================================================
#
# STAGE14 PLAN §5 の実装。Curiosity is Knowledge (2026) の不等式
# β >= E[h(y)] / I(s; (x,y)) を直訳、flat_signal=True (情報利得 flat) なら
# β を上げて pragmatic value 抑制 = epistemic value 相対的増加 = 候補空間
# expand。β は cap で runaway 抑制。
#
# 入力: Step A の state["cumulative_information_gain"]
# 出力: β (PLAN §10-4 tool 別拡張時 candidate 引数活用)、
# controller._predicted_outcome_multiplier で combined / beta に逆作用。

BETA_BASE = 0.5                # 既存挙動相当 (PLAN §5-2 literal)
BETA_CAP = 2.0                 # runaway 抑制 cap (PLAN §5-2 literal)
BETA_TRIGGER_STREAK = 2        # flat_streak >= で動的化発火 (PLAN §5-2 literal)
# 段階14 Step D: phase_transition_pending 時の β boost (PLAN §6-2 literal)
BETA_TRANSITION_BOOST = 1.5    # transition pending で β を 1.5x boost
BETA_TRANSITION_CAP = 3.0      # transition pending 時のみの最大 cap (通常 cap=2.0)


def _compute_dynamic_beta(state: dict, candidate: dict) -> float:
    """β = max(β_base, E[h]/I) の lower bound 直訳 (PLAN §5-2 literal)。

    Curiosity is Knowledge (2026) の不等式 β >= E[h(y)] / I(s; (x,y)) を
    flat_signal/streak ベースで動的算出。flat_streak < BETA_TRIGGER_STREAK
    で β = BETA_BASE (現挙動維持、対症療法回避)、>= で動的増、cap = BETA_CAP。

    段階14.6 (PLAN §10-4 literal「tool 別 (predictor_confidence 同型)」、
    F-006 fix): candidate.tool が cig["per_tool"][tool] に entry あれば
    tool 別 cig を入力源に、なければ global cig fallback。各 tool が独立
    した flat_streak / window_mean を持つので、ある tool だけ flat になって
    その tool に対してのみ β > BETA_BASE が出る挙動が成立する。

    Args:
        state: state dict
        candidate: 当該 candidate (candidate["tool"] で per_tool cig を選択)

    Returns:
        β float (BETA_BASE <= β <= BETA_CAP、phase_transition 時 BETA_TRANSITION_CAP)
    """
    cig = state.get("cumulative_information_gain", {})

    # 段階14.6: tool 別 cig 優先 (PLAN §10-4 literal「predictor_confidence 同型」)
    tool_name = str(candidate.get("tool", "")) if candidate else ""
    per_tool_cig = (
        cig.get("per_tool", {}).get(tool_name) if tool_name else None
    )
    source_cig = per_tool_cig if per_tool_cig else cig

    flat_streak = source_cig.get("flat_streak", 0)

    if flat_streak < BETA_TRIGGER_STREAK:
        # flat_streak 閾値未達: β=BETA_BASE 起点 (現挙動維持、対症療法回避)
        beta_dynamic = BETA_BASE
    else:
        # lower bound β >= E[h] / I を直訳 (PLAN §5-2 literal)
        # E[h] ~= window_mean (予測の不確実性、e2 軸 0-100 scale を 0-1 に normalize)
        # I ~= e2_total / cycle 数 (累積情報利得率、同 normalize)
        e_h = max(0.01, source_cig.get("e2_window_mean", 50.0) / 100.0)
        # i_avg の divisor: PLAN §5-2 literal「cycle 数」(全 update 数) 直訳。
        # per_tool は tcig["call_count"] (Codex P2 fix、累積 update 数で
        # e2_total と窓整合)、global は state.prediction_error_history_e2 の
        # len (FIFO 100 cap、HISTORY_CAP 内なら全 update 数と一致)。
        if per_tool_cig:
            cycle_count = per_tool_cig.get("call_count", 1)
        else:
            cycle_count = len(state.get("prediction_error_history_e2", []))
        i_avg = max(0.01, source_cig.get("e2_total", 1.0) / max(1, cycle_count))
        i_avg_norm = i_avg / 100.0

        beta_required = e_h / max(0.01, i_avg_norm)
        beta_dynamic = min(BETA_CAP, max(BETA_BASE, beta_required))

    # 段階14 Step D: phase_transition_pending で β を更に boost (PLAN §6-2 literal)。
    # basin 滞留閾値超過 → 確率的脱出促進 (強制脱出でなく β 経由間接誘導、
    # memo line 138-140「対症療法でない」+ feedback_freedom_to_die 整合)。
    # Step C の flat_streak < trigger 経路 (β=BETA_BASE) でも適用 = basin 単独
    # 滞留 (情報利得 flat ではない but 同 cluster 滞留) で脱出促進発火。
    bs = state.get("basin_state", {})
    if bs.get("phase_transition_pending"):
        beta_dynamic = min(BETA_TRANSITION_CAP, beta_dynamic * BETA_TRANSITION_BOOST)

    return beta_dynamic


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
                    predicted_e2: int = 50,
                    predicted_ec: Optional[float] = None) -> dict:
    """{category, confidence, detail, predicted_e2, predicted_ec?} 形式の予測 dict。

    不正な category は OTHER に fallback、confidence は [0.0, 1.0] にクランプ、
    predicted_e2 は [0, 100] にクランプ (段階9)。
    predicted_ec は [0.0, 1.0] にクランプ (段階10 柱 C、省略時 None)。
    """
    if category not in _VALID_CATEGORIES:
        category = CATEGORY_OTHER
    conf = max(0.0, min(1.0, float(confidence)))
    try:
        pe2 = max(0, min(100, int(predicted_e2)))
    except (TypeError, ValueError):
        pe2 = 50
    result = {
        "category": category,
        "confidence": conf,
        "detail": str(detail),
        "predicted_e2": pe2,
    }
    if predicted_ec is not None:
        try:
            result["predicted_ec"] = max(0.0, min(1.0, float(predicted_ec)))
        except (TypeError, ValueError):
            pass  # 不正値は付加しない (None 扱い)
    return result


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
            # 段階10 柱 B/C: 学習した e2_conf/ec_conf による selection 接続は
            # controller._predicted_outcome_multiplier に一本化 (案 (イ) 役割分離)。
            # MediumPredictor は LLM① の生予測を返す純粋な役割に徹する。
            # predicted_ec も candidate.prediction から素通しする (柱 C)。
            return make_prediction(
                category=CATEGORY_OTHER,  # category は補助情報、multiplier は predicted_e2/ec 主
                confidence=prediction.get("confidence", 0.7),
                detail="medium",
                predicted_e2=prediction.get("predicted_e2", 50),
                predicted_ec=prediction.get("predicted_ec"),
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


# ============================================================
# 段階10.5 Fix 1: chain 粒度 migration + ec clamp helper
# ============================================================


def migrate_chain_keys(state: dict) -> int:
    """state["predictor_confidence"] の "+" 含むキー (chain 連結キー) を drop。

    段階10 Step 3 smoke で tool 連結文字列をキーに chain 単位学習していた
    15 種 entry を新 smoke 起点で破棄。tool 単位 entry のみ残し、Fix 1 で
    tool 別自己学習重みが本来意図通りに動作するよう正規化。

    Returns:
        drop した entry 数 (0 なら migration 不要)
    """
    pc = state.get("predictor_confidence")
    if not isinstance(pc, dict):
        return 0
    drop_keys = [k for k in pc if "+" in k]
    for k in drop_keys:
        pc.pop(k, None)
    return len(drop_keys)


def clamp_ec(value) -> float:
    """effective_change (ec) を 0.0-1.0 に clamp。

    段階10 Step 3 smoke で actual_ec=1.5 等の記録が 2 件発生、eff_change が
    稀に 1.0 を超えるケースで学習値が壊れる防止。不正値 (None/str 等) は
    0.0 に fallback (neutral/安全側)。
    """
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
