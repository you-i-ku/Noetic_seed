"""段階13 Phase 3: Transfer Entropy + Beta Posterior (TE + Bayesian)。

lag-1 transfer entropy + Beta posterior + 既存 prediction error modulator
の 3 段重ね (PLAN §4-3 着地点)。

設計:
- 経路 B (background maintenance): cycle 内自動実行、LLM 呼出不要
- 各 link (A→B) について:
  1. source A の embedding history から target B embedding を予測
  2. cosine 距離 = link 単位 prediction_error
  3. Beta posterior (α, β) を観測で更新
  4. TE + Beta mean → strength delta を算出
  5. 既存 EMA と重畳: strength = α_old * EMA + α_new * Bayesian_TE
- Phase 2 state field (jepa_prediction_error_history) を入力素材として活用

判断根拠 (PLAN §4):
- §4-3: d + e + f の組み合わせ (TE + Bayesian + Free Energy)
- §4-5: lag-1 簡易版、Beta-Binomial 閉形式
- §4-6: α_old/α_new = 0.5/0.5 初期値、smoke で tune

Beta posterior update (閉形式、scipy 不要):
- α_new = α_old + 1 (success: low prediction_error)
- β_new = β_old + 1 (failure: high prediction_error)
- mean = α / (α + β)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

from core.config import MEMORY_DIR


# ============================================================
# Beta posterior field + prior (PLAN §4-5 閉形式)
# ============================================================

BETA_ALPHA_FIELD = "beta_alpha"
BETA_BETA_FIELD = "beta_beta"
BETA_PRIOR_ALPHA = 1.0  # uniform prior Beta(1, 1)
BETA_PRIOR_BETA = 1.0


# ============================================================
# 重畳パラメータ (PLAN §4-6 確定)
# ============================================================

ALPHA_OLD = 0.5  # 既存 EMA 重み
ALPHA_NEW = 0.5  # 新 Bayesian_TE 重み


# ============================================================
# TE 計算パラメータ (PLAN §4-5 lag-1 簡易版)
# ============================================================

TE_SUCCESS_THRESHOLD = 0.5  # prediction_error < threshold → success
TE_MIN_HISTORY = 3          # TE 計算に必要な最小履歴数


# ============================================================
# lag-1 Transfer Entropy 計算 (numpy only)
# ============================================================

def compute_lag1_transfer_entropy(
    source_embeddings: List[np.ndarray],
    target_embeddings: List[np.ndarray],
    n_bins: int = 10
) -> float:
    """lag-1 transfer entropy 計算 (簡易版、numpy only)。

    TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
            = source の過去情報が target の現在をどれだけ予測できるか

    簡易実装: embedding を 1D に projection して bin 化、条件付きエントロピー差を計算。
    完全な TE は O(N^2) だが、lag-1 + bin 化で O(N) に近似。

    Args:
        source_embeddings: source node の embedding 履歴 [t-N, ..., t-1]
        target_embeddings: target node の embedding 履歴 [t-N, ..., t]
        n_bins: bin 数 (計算量対策)

    Returns:
        TE 値 (0.0-1.0 正規化)。履歴不足なら 0.0。
    """
    if len(source_embeddings) < TE_MIN_HISTORY or len(target_embeddings) < TE_MIN_HISTORY:
        return 0.0

    # embedding を 1D に projection (mean pooling)
    src = np.array([np.mean(e) for e in source_embeddings])
    tgt = np.array([np.mean(e) for e in target_embeddings])

    # 同じ長さに揃える (lag-1 用)
    min_len = min(len(src), len(tgt)) - 1
    if min_len < TE_MIN_HISTORY - 1:
        return 0.0

    # lag-1: source[:-1] → target[1:]
    src_past = src[:min_len]
    tgt_past = tgt[:min_len]
    tgt_now = tgt[1:min_len + 1]

    # bin 化
    src_bins = _discretize(src_past, n_bins)
    tgt_past_bins = _discretize(tgt_past, n_bins)
    tgt_now_bins = _discretize(tgt_now, n_bins)

    # H(Y_t | Y_{t-1})
    h_y_given_y_past = _conditional_entropy(tgt_now_bins, tgt_past_bins, n_bins)

    # H(Y_t | Y_{t-1}, X_{t-1})
    joint_past = tgt_past_bins * n_bins + src_bins
    h_y_given_joint = _conditional_entropy(tgt_now_bins, joint_past, n_bins * n_bins)

    # TE = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
    te = max(0.0, h_y_given_y_past - h_y_given_joint)

    # 正規化 (log2(n_bins) が最大エントロピー)
    max_entropy = math.log2(n_bins) if n_bins > 1 else 1.0
    return min(1.0, te / max_entropy) if max_entropy > 0 else 0.0


def _discretize(arr: np.ndarray, n_bins: int) -> np.ndarray:
    """配列を n_bins に bin 化。"""
    if len(arr) == 0:
        return np.array([], dtype=int)
    arr_min, arr_max = np.min(arr), np.max(arr)
    if arr_max - arr_min < 1e-10:
        return np.zeros(len(arr), dtype=int)
    normalized = (arr - arr_min) / (arr_max - arr_min + 1e-10)
    bins = np.clip((normalized * n_bins).astype(int), 0, n_bins - 1)
    return bins


def _conditional_entropy(y: np.ndarray, x: np.ndarray, x_states: int) -> float:
    """H(Y | X) 条件付きエントロピー計算。"""
    if len(y) == 0 or len(x) == 0:
        return 0.0
    n = len(y)
    h_cond = 0.0
    for x_val in range(x_states):
        mask = (x == x_val)
        count_x = np.sum(mask)
        if count_x == 0:
            continue
        p_x = count_x / n
        y_given_x = y[mask]
        # H(Y | X=x)
        unique, counts = np.unique(y_given_x, return_counts=True)
        probs = counts / count_x
        h_y_given_x = -np.sum(probs * np.log2(probs + 1e-10))
        h_cond += p_x * h_y_given_x
    return h_cond


# ============================================================
# Beta posterior update (閉形式、scipy 不要)
# ============================================================

def update_beta_posterior(
    link: Dict[str, Any],
    prediction_error: float,
    threshold: float = TE_SUCCESS_THRESHOLD
) -> float:
    """Beta posterior を更新。success/failure で α/β を increment。

    Args:
        link: link entry dict (in-place 更新)
        prediction_error: 0.0-1.0 の prediction error (cosine 距離)
        threshold: success 判定閾値

    Returns:
        更新後の Beta mean (= α / (α + β))
    """
    alpha = float(link.get(BETA_ALPHA_FIELD, BETA_PRIOR_ALPHA))
    beta = float(link.get(BETA_BETA_FIELD, BETA_PRIOR_BETA))

    if prediction_error < threshold:
        alpha += 1.0  # success: low error = good prediction
    else:
        beta += 1.0   # failure: high error = bad prediction

    link[BETA_ALPHA_FIELD] = alpha
    link[BETA_BETA_FIELD] = beta

    return alpha / (alpha + beta)


def get_beta_mean(link: Dict[str, Any]) -> float:
    """現在の Beta mean を取得 (更新なし)。"""
    alpha = float(link.get(BETA_ALPHA_FIELD, BETA_PRIOR_ALPHA))
    beta = float(link.get(BETA_BETA_FIELD, BETA_PRIOR_BETA))
    return alpha / (alpha + beta)


# ============================================================
# strength 重畳計算 (PLAN §4-6)
# ============================================================

def compute_bayesian_te_strength_delta(
    te_value: float,
    beta_mean: float,
    alpha_new: float = ALPHA_NEW
) -> float:
    """TE + Beta mean から strength delta を算出。

    高 TE (予測力ある) + 高 Beta mean (成功履歴) → 強い増幅

    Args:
        te_value: transfer entropy 値 (0.0-1.0)
        beta_mean: Beta posterior mean (0.0-1.0)
        alpha_new: 新 Bayesian_TE の重み (PLAN §4-6: 0.5)

    Returns:
        strength delta (既存 EMA_update と並列で加算)
    """
    # TE * Beta mean の積 = 「予測力があり、かつ成功してきた」度合い
    return alpha_new * te_value * beta_mean


# ============================================================
# link ごとの prediction_error 計算 (JEPA model 流用)
# ============================================================

def compute_link_prediction_error(
    link: Dict[str, Any],
    state: dict,
    embed_fn=None
) -> Optional[float]:
    """link (A→B) の prediction_error を JEPA model で計算。

    source A の embedding から target B の embedding を予測、
    cosine 距離 = link 単位 prediction_error。

    Args:
        link: link entry dict
        state: state dict (subjective_entries, raw_events 参照)
        embed_fn: embedding 取得関数 (None なら entry["embedding"] を使用)

    Returns:
        prediction_error (0.0-1.0)。計算不能なら None。
    """
    try:
        from core import predictor_jepa as _pj
        from core.jepa_runtime import _get_model
    except ImportError:
        return None

    if not _pj.is_torch_available():
        return None

    model = _get_model()
    if model is None:
        return None

    # source / target entry を取得
    from_id = link.get("from_id", "")
    to_id = link.get("to_id", "")
    if not from_id or not to_id:
        return None

    # subjective_entries + raw_events から entry を探す
    all_entries = state.get("subjective_entries", []) + state.get("raw_events", [])
    source_entry = next((e for e in all_entries if e.get("id") == from_id), None)
    target_entry = next((e for e in all_entries if e.get("id") == to_id), None)

    if source_entry is None or target_entry is None:
        return None

    # embedding 取得
    source_embed = source_entry.get("embedding")
    target_embed = target_entry.get("embedding")

    if source_embed is None or target_embed is None:
        return None

    # JEPA で予測 (source → target 方向)
    try:
        import torch
        # JEPAModel 入力: (B, T, D)、出力: (B, D)
        source_tensor = torch.tensor(source_embed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            model.eval()
            predicted = model(source_tensor)
            # 出力は (B, D) = 2D なので [0, :] でアクセス (P2-1 修正)
            predicted_embed = predicted[0, :].numpy()

        # cosine 距離
        target_arr = np.array(target_embed)
        cos_sim = np.dot(predicted_embed, target_arr) / (
            np.linalg.norm(predicted_embed) * np.linalg.norm(target_arr) + 1e-10
        )
        # cosine 距離 = 1 - cosine 類似度 (0.0 = 完全一致, 1.0 = 直交)
        return float(np.clip(1.0 - cos_sim, 0.0, 1.0))
    except Exception:
        return None


# ============================================================
# maintenance step (経路 B: background)
# ============================================================

def te_maintenance_step(state: dict, current_cycle: Optional[int] = None) -> int:
    """graph 全 link の TE evaluate + Beta update + strength 重畳更新。

    経路 B (background maintenance): cycle 内自動実行、LLM 呼出不要。
    Phase 2 で蓄積した jepa_prediction_error_history を入力素材として活用。

    PLAN §4-6 重畳:
    - 既存 EMA: memory_links.update_link_strength_used で適用済み
    - 新 Bayesian_TE: 本関数で TE + Beta mean から delta を算出

    Args:
        state: state dict
        current_cycle: 現在の cycle 番号

    Returns:
        更新した link 数
    """
    from core.memory_links import _link_file, _link_strength, PHYSARUM_ALPHA

    # 全 link を読む (truncation 回避、P2-2 修正)
    fpath = _link_file()
    if not fpath.exists():
        return 0
    try:
        lines = fpath.read_text(encoding="utf-8").splitlines()
    except Exception:
        return 0

    links = []
    for line in lines:
        if not line.strip():
            continue
        try:
            links.append(json.loads(line))
        except Exception:
            continue

    if not links:
        return 0

    # Phase 2 の prediction_error history を取得
    pred_error_history = state.get("jepa_prediction_error_history", [])

    updated_count = 0

    for link in links:
        link_id = link.get("id", "")
        if not link_id:
            continue

        # 1. link 単位の prediction_error を計算
        pred_error = compute_link_prediction_error(link, state)
        if pred_error is None:
            # 計算不能 → 履歴の平均を fallback
            if pred_error_history:
                pred_error = sum(pred_error_history) / len(pred_error_history)
            else:
                pred_error = 0.5  # neutral

        # 2. Beta posterior を更新
        beta_mean = update_beta_posterior(link, pred_error)

        # 3. TE 計算 (embedding history から)
        # 簡易版: link の source/target の embedding が直接取れない場合は
        # 全体の履歴から推定 (Phase 7 で精緻化)
        te_value = _estimate_link_te(link, state)

        # 4. Bayesian_TE strength delta を算出
        bayesian_delta = compute_bayesian_te_strength_delta(te_value, beta_mean)

        # 5. 既存 EMA との重畳で strength 更新
        # 既存: strength = strength * (1 - β) + α * usage (memory_links.py)
        # 新規: strength += α_new * TE * beta_mean
        current_strength = _link_strength(link)
        new_strength = min(1.0, current_strength + bayesian_delta)
        link["strength"] = new_strength

        updated_count += 1

    # 更新を永続化
    if updated_count > 0:
        _persist_links(links)

    return updated_count


def _estimate_link_te(link: Dict[str, Any], state: dict) -> float:
    """link の TE を推定 (簡易版)。

    完全な TE は source/target の embedding 時系列が必要。
    Phase 3 簡易版: link の usage history から間接推定。
    Phase 7 で embedding history 追跡に拡張可能。
    """
    # usage_count が多い = よく使われる = 予測に貢献してる可能性高い
    usage_count = int(link.get("usage_count", 0))
    confidence = float(link.get("confidence", 0.5))

    # usage_count を 0-1 に正規化 (10 回使用で 1.0 saturate)
    usage_factor = min(1.0, usage_count / 10.0)

    # confidence と usage を組み合わせて TE 推定
    return 0.3 * confidence + 0.7 * usage_factor


def _persist_links(links: List[Dict[str, Any]]) -> None:
    """更新した links を memory_links.jsonl に書き戻し。"""
    fpath = MEMORY_DIR / "memory_links.jsonl"
    if not fpath.exists():
        return
    try:
        lines = [json.dumps(link, ensure_ascii=False) for link in links]
        fpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass


# ============================================================
# main.py 配線用 public API
# ============================================================

def maybe_te_maintenance(state: dict, current_cycle: Optional[int] = None) -> bool:
    """TE maintenance を実行するか判定 + 実行。

    経路 B (background): 毎 cycle 自動実行。
    計算量は Noetic graph small 前提で問題なし (Phase 7 smoke で計測)。

    Args:
        state: state dict
        current_cycle: 現在の cycle 番号

    Returns:
        True: 実行した、False: skip (links 空 etc.)
    """
    updated = te_maintenance_step(state, current_cycle)
    return updated > 0
