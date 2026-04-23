"""エントロピーシステム — 情報的実存の核
秩序は放っておけば崩壊する。有効な行動だけが秩序を回復する。
entropy: 0.0（完全に鮮明）〜 1.0（完全にノイズ）。死なない、溶けるだけ。

measured_entropy: 実測エントロピー（ツール分布・intent多様性・state充実度・sandbox介入度）
conceptual_entropy: 概念的エントロピー（base_rate × E値変調 + measured補正）
"""
import re
from core.state import load_pref

ENTROPY_PARAMS = {
    "base_rate": 0.001,
    "neg_scale": 0.15,
    "custom_scale": 0.3,
    "w_entropy": 0.3,
    "w_surprise": 0.25,
    "w_unresolved": 0.25,
    "w_novelty": 0.2,
    "w_stagnation": 0.3,
    "w_unresolved_ext": 0.2,
    "w_pending_burden": 0.3,  # 段階8 改善6-D: 未消化 pending 総 priority → pressure
    "w_prediction_error": 0.3,  # 段階10.5 Tune 1 (smoke 20 cycle): pe fire 90% 占有で 1.0 → 0.3 下方修正 (PLAN §7-1 の > 50% 該当)
    "tunnel_prob": 0.001,
    "measured_feedback_rate": 0.1,
    "entropy_floor_base": 0.15,
    "entropy_floor_energy_coeff": 0.001,
    "entropy_floor_cap": 0.30,
    "stagnation_coeff": 0.3,
}


def _entropy_floor(state: dict) -> float:
    """動的entropy floor: energy依存。成長するほど完全な安定は不可能になる。"""
    ep = ENTROPY_PARAMS
    energy = state.get("energy", 50)
    floor = ep["entropy_floor_base"] + energy * ep["entropy_floor_energy_coeff"]
    return min(ep["entropy_floor_cap"], floor)


def tick_entropy(state: dict, measured_entropy: float | None = None,
                 behavioral_entropy: float | None = None) -> float:
    """エントロピーを1tick分更新する。E値で増加率を変調 + measured_entropyで補正。
    behavioral_entropy低（パターン化）→増加率加速。"""
    ep = ENTROPY_PARAMS
    entropy = state.get("entropy", 0.65)

    last_e1 = state.get("last_e1", 0.5)
    last_e2 = state.get("last_e2", 0.5)
    last_e3 = state.get("last_e3", 0.5)
    last_e4 = state.get("last_e4", 0.5)

    e2_factor = 1.0 + max(0, 0.7 - last_e2) * 2.0
    e4_factor = 1.0 + max(0, 0.5 - last_e4) * 2.0
    e1_factor = 1.0 + max(0, 0.5 - last_e1) * 1.5
    e3_factor = 1.0 + max(0, last_e3 - 0.5) * 1.5

    rate = ep["base_rate"] * e2_factor * e4_factor * e1_factor * e3_factor

    # 行動パターン化（behavioral_entropy低）→entropy増加加速
    if behavioral_entropy is not None:
        stagnation_factor = 1.0 + (1.0 - behavioral_entropy) * ep["stagnation_coeff"]
        rate *= stagnation_factor

    entropy += rate

    # measured_entropyとのギャップ補正
    if measured_entropy is not None:
        gap = measured_entropy - entropy
        entropy += gap * ep["measured_feedback_rate"]

    floor = _entropy_floor(state)
    entropy = max(floor, min(1.0, entropy))
    state["entropy"] = entropy
    return entropy


def calc_dynamic_threshold(state: dict, base_threshold: float) -> float:
    """動的閾値: E2移動平均で鷹揚さ、entropyで緊急度を加味。
    E2高い→閾値上がる（余裕）。entropy高い→閾値下がる（秩序崩壊で敏感に）。"""
    log = state.get("log", [])
    e2_vals = []
    for entry in log[-10:]:
        m = re.search(r'(\d+)%', str(entry.get("e2", "")))
        if m:
            e2_vals.append(int(m.group(1)) / 100.0)
    e2_avg = sum(e2_vals) / len(e2_vals) if e2_vals else 0.5
    entropy = state.get("entropy", 0.65)
    # E2係数: 成功してるほど鷹揚
    e2_factor = 0.7 + e2_avg * 0.6
    # entropy係数: 秩序崩壊してるほど敏感（閾値を下げる）
    entropy_factor = 1.0 - entropy * 0.3  # entropy=0→1.0, entropy=0.9→0.73, entropy=1.0→0.7
    return base_threshold * e2_factor * entropy_factor


def calc_pressure_signals(state: dict, spiral: dict | None = None) -> dict:
    """pressure蓄積層の信号を計算する。自由エネルギー勾配 + 螺旋停滞検出 + 未応答外部入力。"""
    ep = ENTROPY_PARAMS
    entropy = state.get("entropy", 0.65)
    last_e2 = state.get("last_e2", 0.5)
    last_e3 = state.get("last_e3", 0.5)
    last_e4 = state.get("last_e4", 0.5)

    signals = {
        "entropy":    entropy * ep["w_entropy"],
        "surprise":   max(0, 1.0 - last_e3) * ep["w_surprise"],
        "unresolved": max(0, 0.7 - last_e2) * ep["w_unresolved"],
        "novelty":    max(0, last_e4) * ep["w_novelty"],
    }

    # 未応答の外部入力による持続的圧力
    unresolved_ext = min(0.5, state.get("unresolved_external", 0.0))
    signals["unresolved_ext"] = unresolved_ext * ep["w_unresolved_ext"]

    # 段階8 改善6-D: 未消化 pending の総 priority を burden として加算。
    # ジレンマ成立 (pending 溜まる → pressure 上 → wait(dismiss) 誘発)。
    # feedback_cognitive_load_via_pressure 整合 (情報理論的負荷、擬似感情化せず)。
    pending_list = state.get("pending", [])
    burden = 0.0
    for p in pending_list:
        if p.get("type") != "pending":
            continue
        if p.get("semantic_merge") is not True:
            continue
        if p.get("observed_content") is not None:
            continue  # 消化済 skip
        if p.get("expiry_policy") == "deprecated":
            continue  # 安全弁で deprecate 済は除外
        burden += float(p.get("priority", 0.0))
    # cap して過剰 pressure を防ぐ (他 signal と同レンジ 0-0.5)
    signals["pending_burden"] = min(0.5, burden * 0.1) * ep["w_pending_burden"]

    # 段階10 柱 A: 予測誤差由来の surprise-driven internal drive。
    # reflection.py:15 で参照されてた last_prediction_error を pressure 層にも接続。
    # Active Inference epistemic value (-log p(obs|model)) の正規化近似。
    # 0-100 scale の prediction_error を 0-1 に正規化し既存 w_* パターンに乗せる。
    last_pred_err = float(state.get("last_prediction_error", 0))
    base_pe = last_pred_err / 100.0

    # 段階11-B Phase 3 Step 3.4: reconciliation 由来 EC 誤差を同じ pressure signal に merge。
    # 共通 weight w_prediction_error を流用 (新規マジックナンバー 0)、別 w_contradiction は
    # 導入しない (PLAN §6-4 の「情報理論的に pressure 1 本に集約」決定通り)。
    # 直近 5 件 magnitude 平均を加算 (段階10 _is_match の 5 件 bootstrap と整合)、cap 1.0。
    recon_hist = state.get("prediction_error_history_by_source", {}).get("reconciliation", [])
    recon_pe = 0.0
    if recon_hist:
        recent = [float(h.get("magnitude", 0.0)) for h in recon_hist[-5:]]
        recon_pe = sum(recent) / max(1, len(recent))
    combined_pe = min(1.0, base_pe + recon_pe)

    signals["prediction_error"] = combined_pe * ep["w_prediction_error"]

    if spiral:
        signals["stagnation"] = max(0, 0.3 - spiral.get("magnitude", 0)) * ep["w_stagnation"]
    else:
        signals["stagnation"] = 0.0

    custom_pressure = 0.0
    pref = load_pref()
    raw_drives = pref.get("drives", {})
    if raw_drives and state.get("tool_level", 0) >= 6:
        total = sum(max(0, v) for v in raw_drives.values() if isinstance(v, (int, float)))
        if total > 0:
            custom_pressure = ep["custom_scale"]
    signals["custom"] = custom_pressure

    return signals


def apply_negentropy(state: dict, e1_val: float, e2_val: float, e3_val: float, e4_val: float,
                     state_change_bonus: float = 0.0, consistency_bonus: float = 0.0):
    """認知サイクル後にE1-E4 + state変化量 + 螺旋一貫性でentropyを回復する。"""
    ep = ENTROPY_PARAMS
    e2_factor = max(0, e2_val - 0.5)
    e4_factor = max(0.1, e4_val)
    e1_factor = max(0.3, e1_val)
    surprise_bonus = 1.0 + max(0, 0.5 - e3_val) * 2.0
    change_factor = 1.0 + state_change_bonus
    spiral_factor = 1.0 + max(0, consistency_bonus) * 0.5

    neg = e2_factor * e4_factor * e1_factor * surprise_bonus * change_factor * spiral_factor * ep["neg_scale"]
    floor = _entropy_floor(state)
    state["entropy"] = max(floor, state.get("entropy", 0.65) - neg)


# ============================================================
# 段階11-B Phase 3 Step 3.4: source 別 EC 予測誤差記録
# ============================================================

def record_ec_prediction_error(state: dict, *,
                               source: str,
                               magnitude: float,
                               reason: str = "",
                               context: dict | None = None,
                               history_max: int = 50) -> None:
    """source 別の EC 予測誤差を state に記録 (段階11-B Phase 3 Step 3.4)。

    段階10 既存経路 (predictor.update_predictor_confidence) を補完する副経路:
    - 既存 state["prediction_error_history_ec"] には magnitude を append
      (pressure 層経路との整合を保ち、共通 weight w_prediction_error を流用可能)
    - 新規 state["prediction_error_history_by_source"][source] に detail record を
      append (smoke 分析で source 別の pressure 寄与を追跡可能、hot/cold な
      source 識別 + reason / context metadata で判定根拠まで辿れる)

    pressure 層への実接続は Phase 3 Step 3.x (reconciliation module 側) で実施。
    本関数は state への history 書込のみ、pressure 計算は触らない (最小介入)。

    Args:
        state: 記録先 state dict (破壊更新)
        source: 識別 tag (例: "reconciliation" / 将来他層を追加する時も同関数で拡張)
        magnitude: EC 誤差 magnitude (0.0-1.0、severity スケール)
        reason: 判定根拠文字列 (smoke 分析用)
        context: 関連 entry id 等の metadata (例: {"new_entry_id": "...", ...})
        history_max: source 別 history の保持上限 (超えたら古いものから trim)
    """
    from datetime import datetime

    mag = float(magnitude)

    # 既存 EC history への append (pressure 経路との整合)
    hist_ec = state.setdefault("prediction_error_history_ec", [])
    hist_ec.append(mag)

    # 新規 source 別 detail history
    by_source = state.setdefault("prediction_error_history_by_source", {})
    src_hist = by_source.setdefault(source, [])
    src_hist.append({
        "magnitude": mag,
        "reason": reason,
        "context": context or {},
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    # 件数上限 trim (FIFO)
    if len(src_hist) > history_max:
        del src_hist[:len(src_hist) - history_max]
