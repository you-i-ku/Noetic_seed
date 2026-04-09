"""エントロピーシステム v2 — 情報的実存の核
秩序は放っておけば崩壊する。有効な行動だけが秩序を回復する。
entropy: 0.0（完全に鮮明）〜 1.0（完全にノイズ）。死なない、溶けるだけ。

v2変更:
- 動的floor（energy依存: 成長するほど凍れない）
- prediction_error → entropy加速（サプライズ = 自己モデルの危機）
- coherence_drop → entropy加速（自己と行動の乖離）
- behavioral_entropy → stagnation加速（パターン化検出）
- 5信号: entropy, surprise, pending, stagnation, drives
"""

ENTROPY_PARAMS = {
    "base_rate": 0.001,
    "neg_scale": 0.15,
    "plan_multiplier": 1.5,
    # 圧力信号の重み
    "w_entropy": 0.3,
    "w_surprise": 0.25,
    "w_pending": 0.25,
    "w_stagnation": 0.3,
    "w_drives": 0.2,
    # トンネル
    "tunnel_prob": 0.001,
    # entropy floor
    "entropy_floor_base": 0.15,
    "entropy_floor_energy_coeff": 0.001,
    "entropy_floor_cap": 0.30,
    # 加速係数
    "stagnation_coeff": 0.3,
    "prediction_error_coeff": 0.5,
    "coherence_drop_coeff": 0.3,
}


def _entropy_floor(state: dict) -> float:
    """動的entropy floor: energy依存。成長するほど完全な安定は不可能。"""
    ep = ENTROPY_PARAMS
    energy = state.get("energy", 50)
    floor = ep["entropy_floor_base"] + energy * ep["entropy_floor_energy_coeff"]
    return min(ep["entropy_floor_cap"], floor)


def tick_entropy(state: dict, behavioral_entropy: float | None = None,
                 prediction_error: float | None = None,
                 coherence_drop: float | None = None) -> float:
    """エントロピーを1tick分更新する。"""
    ep = ENTROPY_PARAMS
    entropy = state.get("entropy", 0.65)

    ev = state.get("last_e_values", {})
    achievement = ev.get("achievement", 0.5)
    diversity = ev.get("diversity", 0.5)

    # 低achievement → entropy加速
    ach_factor = 1.0 + max(0, 0.7 - achievement) * 2.0
    # 低diversity → entropy加速
    div_factor = 1.0 + max(0, 0.5 - diversity) * 2.0

    rate = ep["base_rate"] * ach_factor * div_factor

    # behavioral_entropy低（パターン化）→ 加速
    if behavioral_entropy is not None:
        rate *= 1.0 + (1.0 - behavioral_entropy) * ep["stagnation_coeff"]

    # prediction_error高 → 加速（サプライズ = 自己モデルの不整合）
    if prediction_error is not None:
        rate *= 1.0 + prediction_error * ep["prediction_error_coeff"]

    # coherence低下 → 加速（自己と行動の乖離）
    if coherence_drop is not None and coherence_drop > 0:
        rate *= 1.0 + coherence_drop * ep["coherence_drop_coeff"]

    if state.get("plan", {}).get("goal"):
        rate *= ep["plan_multiplier"]

    entropy += rate

    floor = _entropy_floor(state)
    entropy = max(floor, min(1.0, entropy))
    state["entropy"] = entropy
    return entropy


def calc_dynamic_threshold(state: dict, base_threshold: float) -> float:
    """動的閾値: achievement移動平均で鷹揚さ、entropyで緊急度。"""
    ev = state.get("last_e_values", {})
    achievement = ev.get("achievement", 0.5)
    entropy = state.get("entropy", 0.65)
    ach_factor = 0.7 + achievement * 0.6
    entropy_factor = 1.0 - entropy * 0.3
    return base_threshold * ach_factor * entropy_factor


def calc_pressure_signals(state: dict) -> dict:
    """5信号の圧力計算。"""
    ep = ENTROPY_PARAMS
    entropy = state.get("entropy", 0.65)
    ev = state.get("last_e_values", {})
    prediction = ev.get("prediction", 0.5)
    diversity = ev.get("diversity", 0.5)

    # pending: 優先度加重の未対応事項
    pending = state.get("pending", [])
    pending_pressure = sum(p.get("priority", 1.0) for p in pending)

    # disposition-based drives
    disp = state.get("disposition", {})
    curiosity = disp.get("curiosity", 0.5)
    drives_signal = max(0, curiosity - 0.3) * 0.5  # 好奇心が高いと自発的に動く

    signals = {
        "entropy":    entropy * ep["w_entropy"],
        "surprise":   max(0, 1.0 - prediction) * ep["w_surprise"],
        "pending":    min(2.0, pending_pressure * 0.3) * ep["w_pending"],
        "stagnation": max(0, 0.5 - diversity) * ep["w_stagnation"],
        "drives":     drives_signal * ep["w_drives"],
    }
    return signals


def apply_negentropy(state: dict, eval_result: dict):
    """評価結果からnegentropy適用。achievement × prediction × diversity。"""
    ep = ENTROPY_PARAMS
    achievement = eval_result.get("achievement", 0)
    prediction = eval_result.get("prediction", 0.5)
    diversity = eval_result.get("diversity", 0.5)

    # achievementが0なら negentropy = 0（変化なし = 秩序回復なし）
    if achievement <= 0:
        return

    neg = achievement * (0.5 + prediction * 0.5) * (0.5 + diversity * 0.5) * ep["neg_scale"]

    floor = _entropy_floor(state)
    state["entropy"] = max(floor, state.get("entropy", 0.65) - neg)
