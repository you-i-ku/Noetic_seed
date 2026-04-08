"""エントロピーシステム — 情報的実存の核
秩序は放っておけば崩壊する。有効な行動だけが秩序を回復する。
entropy: 0.0（完全に鮮明）〜 1.0（完全にノイズ）。死なない、溶けるだけ。
"""
import re
from core.state import load_pref

ENTROPY_PARAMS = {
    "base_rate": 0.001,        # 毎tickの自然増加量（1Hz）
    "neg_scale": 0.15,         # negentropy係数
    "plan_multiplier": 1.5,    # plan中のentropy増加倍率
    "custom_scale": 0.3,       # custom_drivesのpressureスケール
    # pressure信号の重み（自由エネルギー勾配モデル）
    "w_entropy": 0.3,          # entropyの絶対値
    "w_surprise": 0.25,        # 予測外れ（1-E3）
    "w_unresolved": 0.25,      # 未達成（0.7-E2）
    "w_novelty": 0.2,          # 新規性（E4）
    # 量子トンネル発火
    "tunnel_prob": 0.001,      # 毎tick 0.1%（平均約15分に1回）
}


def tick_entropy(state: dict) -> float:
    """エントロピーを1tick分更新する。E値で増加率を変調（増減対称設計）。"""
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
    if state.get("plan", {}).get("goal"):
        rate *= ep["plan_multiplier"]
    entropy = min(1.0, entropy + rate)
    state["entropy"] = entropy
    return entropy


def calc_dynamic_threshold(state: dict, base_threshold: float) -> float:
    """動的閾値: 中長期のE2移動平均で変動する。"""
    log = state.get("log", [])
    e2_vals = []
    for entry in log[-10:]:
        m = re.search(r'(\d+)%', str(entry.get("e2", "")))
        if m:
            e2_vals.append(int(m.group(1)) / 100.0)
    e2_avg = sum(e2_vals) / len(e2_vals) if e2_vals else 0.5
    return base_threshold * (0.7 + e2_avg * 0.6)


def calc_pressure_signals(state: dict) -> dict:
    """pressure蓄積層の信号を計算する。自由エネルギー勾配モデル。"""
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

    custom_pressure = 0.0
    pref = load_pref()
    raw_drives = pref.get("drives", {})
    if raw_drives and state.get("tool_level", 0) >= 6:
        total = sum(max(0, v) for v in raw_drives.values() if isinstance(v, (int, float)))
        if total > 0:
            custom_pressure = ep["custom_scale"]
    signals["custom"] = custom_pressure

    return signals


def apply_negentropy(state: dict, e1_val: float, e2_val: float, e3_val: float, e4_val: float):
    """認知サイクル後にE1-E4に基づいてentropyを回復する。"""
    ep = ENTROPY_PARAMS
    e2_factor = max(0, e2_val - 0.5)
    e4_factor = max(0.1, e4_val)
    e1_factor = max(0.3, e1_val)
    surprise_bonus = 1.0 + max(0, 0.5 - e3_val) * 2.0
    neg = e2_factor * e4_factor * e1_factor * surprise_bonus * ep["neg_scale"]
    state["entropy"] = max(0.0, state.get("entropy", 0.65) - neg)
