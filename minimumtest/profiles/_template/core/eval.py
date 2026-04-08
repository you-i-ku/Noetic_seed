"""E値計算（E4多様性・energy更新）"""
import re
from core.embedding import _vector_ready, _embed_sync, cosine_similarity


def _calc_e4(current_intent: str, recent_entries: list, n: int = 5) -> str:
    """現在のintentが直近n件と異なるほど高い（反復=低、新規性=高）"""
    if not current_intent:
        return ""
    past_intents = [e["intent"] for e in recent_entries if e.get("intent")][-n:]
    if not past_intents:
        return ""

    if _vector_ready:
        try:
            vecs = _embed_sync([current_intent] + past_intents)
            if vecs and len(vecs) == 1 + len(past_intents):
                current_vec = vecs[0]
                sims = [cosine_similarity(current_vec, vecs[i + 1]) for i in range(len(past_intents))]
                avg_sim = sum(sims) / len(sims)
                return f"{round((1 - avg_sim) * 100)}%"  # 反転: 新規性スコア
        except Exception:
            pass

    # フォールバック: キーワード非一致の平均
    current_tokens = set(re.findall(r'\w+', current_intent.lower()))
    if not current_tokens:
        return ""
    ratios = []
    for past in past_intents:
        past_tokens = set(re.findall(r'\w+', past.lower()))
        if past_tokens:
            overlap = current_tokens & past_tokens
            ratios.append(len(overlap) / max(len(current_tokens), len(past_tokens)))
    if not ratios:
        return ""
    avg = round((1 - sum(ratios) / len(ratios)) * 100)  # 反転
    return f"{avg}%"


def _update_energy(state: dict, e2: str, e3: str, e4: str) -> float:
    """E値の平均から energy delta を計算。50%が損益分岐点。"""
    vals = []
    for e_str in (e2, e3, e4):
        m = re.search(r'(\d+)%', str(e_str))
        if m:
            vals.append(int(m.group(1)))
    if not vals:
        return 0.0
    e_mean = sum(vals) / len(vals)
    delta = e_mean / 50.0 - 1.0  # 50%で±0
    state["energy"] = max(0, min(100, state.get("energy", 50) + delta))
    return delta
