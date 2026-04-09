"""評価システム v2 — 完全プログラム計測（LLM評価なし）
4指標: achievement, prediction, diversity, coherence
"""
import json
import math
from collections import Counter
from difflib import SequenceMatcher
from core.embedding import _vector_ready, _embed_sync, cosine_similarity

# 外界に不可逆な作用を及ぼすツール
EXTERNAL_ACTION_TOOLS = {
    "output_display",
    "elyth_post", "elyth_reply", "elyth_like", "elyth_follow",
    "x_post", "x_reply", "x_quote", "x_like",
}


def evaluate_cycle(state_before: dict, state_after: dict,
                   tool_names: list[str], tool_result: str,
                   intent: str, expect: str) -> dict:
    """完全プログラム計測。LLM呼び出しなし。"""
    achievement = _calc_achievement(state_before, state_after, tool_names, tool_result)
    prediction = _calc_prediction(expect, tool_result)
    diversity = _calc_diversity(tool_names, state_after.get("log", []))
    coherence = _calc_coherence(state_after)

    return {
        "achievement": round(achievement, 4),
        "prediction": round(prediction, 4),
        "diversity": round(diversity, 4),
        "coherence": round(coherence, 4),
        "negentropy": 0.0,  # main.pyでapply_negentropy後に設定
    }


def _calc_achievement(state_before: dict, state_after: dict,
                      tool_names: list[str], tool_result: str) -> float:
    """実質的な情報変化量。0.0（変化なし）〜 1.5（大きな変化）。
    外界作用はシステム側で意味判定: 相手がいるか × 内容が新しいか（LLM不使用）。"""
    score = 0.0

    # self model変化量（value差分）
    old_self = state_before.get("self", {})
    new_self = state_after.get("self", {})
    for key in new_self:
        if key == "name":
            continue
        if key not in old_self:
            score += 1.0
        elif str(new_self[key]) != str(old_self[key]):
            old_v, new_v = str(old_self[key]), str(new_self[key])
            dist = 1.0 - SequenceMatcher(None, old_v, new_v).ratio()
            score += dist * 0.5

    # ファイル操作
    old_fw = set(state_before.get("files_written", []))
    new_fw = set(state_after.get("files_written", []))
    if new_fw - old_fw:
        score += 0.8

    old_fr = set(state_before.get("files_read", []))
    new_fr = set(state_after.get("files_read", []))
    if new_fr - old_fr:
        score += 0.5
    elif any(n == "read_file" for n in tool_names):
        score += 0.1

    # 外界作用 — システム側意味判定（LLM不使用、embedding使用）
    for tn in tool_names:
        if tn in EXTERNAL_ACTION_TOOLS:
            # ① 相手がいるか（構造判定: pendingにuser_messageがあるか）
            pending = state_after.get("pending", [])
            has_addressee = (
                any(p.get("type") == "user_message" for p in pending) or
                state_after.get("unresponded_external_count", 0) > 0
            )
            addressee_factor = 1.0 if has_addressee else 0.15

            # ② 内容の新規性（意味判定: 直近同一ツール出力との類似度）
            log = state_after.get("log", [])
            recent_same = [str(e.get("result", ""))[:300]
                           for e in log[-10:] if e.get("tool") == tn]
            content_novelty = 1.0
            if recent_same and _vector_ready:
                try:
                    texts = [tool_result[:300]] + recent_same[-3:]
                    vecs = _embed_sync(texts)
                    if vecs and len(vecs) >= 2:
                        sims = [cosine_similarity(vecs[0], vecs[i + 1])
                                for i in range(len(vecs) - 1)]
                        content_novelty = max(0.0, 1.0 - max(sims))
                except Exception:
                    pass

            score += 0.7 * addressee_factor * content_novelty
            break

    # エラー
    if "エラー" in tool_result:
        score *= 0.2

    # 計画変更
    old_plan = state_before.get("plan", {}).get("goal", "")
    new_plan = state_after.get("plan", {}).get("goal", "")
    if new_plan != old_plan and new_plan:
        score += 0.8

    # pending変化（対応して減った）
    old_pending = len(state_before.get("pending", []))
    new_pending = len(state_after.get("pending", []))
    if new_pending < old_pending:
        score += 0.5 * (old_pending - new_pending)

    return min(1.5, score)


def _calc_prediction(expect: str, result: str) -> float:
    """予測精度。expect vs result。0.0-1.0。"""
    if not expect or not result:
        return 0.5

    if _vector_ready:
        try:
            vecs = _embed_sync([expect[:500], result[:500]])
            if vecs and len(vecs) == 2:
                return max(0.0, cosine_similarity(vecs[0], vecs[1]))
        except Exception:
            pass

    # フォールバック: キーワード一致
    import re
    expect_tokens = set(re.findall(r'\w+', expect.lower()))
    result_tokens = set(re.findall(r'\w+', result.lower()))
    if not expect_tokens:
        return 0.5
    overlap = expect_tokens & result_tokens
    return len(overlap) / max(len(expect_tokens), 1)


def _calc_diversity(tool_names: list[str], log: list) -> float:
    """行動多様性。直近20件のツール使用分布のShannon entropy。0.0-1.0。"""
    recent_tools = [e.get("tool", "unknown") for e in log[-20:]]
    for tn in tool_names:
        recent_tools.append(tn)

    if len(recent_tools) < 2:
        return 1.0

    counts = Counter(recent_tools)
    if len(counts) <= 1:
        return 0.0  # 1種類のツールしか使ってない = 多様性ゼロ
    total = sum(counts.values())
    H = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_H = math.log2(len(counts))
    return H / max_H if max_H > 0 else 0.0


def _calc_coherence(state: dict) -> float:
    """自己モデルと直近行動の一致度。0.0-1.0。"""
    self_model = state.get("self", {})
    log = state.get("log", [])

    if not self_model or not log or not _vector_ready:
        return 0.5

    # self_modelのテキスト表現
    self_text = json.dumps(self_model, ensure_ascii=False)[:500]

    # 直近5件のintent+resultの連結
    recent_texts = []
    for entry in log[-5:]:
        parts = []
        if entry.get("intent"):
            parts.append(entry["intent"])
        if entry.get("result"):
            parts.append(str(entry["result"])[:200])
        if parts:
            recent_texts.append(" ".join(parts))

    if not recent_texts:
        return 0.5

    behavior_text = " ".join(recent_texts)[:500]

    try:
        vecs = _embed_sync([self_text, behavior_text])
        if vecs and len(vecs) == 2:
            return max(0.0, cosine_similarity(vecs[0], vecs[1]))
    except Exception:
        pass

    return 0.5


def update_energy(state: dict, eval_result: dict) -> float:
    """評価結果からenergy更新。achievementとdiversityの平均が50%超で上昇。"""
    achievement = eval_result.get("achievement", 0.5)
    diversity = eval_result.get("diversity", 0.5)
    prediction = eval_result.get("prediction", 0.5)
    avg = (achievement + diversity + prediction) / 3.0
    # avgを0-1にスケール（achievementは0-1.5なので正規化）
    normalized = min(1.0, avg)
    delta = normalized * 2.0 - 1.0  # 0.5が損益分岐
    state["energy"] = max(0, min(100, state.get("energy", 50) + delta))
    return delta
