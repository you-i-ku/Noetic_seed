"""E値計算（E4多様性・energy更新・LLM評価・state変化量・螺旋ベクトル・effective_change）"""
import re
import json
import math
from difflib import SequenceMatcher
from core.embedding import _vector_ready, _embed_sync, cosine_similarity

# 外界に不可逆な作用を及ぼすツール（output_display + SNSポスト系）
EXTERNAL_ACTION_TOOLS = {
    "output_display",
    "elyth_post", "elyth_reply", "elyth_like", "elyth_follow",
    "x_post", "x_reply", "x_quote", "x_like",
}


def _calc_e4(current_intent: str, current_result: str, recent_entries: list, n: int = 5) -> str:
    """現在の(intent+result)が直近n件と異なるほど高い（反復=低、新規性=高）。"""
    if not current_intent:
        return ""
    past_texts = []
    for e in recent_entries:
        if e.get("intent"):
            past_texts.append(f"{e['intent']} {str(e.get('result', ''))[:500]}")
    past_texts = past_texts[-n:]
    if not past_texts:
        return ""

    current_text = f"{current_intent} {current_result[:500]}"

    if _vector_ready:
        try:
            vecs = _embed_sync([current_text] + past_texts)
            if vecs and len(vecs) == 1 + len(past_texts):
                current_vec = vecs[0]
                sims = [cosine_similarity(current_vec, vecs[i + 1]) for i in range(len(past_texts))]
                avg_sim = sum(sims) / len(sims)
                return f"{round((1 - avg_sim) * 100)}%"
        except Exception:
            pass

    current_tokens = set(re.findall(r'\w+', current_text.lower()))
    if not current_tokens:
        return ""
    ratios = []
    for past in past_texts:
        past_tokens = set(re.findall(r'\w+', past.lower()))
        if past_tokens:
            overlap = current_tokens & past_tokens
            ratios.append(len(overlap) / max(len(current_tokens), len(past_tokens)))
    if not ratios:
        return ""
    avg = round((1 - sum(ratios) / len(ratios)) * 100)
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
    delta = e_mean / 50.0 - 1.0
    state["energy"] = max(0, min(100, state.get("energy", 50) + delta))
    return delta


def eval_with_llm(intent: str, expect: str, result: str, recent_intents: list, call_llm_fn) -> dict | None:
    """LLMにE1-E4を一括評価させる。失敗時はNone（ベクトル類似度にフォールバック）。"""
    recent_str = " / ".join(recent_intents[:3]) if recent_intents else "(none)"
    prompt = f"""以下の行動を評価してください。各項目を0-100の数値で採点してください。
説明や分析は不要です。必ず以下の形式のみで出力してください:

E1=数値
E2=数値
E3=数値
E4=数値

例:
E1=75
E2=80
E3=60
E4=90

評価対象:
意図: {intent[:500]}
予測: {expect[:500]}
結果: {result[:1000]}
直近の行動: {recent_str}

E1(意図と予測の整合性):
E2(意図の達成度):
E3(予測の正確さ):
E4(この行動の新規性):"""

    try:
        resp = call_llm_fn(prompt, max_tokens=24000, temperature=0.1)
        # デバッグログ
        from core.state import append_debug_log
        append_debug_log("LLM3 (E-value eval)", resp)
        scores = {}
        # パース: E1=75 or E1:75 or E1(...)=75 形式
        for line in resp.strip().splitlines():
            for key in ("E1", "E2", "E3", "E4"):
                if key in line:
                    m = re.search(r'(\d+)', line.split(key, 1)[1]) if key in line else None
                    if m:
                        scores[key.lower()] = int(m.group(1)) / 100.0
        # フォールバック: 数字だけの行を順にE1-E4に割り当て
        if len(scores) < 3:
            numbers = []
            for line in resp.strip().splitlines():
                line = line.strip()
                m = re.match(r'^(\d+)$', line)
                if m:
                    numbers.append(int(m.group(1)))
            if len(numbers) >= 3:
                for i, num in enumerate(numbers[:4]):
                    scores[["e1", "e2", "e3", "e4"][i]] = num / 100.0
        if len(scores) >= 3:
            return scores
        print(f"  [eval] LLM評価パース失敗: scores={scores} resp={resp[:100]}")
    except Exception as e:
        print(f"  [eval] LLM評価エラー: {e}")
    return None


def calc_state_change_bonus(state_before: dict, state_after: dict) -> float:
    """行動前後のstate差分からnegentropyボーナスを計算。"""
    changes = 0.0

    old_self = state_before.get("self", {})
    new_self = state_after.get("self", {})
    if len(new_self) > len(old_self):
        changes += 1.0  # キーが増えた（自己構造化）
    elif old_self != new_self:
        changes += 0.5  # 値が変わった（自己更新）

    old_fw = set(state_before.get("files_written", []))
    new_fw = set(state_after.get("files_written", []))
    if new_fw - old_fw:
        changes += 1.0  # 新しいファイルを書いた

    old_fr = set(state_before.get("files_read", []))
    new_fr = set(state_after.get("files_read", []))
    if new_fr - old_fr:
        changes += 0.5  # 新しいファイルを読んだ

    old_plan = state_before.get("plan", {}).get("goal", "")
    new_plan = state_after.get("plan", {}).get("goal", "")
    if new_plan != old_plan:
        changes += 1.0  # 計画が変わった

    return min(1.0, changes * 0.3)


def calc_effective_change(tool_names: list[str], tool_result: str,
                          state_before: dict, state_after: dict) -> float:
    """行動の実質的な情報変化量を測定する。
    変化ゼロの行動（同じkeyに同じようなvalue書き込み等）を正しくゼロ評価する。
    戻り値: 0.0（変化なし）〜 1.5（大きな変化）"""
    score = 0.0

    # --- self model の変化量（value差分で測定）---
    old_self = state_before.get("self", {})
    new_self = state_after.get("self", {})
    for key in new_self:
        if key == "name":
            continue
        if key not in old_self:
            score += 1.0  # 新しいkey = 新しい認識
        elif str(new_self[key]) != str(old_self[key]):
            old_v, new_v = str(old_self[key]), str(new_self[key])
            dist = 1.0 - SequenceMatcher(None, old_v, new_v).ratio()
            score += dist * 0.5  # 既存keyの更新: 差分が大きいほど高い

    # --- ファイル操作 ---
    old_fw = set(state_before.get("files_written", []))
    new_fw = set(state_after.get("files_written", []))
    if new_fw - old_fw:
        score += 0.8  # 新規ファイル作成

    old_fr = set(state_before.get("files_read", []))
    new_fr = set(state_after.get("files_read", []))
    if new_fr - old_fr:
        score += 0.5  # 新しいファイルを読んだ（情報獲得）
    elif any(n == "read_file" for n in tool_names):
        score += 0.1  # 既読再読

    # --- 外界への不可逆な作用 ---
    for tn in tool_names:
        if tn in EXTERNAL_ACTION_TOOLS:
            score += 0.7
            break  # 1回分のみ

    # --- エラー（変化なし）---
    if "エラー" in tool_result:
        score *= 0.2  # エラーは変化量を大幅減

    # --- 計画変更 ---
    old_plan = state_before.get("plan", {}).get("goal", "")
    new_plan = state_after.get("plan", {}).get("goal", "")
    if new_plan != old_plan and new_plan:
        score += 0.8

    return min(1.5, score)


def apply_effective_change_to_e2(e2_raw: float, effective_change: float) -> float:
    """effective_changeでE2を変調する。
    変化ゼロ→E2上限30%、変化大→E2そのまま。"""
    change_factor = min(1.0, 0.3 + effective_change * 0.7)
    return e2_raw * change_factor


def calc_spiral_vector(state: dict, log: list, k: int = 20) -> dict:
    """螺旋の上昇ベクトルを計算。
    magnitude: 変化してるか（大きさ）
    consistency: 方向が一貫してるか（-1.0〜1.0）"""
    magnitude = 0.0
    consistency = 0.0

    if len(log) < k * 2 or not _vector_ready:
        return {"magnitude": magnitude, "consistency": consistency}

    # --- magnitude: k期間前と今の差 ---
    old_entries = log[-k*2:-k]
    new_entries = log[-k:]

    old_text = " ".join(e.get("intent", "") for e in old_entries)
    new_text = " ".join(e.get("intent", "") for e in new_entries)
    # self_modelの現在の内容も加味
    self_text = json.dumps(state.get("self", {}), ensure_ascii=False)[:500]
    new_text += " " + self_text

    try:
        vecs = _embed_sync([old_text[:2000], new_text[:2000]])
        if vecs and len(vecs) == 2:
            magnitude = max(0.0, 1 - cosine_similarity(vecs[0], vecs[1]))
    except Exception:
        pass

    # --- consistency: 3期間の方向一貫性 ---
    if len(log) >= k * 3:
        period_a = " ".join(e.get("intent", "") for e in log[-k*3:-k*2])
        period_b = " ".join(e.get("intent", "") for e in log[-k*2:-k])
        period_c = " ".join(e.get("intent", "") for e in log[-k:])

        try:
            vecs3 = _embed_sync([period_a[:2000], period_b[:2000], period_c[:2000]])
            if vecs3 and len(vecs3) == 3:
                delta1 = [b - a for a, b in zip(vecs3[0], vecs3[1])]
                delta2 = [c - b for b, c in zip(vecs3[1], vecs3[2])]
                consistency = cosine_similarity(delta1, delta2)
        except Exception:
            pass

    return {"magnitude": magnitude, "consistency": consistency}


def calc_measured_entropy(state: dict, log: list) -> float:
    """AIの実測エントロピー。4指標の均等平均。0.0（完全秩序）〜1.0（完全ノイズ）。"""
    from collections import Counter

    # 1. behavioral_entropy: ツール使用分布（直近20件）
    recent_tools = [e.get("tool", "unknown") for e in log[-20:]]
    if len(recent_tools) >= 2:
        counts = Counter(recent_tools)
        total = sum(counts.values())
        H = -sum((c/total) * math.log2(c/total) for c in counts.values())
        max_H = math.log2(len(counts)) if len(counts) > 1 else 1.0
        behavioral = H / max_H if max_H > 0 else 0.0
    else:
        behavioral = 1.0  # データ不足→高エントロピー

    # 2. intent_diversity: intent埋め込みの非類似度（直近10件）
    recent_intents = [e.get("intent", "") for e in log[-10:] if e.get("intent")]
    if len(recent_intents) >= 2 and _vector_ready:
        try:
            vecs = _embed_sync(recent_intents)
            if vecs and len(vecs) == len(recent_intents):
                sims = []
                for i in range(len(vecs)):
                    for j in range(i+1, len(vecs)):
                        sims.append(cosine_similarity(vecs[i], vecs[j]))
                intent_div = 1 - (sum(sims) / len(sims)) if sims else 1.0
            else:
                intent_div = 0.5
        except Exception:
            intent_div = 0.5
    else:
        intent_div = 0.5

    # 3. state_richness: self_modelの充実度（キー数 / 8基準）
    self_keys = len([k for k in state.get("self", {}) if k != "name"])
    state_rich = min(1.0, self_keys / 8.0)

    # 4. sandbox_richness: sandbox介入度（ファイル数 / 5基準）
    sandbox_count = len(state.get("files_written", []))
    sandbox_rich = min(1.0, sandbox_count / 5.0)

    # 均等平均（state/sandboxは反転: 充実→低エントロピー）
    measured = (behavioral + intent_div + (1 - state_rich) + (1 - sandbox_rich)) / 4.0
    return max(0.0, min(1.0, measured))
