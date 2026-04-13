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

# 内省系ツール（外界作用はないが繰り返し判定の対象にする）
INTERNAL_REFLECT_TOOLS = {"reflect"}

# 意味的重複ペナルティをかけるツール全体
ACTIONABLE_TOOLS = EXTERNAL_ACTION_TOOLS | INTERNAL_REFLECT_TOOLS


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

    return min(1.0, changes * 0.3)


def calc_effective_change(tool_names: list[str], tool_result: str,
                          state_before: dict, state_after: dict,
                          current_intent: str = "") -> float:
    """行動の実質的な情報変化量を測定する。
    変化ゼロの行動（同じkeyに同じようなvalue書き込み等）を正しくゼロ評価する。
    current_intent: 今回の行動の intent。過去の同ツール呼び出し intent との
    embedding 類似度で「意味的な繰り返し」を検出するのに使う。
    戻り値: 0.0（変化なし）〜 1.5（大きな変化）"""
    score = 0.0

    # --- self model の変化量（value差分で測定、微小更新は無視）---
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
            # 15%未満の微小差分はカウントしない（reflectのdisposition_delta等を弾く）
            if dist >= 0.15:
                score += dist * 0.5

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

    # --- 行動の意味的新規性（外界作用 or 内省の繰り返し検出）---
    # intent + result を過去の同ツール呼び出しと embedding 類似度で比較
    # 改良: window を広げ、result snippet も比較対象に含め、非線形で中程度類似もペナルティ化
    log = state_after.get("log", [])
    for tn in tool_names:
        if tn not in ACTIONABLE_TOOLS:
            continue

        # ① 相手がいるか（外界作用のみ適用、内省は常に 1.0）
        if tn in EXTERNAL_ACTION_TOOLS:
            pending = state_after.get("pending", [])
            has_addressee = (
                any(p.get("type") == "external_message" for p in pending) or
                state_after.get("unresponded_external_count", 0) > 0
            )
            addressee_factor = 1.0 if has_addressee else 0.15
        else:
            addressee_factor = 1.0

        # ② 意図の新規性: 過去30件から同ツールを含むチェーンの intent + result を抽出
        #    reflect のように intent が毎回違う表現でも、OPINIONS (result) が似ていれば
        #    繰り返しとして検出される
        recent_texts = []
        for e in log[-30:]:
            past_chain = str(e.get("tool", "")).split("+")
            if tn in past_chain and e.get("intent"):
                # intent + result snippet で reflect の OPINIONS も比較対象に含める
                combined = f"{str(e['intent'])[:300]} {str(e.get('result', ''))[:200]}"
                recent_texts.append(combined[:500])

        content_novelty = 1.0
        if recent_texts and current_intent and _vector_ready:
            try:
                # 現在の行動: 今の intent + 実行結果（reflect なら OPINIONS が tool_result に入ってる）
                current_text = f"{current_intent[:300]} {tool_result[:200]}"[:500]
                texts = [current_text] + recent_texts[-5:]
                vecs = _embed_sync(texts)
                if vecs and len(vecs) >= 2:
                    sims = [cosine_similarity(vecs[0], vecs[i + 1])
                            for i in range(len(vecs) - 1)]
                    max_sim = max(sims)
                    # 非線形: sqrt で中程度類似（sim=0.5 等）でも確実にペナルティが効く
                    # sim=0.9 → novelty=0.05 (強抑制)
                    # sim=0.7 → novelty=0.16 (中抑制)
                    # sim=0.5 → novelty=0.29 (弱抑制)
                    # sim=0.3 → novelty=0.45 (僅か抑制)
                    content_novelty = max(0.0, 1.0 - max_sim ** 0.5)
            except Exception:
                pass

        score += 0.7 * addressee_factor * content_novelty
        break  # 複数の actionable がチェーンにあっても1回だけ加算

    # --- エラー（変化なし）---
    if "エラー" in tool_result:
        score *= 0.2

    return min(1.5, score)


def update_unresolved_intents(state: dict, intent: str, e3_str: str, cycle_id: int) -> None:
    """E3から予測誤差 gap=1-E3 を計算し、unresolved_intent として pending に追加。
    rate-distortion 的: 閾値なし、動的容量 N で上位gap順に自動選別。
    semantic merge: 既存 unresolved と類似度 > 0.72 なら attempts を加算してマージ。
    N = max(3, min(20, len(log)//5)) で log 成長に応じて枠が広がる。"""
    import time
    from datetime import datetime

    if not intent or not e3_str:
        return

    m = re.search(r'(\d+)', str(e3_str))
    if not m:
        return
    e3_val = int(m.group(1)) / 100.0
    gap = max(0.0, 1.0 - e3_val)

    pending = state.setdefault("pending", [])
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 既存 unresolved_intent との意味的マージ
    merged = False
    unresolved = [p for p in pending if p.get("type") == "unresolved_intent"]
    if unresolved and _vector_ready:
        try:
            texts = [intent[:200]] + [u.get("content", "")[:200] for u in unresolved]
            vecs = _embed_sync(texts)
            if vecs and len(vecs) == len(texts):
                for i, u in enumerate(unresolved):
                    sim = cosine_similarity(vecs[0], vecs[i + 1])
                    if sim > 0.72:
                        u["attempts"] = u.get("attempts", 1) + 1
                        u["last_cycle"] = cycle_id
                        u["gap"] = gap  # 最新の gap で上書き（回復したら低くなる→自然に選別外）
                        u["timestamp"] = now_ts
                        u["priority"] = gap * 3.0
                        merged = True
                        break
        except Exception:
            pass

    if not merged:
        pending.append({
            "type": "unresolved_intent",
            "id": f"uri_{cycle_id:04d}_{int(time.time() * 1000) % 10000}",
            "content": intent[:200],
            "gap": gap,
            "attempts": 1,
            "origin_cycle": cycle_id,
            "last_cycle": cycle_id,
            "timestamp": now_ts,
            "priority": gap * 3.0,
        })

    # 動的容量: log 成長で枠が広がる。gap 上位 N のみ保持
    log_count = len(state.get("log", []))
    n_cap = max(3, min(20, log_count // 5))
    unresolved_now = [p for p in pending if p.get("type") == "unresolved_intent"]
    unresolved_now.sort(key=lambda p: -p.get("gap", 0.0))
    keep_ids = {u["id"] for u in unresolved_now[:n_cap]}
    state["pending"] = [
        p for p in pending
        if p.get("type") != "unresolved_intent" or p.get("id") in keep_ids
    ]


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


# === Action Ledger: 行動 + result の永続記録 ===

_LEDGER_MAX = 50


def _extract_action_key(tool_name: str, targs: dict) -> str:
    """確定的キー。対象IDがあるツールのみ。"""
    for key in ("reply_to_id", "tweet_url", "post_id", "query", "path", "file", "name", "url"):
        val = targs.get(key, "")
        if val:
            return f"{tool_name}:{str(val)[:80]}"
    return ""


def append_action_ledger(state: dict, tool_name: str, action_key: str,
                         intent: str, result: str, ec: float, cycle_id: int):
    """行動台帳に追記。result の先頭300字も保存（事前予測の材料）。"""
    from datetime import datetime
    ledger = state.setdefault("action_ledger", [])
    ledger.append({
        "tool": tool_name,
        "action_key": action_key,
        "intent": intent[:200],
        "result_snippet": result[:300],
        "ec": round(ec, 4),
        "cycle": cycle_id,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    if len(ledger) > _LEDGER_MAX:
        state["action_ledger"] = ledger[-_LEDGER_MAX:]
    # デバッグログ
    try:
        from core.config import RESOLUTION_LOG
        with open(RESOLUTION_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{ledger[-1]['ts']}] cycle={cycle_id} "
                    f"tool={tool_name} key={action_key or '(none)'} "
                    f"ec={ec:.3f} intent={intent[:60]}\n")
    except Exception:
        pass


def predict_result_novelty(state: dict, tool_name: str, intent: str,
                           action_key: str = "") -> float:
    """候補行動の結果新規性を事前予測する（報酬予測誤差）。
    action_ledger の過去 result_snippet と現在の intent を比較。
    過去に似た行動で似た結果が出ていたら novelty は低い。
    戻り値: 0.0（同じ結果になる）〜 1.0（未知の結果が期待できる）"""
    ledger = state.get("action_ledger", [])
    if not ledger:
        return 1.0  # 履歴なし = 初めて = 高い新規性

    # layer 1: action_key 完全一致（同じ reply_to_id 等）
    if action_key:
        matches = [e for e in ledger if e.get("action_key") == action_key]
        if matches:
            past_results = [e["result_snippet"] for e in matches[-3:]]
            if past_results and _vector_ready:
                try:
                    vecs = _embed_sync([intent[:200]] + [r[:200] for r in past_results])
                    if vecs and len(vecs) >= 2:
                        sims = [cosine_similarity(vecs[0], vecs[i+1])
                                for i in range(len(vecs)-1)]
                        return max(0.0, 1.0 - max(sims))
                except Exception:
                    pass
            return 0.2  # embedding 失敗でも action_key 一致 = 低 novelty

    # layer 2: 同ツールの past results とベクトル類似度
    if not _vector_ready or not intent:
        return 1.0

    same_tool = [e for e in ledger if e.get("tool") == tool_name]
    if not same_tool:
        return 1.0  # このツール初使用

    try:
        recent = same_tool[-8:]
        texts = [f"{tool_name}: {intent[:180]}"] + [e["result_snippet"][:200] for e in recent]
        vecs = _embed_sync(texts)
        if not vecs or len(vecs) < 2:
            return 1.0
        sims = [cosine_similarity(vecs[0], vecs[i+1]) for i in range(len(vecs)-1)]
        max_sim = max(sims)
        if max_sim > 0.8:
            return 0.05
        elif max_sim > 0.6:
            return max(0.1, 1.0 - max_sim)
        return 1.0
    except Exception:
        return 1.0
