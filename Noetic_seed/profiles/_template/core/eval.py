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
                          current_intent: str = "",
                          target_id: str = "") -> float:
    """行動の実質的な情報変化量を測定する。
    変化ゼロの行動（同じkeyに同じようなvalue書き込み等）を正しくゼロ評価する。
    current_intent: 今回の行動の intent。
    target_id: reply_to_id 等の確定的対象ID（変更7: 同一対象への繰り返し検出用）。
    戻り値: 0.0（変化なし）〜 1.5（大きな変化）"""
    score = 0.0

    # --- 5-c: self model の変化量（意味的重複チェック強化）---
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
            if dist >= 0.15:
                # 5-c: 文字列 diff は大きいが意味的に同じ（言い換え）を弾く
                if _vector_ready:
                    try:
                        vecs = _embed_sync([old_v[:200], new_v[:200]])
                        if vecs and len(vecs) == 2:
                            sem_sim = cosine_similarity(vecs[0], vecs[1])
                            if sem_sim > 0.8:
                                score += 0.05  # 言い換えのみ
                                continue
                    except Exception:
                        pass
                score += dist * 0.5

    # --- ファイル操作 ---
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

    # --- 5-b: pending 件数 diff ---
    old_pending = state_before.get("pending_count", len(state_before.get("pending", [])))
    new_pending = len(state_after.get("pending", []))
    if new_pending < old_pending:
        score += 0.3 * (old_pending - new_pending)

    # --- 行動の意味的新規性（外界作用 or 内省の繰り返し検出）---
    log = state_after.get("log", [])
    for tn in tool_names:
        if tn not in ACTIONABLE_TOOLS:
            continue

        if tn in EXTERNAL_ACTION_TOOLS:
            pending = state_after.get("pending", [])
            # UPS v2: source_action="living_presence" + channel="device" で検出
            has_addressee = (
                any(
                    p.get("type") == "pending"
                    and p.get("source_action") == "living_presence"
                    and (p.get("observed_channel") == "device"
                         or p.get("expected_channel") == "device")
                    for p in pending
                )
                or state_after.get("unresponded_external_count", 0) > 0
            )
            addressee_factor = 1.0 if has_addressee else 0.15
        else:
            addressee_factor = 1.0

        # 変更7: target_id があれば同一対象のエントリだけと比較
        recent_texts = []
        for e in log[-30:]:
            past_chain = str(e.get("tool", "")).split("+")
            if tn in past_chain and e.get("intent"):
                # target_id 指定時: 同一対象のみ
                if target_id:
                    if target_id not in str(e.get("result", "")):
                        continue
                combined = f"{str(e['intent'])[:300]} {str(e.get('result', ''))[:200]}"
                recent_texts.append(combined[:500])

        content_novelty = 1.0
        if recent_texts and current_intent and _vector_ready:
            try:
                current_text = f"{current_intent[:300]} {tool_result[:200]}"[:500]
                texts = [current_text] + recent_texts[-5:]
                vecs = _embed_sync(texts)
                if vecs and len(vecs) >= 2:
                    sims = [cosine_similarity(vecs[0], vecs[i + 1])
                            for i in range(len(vecs) - 1)]
                    max_sim = max(sims)
                    content_novelty = max(0.0, 1.0 - max_sim ** 0.5)
            except Exception:
                pass

        score += 0.7 * addressee_factor * content_novelty
        break

    # --- 5-a: 記憶新規性チェック（reflect/memory_store の同内容を弾く）---
    if any(n in ("memory_store", "reflect") for n in tool_names):
        try:
            from core.memory import memory_network_search
            existing = memory_network_search(tool_result[:200],
                                             networks=["opinion", "experience", "entity"],
                                             limit=3)
            # 5-a デバッグログ
            try:
                from core.config import RESOLUTION_LOG
                with open(RESOLUTION_LOG, "a", encoding="utf-8") as _f:
                    if existing:
                        _sims = [f"{e.get('score',0):.2f}" for e in existing[:3]]
                        _f.write(f"  [5-a] {'+'.join(tool_names)} existing_sims=[{','.join(_sims)}] "
                                 f"result_snippet={tool_result[:60]}\n")
                    else:
                        _f.write(f"  [5-a] {'+'.join(tool_names)} no_existing_memories\n")
            except Exception:
                pass
            if existing:
                max_existing_sim = max(e.get("score", 0) for e in existing)
                if max_existing_sim > 0.7:
                    score *= max(0.1, 1.0 - max_existing_sim)
        except Exception:
            pass

    # --- 5-d: 対エンティティ行動は仮 ec（送信成功だけでは世界は変わってない）---
    _ASYNC_TOOLS = {"output_display", "elyth_post", "elyth_reply", "x_post", "x_reply", "x_quote"}
    if any(n in _ASYNC_TOOLS for n in tool_names):
        score = min(score, 0.15)  # 上限0.15。遅延フィードバックで後から上がる

    # --- エラー（変化なし）---
    if "エラー" in tool_result:
        score *= 0.2

    return min(1.5, score)


def update_gaps_by_relevance(state: dict, result_str: str, ec: float):
    """変更6: 行動結果が既存 UPS v2 semantic_merge 系 pending に関連していれば
    gap を下げる。別の方法で課題に取り組んでも gap が下がる（同じ方法の再試行に
    限らない）。
    """
    pending = state.get("pending", [])
    unresolved = [
        p for p in pending
        if p.get("type") == "pending" and p.get("semantic_merge") is True
    ]
    if not unresolved or not _vector_ready or not result_str:
        return
    try:
        from core.pending_unified import calc_priority as _ups_priority
        texts = [result_str[:200]] + [u.get("content", "")[:200] for u in unresolved]
        vecs = _embed_sync(texts)
        if not vecs or len(vecs) != len(texts):
            return
        for i, u in enumerate(unresolved):
            sim = cosine_similarity(vecs[0], vecs[i + 1])
            if sim > 0.5:
                decay = 1.0 - ec * 0.5 * sim
                u["gap"] = u["gap"] * max(0.1, decay)
                u["priority"] = _ups_priority(u)
    except Exception:
        pass


def update_unresolved_intents(
    state: dict, intent: str, e3_str: str, cycle_id: int,
    source_action: str = "reflection", lag_kind: str = "cycles",
) -> None:
    """E3 から予測誤差 gap=1-E3 を計算し、UPS v2 pending に追加。

    UPS v2 対応 (Phase 4 Step C-2): 出力を type="pending" /
    semantic_merge=True / expected_channel="self" に統一。
    内部ロジック (rate-distortion 容量管理、semantic merge) は不変。

    Args:
        source_action: どの tool/action 由来の unresolved か。
            PostToolUse hook では直前の tool_name を渡す。
            reflection サイクル内では "reflection" (default)。
        lag_kind: observation_lag_kind。unresolved 系は通常 "cycles"
            (時間スケールが cycle 単位で広がる)。

    rate-distortion: 閾値なし、動的容量 N で上位 gap 順に自動選別。
    semantic merge: 既存 UPS v2 pending (semantic_merge=True) と類似度
    > 0.72 で attempts 加算マージ。N = max(3, min(20, len(log)//5))。
    """
    from datetime import datetime
    from core.pending_unified import calc_priority as _ups_priority
    from core.pending_unified import pending_add as _ups_pending_add

    if not intent or not e3_str:
        return

    m = re.search(r'(\d+)', str(e3_str))
    if not m:
        return
    e3_val = int(m.group(1)) / 100.0
    gap = max(0.0, 1.0 - e3_val)

    pending = state.setdefault("pending", [])
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # semantic merge 対象: UPS v2 で semantic_merge=True な pending
    merged = False
    candidates = [
        p for p in pending
        if p.get("type") == "pending" and p.get("semantic_merge") is True
    ]
    if candidates and _vector_ready:
        try:
            texts = [intent[:200]] + [c.get("content", "")[:200] for c in candidates]
            vecs = _embed_sync(texts)
            if vecs and len(vecs) == len(texts):
                for i, c in enumerate(candidates):
                    sim = cosine_similarity(vecs[0], vecs[i + 1])
                    if sim > 0.72:
                        c["attempts"] = c.get("attempts", 1) + 1
                        c["last_cycle"] = cycle_id
                        c["gap"] = gap  # 最新 gap で上書き (回復→低→自然に選別外)
                        c["timestamp"] = now_ts
                        c["priority"] = _ups_priority(c)
                        merged = True
                        break
        except Exception:
            pass

    if not merged:
        _ups_pending_add(
            state=state,
            source_action=source_action,
            expected_observation=intent[:200],
            lag_kind=lag_kind,
            content=intent[:200],
            cycle_id=cycle_id,
            channel="self",  # unresolved 系は内省由来
            expiry_policy="dynamic_n",
            initial_gap=gap,
            semantic_merge=True,
        )

    # 動的容量管理: UPS v2 semantic_merge=True 系のみ対象、gap 上位 N 保持
    log_count = len(state.get("log", []))
    n_cap = max(3, min(20, log_count // 5))
    merge_entries = [
        p for p in state["pending"]
        if p.get("type") == "pending" and p.get("semantic_merge") is True
    ]
    merge_entries.sort(key=lambda p: -p.get("gap", 0.0))
    keep_ids = {c["id"] for c in merge_entries[:n_cap]}
    state["pending"] = [
        p for p in state["pending"]
        if not (p.get("type") == "pending" and p.get("semantic_merge") is True)
        or p.get("id") in keep_ids
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

    if not _vector_ready or not intent:
        return 1.0

    # layer 2: 同ツールの past results とベクトル類似度
    layer2_novelty = 1.0
    same_tool = [e for e in ledger if e.get("tool") == tool_name]
    if same_tool:
        try:
            recent = same_tool[-8:]
            texts = [f"{tool_name}: {intent[:180]}"] + [e["result_snippet"][:200] for e in recent]
            vecs = _embed_sync(texts)
            if vecs and len(vecs) >= 2:
                sims = [cosine_similarity(vecs[0], vecs[i+1]) for i in range(len(vecs)-1)]
                max_sim = max(sims)
                if max_sim > 0.8:
                    layer2_novelty = 0.05
                elif max_sim > 0.6:
                    layer2_novelty = max(0.1, 1.0 - max_sim)
        except Exception:
            pass

    # layer 3: トピック飽和（全ツール横断、エントリ一件ずつ MIN）
    # 各 ledger エントリの intent と比較。sim と ec はAI自身の過去実績。
    # 高 sim（同じことを考えてる）時は ec に関係なく novelty を下げる。
    layer3_novelty = 1.0
    recent_all = ledger[-15:]
    if recent_all:
        try:
            # intent 同士を純粋に比較（tool_name プレフィックスなし）
            texts = [intent[:200]] + [e["intent"][:200] for e in recent_all]
            vecs = _embed_sync(texts)
            if vecs and len(vecs) >= 2:
                for i in range(len(vecs) - 1):
                    sim = cosine_similarity(vecs[0], vecs[i + 1])
                    past_ec = recent_all[i].get("ec", 0.5)
                    if sim > 0.8:
                        # 高類似: ほぼ同じ intent → ec に関係なく抑制
                        entry_novelty = max(0.05, 1.0 - sim)
                    else:
                        # 中〜低類似: ec が高かった行動は繰り返す価値がありうる
                        entry_novelty = 1.0 - sim * (1.0 - past_ec)
                    layer3_novelty = min(layer3_novelty, entry_novelty)
        except Exception:
            pass

    return min(layer2_novelty, layer3_novelty)
