"""Slice 5 (orchestration §5 Slice 5): Goal Shadow Observer.

暗黙 goal を固定せず影として観測する薄い層。controller への影響なし、
LLM judge 経由なし、構造的観察のみ。Slice 6 で resonance multiplier に弱接続予定
(本 Slice では interface のみ、配線なし)。

設計確定事項 (2026-05-10 ゆう判断 + Codex 5 周目 review):
  ① label 同定: bge-m3 embedding (L2 normalized) cosine ≥ LABEL_SAME_THRESHOLD
     embedding 不在 (bge-m3 未初期化 / pending text 未 embed) 時は label 完全一致 fallback
  ② centroid update: EMA α=CENTROID_EMA_ALPHA、毎回 L2 normalize
     (Codex 追加指摘 #2: bge-m3 規約 = L2 normalized、EMA で magnitude drift する罠を回避)
  ③ EvidenceRef schema: {source: str, id: str, cycle_id: int}
     (role は YAGNI で削除、Slice 6+ で必要なら追加)
     raw_events と subjective_entries は同 id keyspace を共有する Noetic 固有設計
     (state.merge_log_view) のため、source field で区別する。
  ④ realized 判定: cycle 内 files_written delta ≥ 1 + persistence ≥ 1
     (Codex 追加指摘 #1: snapshot 更新は update 後 / emit 前、caller 責務)
  ⑤ status 遷移 (smoke 後 tune 前提):
     latent → active:    activation ≥ ACTIVATION_TO_ACTIVE
     active → cooling:   cid - last_seen_cycle > COOLING_IDLE_CYCLES
     cooling → realized: files_delta ≥ 1 かつ persistence ≥ REALIZED_PERSISTENCE_MIN
     cooling → abandoned: cid - last_seen_cycle > ABANDON_IDLE_CYCLES
  ⑥ controller 影響なし (Slice 6 範囲外)

正典 pointer:
  WORLD_MODEL_DESIGN/NOETIC_INTEGRATED_ORCHESTRATION_PLAN.md §3 P2 #7 + §5 Slice 5
  WORLD_MODEL_DESIGN/NOETIC_AUTONOMY_REINFORCEMENT_PLAN.md §5 + §9 Phase C

たとえ:
  goal_shadow = 霧の中にうっすら浮かぶ足跡。命令ではなく、Noetic が
  どこに引かれているかの輪郭を観察するだけ。realized / abandoned は
  足跡が地面に固定された / 風で消えた状態。
"""
from __future__ import annotations

import uuid
from typing import Callable, Optional

# ============================================================
# 定数 (smoke 後 BP-3 観察で tune)
# ============================================================

LABEL_SAME_THRESHOLD = 0.85
"""embedding cosine 同定閾値。entity_resolver.EMBEDDING_SAME_THRESHOLD と統一。"""

CENTROID_EMA_ALPHA = 0.7
"""EMA で過去 centroid を保つ重み。Phase 6 link strength EMA と同水準。"""

ACTIVATION_TO_ACTIVE = 3.0
"""latent → active 閾値 (累積 activation)。"""

COOLING_IDLE_CYCLES = 5
"""active → cooling 閾値 (cid - last_seen_cycle で連続未言及 cycle 数)。"""

ABANDON_IDLE_CYCLES = 15
"""cooling → abandoned 閾値 (連続未言及 cycle 数)。"""

REALIZED_PERSISTENCE_MIN = 1.0
"""cooling → realized で必要な persistence 累積。"""

DECAY_RATE = 0.95
"""未言及 cycle で activation を緩く落とす rate (Physarum decay と同精神)。"""

LABEL_MAX_LEN = 80
"""GoalShadow.label の最大文字数 (signal text 先頭からの切出長)。"""


# evidence source vocabulary (open string、初期セット、§6 trajectory 等で拡張可)
SOURCE_SUBJECTIVE_ENTRY = "subjective_entry"
SOURCE_PENDING = "pending"
SOURCE_FILE_WRITE = "file_write"
SOURCE_RAW_EVENT = "raw_event"


# ============================================================
# Schema helpers
# ============================================================

def make_evidence_ref(source: str, id_: str, cycle_id: int) -> dict:
    """EvidenceRef = {source, id, cycle_id} を作る単一経路 helper。

    raw_events と subjective_entries が同 id keyspace を共有する Noetic 固有設計
    (state.merge_log_view) のため、source field で区別する。
    """
    return {"source": str(source), "id": str(id_), "cycle_id": int(cycle_id)}


def _new_goal_shadow(
    label: str,
    origin: str,
    cycle_id: int,
    embedding: Optional[list] = None,
) -> dict:
    """新規 GoalShadow dict (orchestration §5 schema literal)。"""
    return {
        "id": f"goal_{uuid.uuid4().hex[:12]}",
        "label": label,
        "status": "latent",
        "origin": origin,
        "evidence_refs": [],
        "embedding": list(embedding) if embedding else [],
        "activation": 0.0,
        "persistence": 0.0,
        "competence": 0.0,
        "learning_progress": 0.0,
        "interestingness": 0.0,
        "risk": 0.0,
        "last_seen_cycle": int(cycle_id),
        "created_cycle": int(cycle_id),
    }


# ============================================================
# Centroid EMA (Codex 追加指摘 #2: L2 normalize after EMA)
# ============================================================

def _l2_normalize(vec: list) -> list:
    """L2 norm で正規化、ゼロベクトルはそのまま返す (defensive)。"""
    if not vec:
        return list(vec)
    norm_sq = sum(float(x) * float(x) for x in vec)
    if norm_sq < 1e-18:
        return list(vec)
    norm = norm_sq ** 0.5
    return [float(x) / norm for x in vec]


def update_centroid(
    old_centroid: list,
    evidence_emb: list,
    alpha: float = CENTROID_EMA_ALPHA,
) -> list:
    """EMA で centroid を更新し、L2 normalize する。

    Codex 追加指摘 #2 (2026-05-10): EMA 後に normalize しないと、bge-m3 embedding の
    magnitude が浮動 round-off + α 重み積で drift する。bge-m3 規約は L2 normalized
    なので EMA 直後に毎回 normalize で空間規約に整合させる (cosine 0.85 閾値の安定性)。
    """
    if not old_centroid:
        return _l2_normalize(list(evidence_emb))
    if not evidence_emb:
        return list(old_centroid)
    if len(old_centroid) != len(evidence_emb):
        return list(old_centroid)
    new = [
        alpha * float(o) + (1.0 - alpha) * float(e)
        for o, e in zip(old_centroid, evidence_emb)
    ]
    return _l2_normalize(new)


# ============================================================
# Identification (cosine 0.85、embedding 不在時 label fallback)
# ============================================================

def find_matching_shadow(
    shadows: list,
    label: str,
    evidence_emb: Optional[list],
    cosine_fn: Optional[Callable] = None,
) -> Optional[dict]:
    """既存 shadows から label / centroid 類似のものを返す。なければ None。

    embedding 経路 (cosine_fn + evidence_emb 両方あり) が優先。複数候補がある場合
    cosine 最大かつ閾値以上の 1 件を選ぶ。embedding 経路が active な時は cosine
    不足 (全 < LABEL_SAME_THRESHOLD) で None を返す = label string fallback には
    流さない (Codex 1 周目 P2 fix、2026-05-10): 「label fallback は embedding
    完全不在時のみ」の docstring contract literal を実装で担保。同 label でも
    semantic 距離が遠い goal が誤って merge される silent bug を防ぐ。

    embedding 不在時 (cosine_fn None or evidence_emb 空、bge-m3 未初期化 / pending
    text 未 embed 等) のみ label string 完全一致で fallback 同定する (graceful
    degradation 経路)。
    """
    has_embedding_path = cosine_fn is not None and bool(evidence_emb)
    if has_embedding_path:
        best: Optional[dict] = None
        best_sim: float = -1.0
        # Codex 2 周目 P2 fix (2026-05-10): empty centroid shadow の upgrade 経路。
        # is_vector_ready=False 時に生成された shadow は embedding=[] (空) なので
        # cosine ループでスキップされる。後で vector_ready=True になり同 label の
        # signal が embedding 経路に入った時、 cosine match なしで None を返すと
        # caller が新規 shadow 生成して重複発生 → metrics 歪む。empty centroid +
        # label 一致を upgrade 候補として追跡し、cosine match なし時 fallback で
        # 返す (caller の update_centroid が old=[] → 新 emb の normalized で
        # 自動 upgrade される、外側変更不要)。
        empty_centroid_label_match: Optional[dict] = None
        for s in shadows:
            centroid = s.get("embedding")
            if not centroid:
                # 空 centroid + label 一致 → upgrade 候補に記録 (1 件のみ)
                if (
                    label
                    and s.get("label") == label
                    and empty_centroid_label_match is None
                ):
                    empty_centroid_label_match = s
                continue
            if len(centroid) != len(evidence_emb):
                continue
            try:
                sim = float(cosine_fn(centroid, evidence_emb))
            except Exception:
                continue
            if sim >= LABEL_SAME_THRESHOLD and sim > best_sim:
                best, best_sim = s, sim
        if best is not None:
            return best
        # cosine match なし → empty centroid + label 一致なら upgrade 候補を返す。
        # それも無ければ None (1 周目 P2 fix: label-only fallback には流さない、
        # semantic 距離の遠い goal が同 label で誤 merge される silent bug 防止)。
        return empty_centroid_label_match
    # embedding 不在時のみ label string 完全一致 fallback
    if label:
        for s in shadows:
            if s.get("label") == label:
                return s
    return None


# ============================================================
# Status transitions
# ============================================================

def _apply_status_transition(shadow: dict, cycle_id: int, files_delta: int) -> None:
    """1 つの shadow に対して status 遷移を適用 (in-place)。

    遷移マトリクス (§5 literal、smoke 後 BP-3 で tune):
      latent → active:    activation ≥ ACTIVATION_TO_ACTIVE
      active → cooling:   cid - last_seen_cycle > COOLING_IDLE_CYCLES
      cooling → realized: files_delta ≥ 1 かつ persistence ≥ REALIZED_PERSISTENCE_MIN
                          (Codex 追加指摘 #3: 当該 cycle 内 files_delta、過去累積で誤判定回避)
      cooling → abandoned: cid - last_seen_cycle > ABANDON_IDLE_CYCLES
      realized / abandoned: 終端 (本 Slice では再活性化なし、Slice 6+ で要検討)
    """
    status = shadow.get("status", "latent")
    activation = float(shadow.get("activation", 0.0))
    persistence = float(shadow.get("persistence", 0.0))
    last_seen = int(shadow.get("last_seen_cycle", cycle_id))
    idle = int(cycle_id) - last_seen

    if status == "latent":
        if activation >= ACTIVATION_TO_ACTIVE:
            shadow["status"] = "active"
    elif status == "active":
        if idle > COOLING_IDLE_CYCLES:
            shadow["status"] = "cooling"
    elif status == "cooling":
        if files_delta >= 1 and persistence >= REALIZED_PERSISTENCE_MIN:
            shadow["status"] = "realized"
        elif idle > ABANDON_IDLE_CYCLES:
            shadow["status"] = "abandoned"
    # realized / abandoned: terminal


# ============================================================
# Decay (orchestration §5 「放置 → decay」literal)
# ============================================================

def _decay_unmentioned(shadows: list, cycle_id: int) -> None:
    """当該 cycle で言及されなかった shadow の activation を緩く減衰 (in-place)。

    realized / abandoned は除外 (終端ステータス、再活性しない設計)。
    last_seen_cycle == cycle_id は当該 cycle で touch された印 → decay 対象外。
    """
    for s in shadows:
        if s.get("status") in ("realized", "abandoned"):
            continue
        if int(s.get("last_seen_cycle", 0)) >= int(cycle_id):
            continue
        s["activation"] = round(float(s.get("activation", 0.0)) * DECAY_RATE, 6)


# ============================================================
# Signal extraction (cycle 内 observable evidence、構造的 read-only)
# ============================================================

def _extract_signals(state: dict, cycle_id: int, subj_count_prev: int) -> list:
    """当該 cycle 内の observable evidence を抽出。

    識別力ある source 選定 (smoke 観察で tune 可):
      - subjective_entries 末尾 delta (本 cycle で append された subj、intent と
        expect から signal を作る)。embedding が既に注入済 (Phase 1 enrichment) で
        cost 0 で再利用可能。
      - pending で last_cycle == cycle_id のもの (本 cycle に touch された pending、
        content_intent / content_observable から signal を作る、embedding 別途生成)

    過剰 source 取込み回避 (CLAUDE.md「投機的なものはゼロ」):
      reflection note は memory_store 経由で別 jsonl に書かれるため Slice 5 では skip。
      raw_events 失敗 entry も interestingness 増分の signal 候補だが Slice 5 範囲外
      (BP-3 観察後に必要になれば追加)。

    Returns:
        list[dict] — 各 entry: {source, id, label, text, origin, type, embedding?}
    """
    signals: list = []
    cur = int(cycle_id)

    # 1) subjective_entries の delta (本 cycle 末尾 append 分)
    subj_list = state.get("subjective_entries", []) or []
    delta_subj = subj_list[max(0, int(subj_count_prev)):]
    for s in delta_subj:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id", ""))
        intent = (s.get("intent") or "").strip()
        expect = (s.get("expect") or "").strip()
        emb = s.get("embedding") if isinstance(s.get("embedding"), list) else None
        # intent 主軸の signal (label = intent 先頭)
        if intent:
            signals.append({
                "source": SOURCE_SUBJECTIVE_ENTRY,
                "id": sid,
                "label": intent[:LABEL_MAX_LEN],
                "text": f"intent: {intent}\nexpect: {expect}" if expect else intent,
                "origin": "intent",
                "type": "intent",
                "embedding": emb,
            })

    # 2) pending で last_cycle == cycle_id のもの
    for p in state.get("pending", []) or []:
        if not isinstance(p, dict):
            continue
        if int(p.get("last_cycle", -1)) != cur:
            continue
        text = (p.get("content_intent") or p.get("content_observable") or "").strip()
        if not text:
            continue
        signals.append({
            "source": SOURCE_PENDING,
            "id": str(p.get("id", "")),
            "label": text[:LABEL_MAX_LEN],
            "text": text,
            "origin": "pending",
            "type": "pending",
            "embedding": None,  # pending は embedding 未保持、必要なら caller embed_fn
        })

    return signals


# ============================================================
# Reinforcement (signal → shadow 更新)
# ============================================================

def _apply_signal(shadow: dict, signal: dict, cycle_id: int) -> None:
    """1 つの signal を 1 つの shadow に反映 (in-place)。

    更新規則 (orchestration §5 literal):
      intent / pending / reflection 言及 → activation += 1
      raw failure → interestingness += 1 (Slice 5 範囲外、Slice 6+ で追加)
      file_write delta → caller 側で persistence += delta (本関数は触らない)
      tool_level >= 2 → caller 側で risk += (本関数は触らない、Slice 6+)
    """
    typ = signal.get("type", "")
    if typ in ("intent", "pending", "reflection"):
        shadow["activation"] = float(shadow.get("activation", 0.0)) + 1.0
    refs = shadow.setdefault("evidence_refs", [])
    refs.append(make_evidence_ref(
        signal.get("source", ""), signal.get("id", ""), cycle_id
    ))
    shadow["last_seen_cycle"] = int(cycle_id)


# ============================================================
# Public entry: update_goal_shadows
# ============================================================

def update_goal_shadows(
    state: dict,
    cycle_id: int,
    files_written_delta: int = 0,
    subj_count_prev: int = 0,
    *,
    embed_fn: Optional[Callable] = None,
    cosine_fn: Optional[Callable] = None,
) -> list:
    """cycle 末で呼び出す observe + update entry (controller 不変、観測のみ)。

    main.py:1138 emit_cycle_metrics 直前で呼ぶ。reflect 継続原則で例外は呼出側で
    catch する (cycle 全体は止めない、metrics と同 pattern)。

    処理順 (Codex 追加指摘 #1 採用、caller 側責務との分担):
      1. signals 抽出 (subj delta + pending touched)
      2. embed_fn ある & signal.embedding 不在のものを batch embed (cost 抑制)
      3. 各 signal で既存 shadow を embedding cosine ≥ 0.85 で同定、なければ新規生成
      4. signal を shadow に反映 (activation / evidence_refs / last_seen)
      5. files_written_delta を全非終端 shadow の persistence に加算
      6. 当該 cycle 未言及の shadow に activation decay
      7. 全 shadow に status 遷移を適用

    Args:
        state: profile state (in-place 変更)
        cycle_id: 現 cycle id
        files_written_delta: 当該 cycle 内で増えた files_written 件数 (caller 計算)
        subj_count_prev: 前 cycle 末時点の subjective_entries 件数 (caller 渡し)
        embed_fn: (list[str]) -> list[list[float]] (None で embedding 同定 skip)
        cosine_fn: (vec, vec) -> float (None で skip)

    Returns:
        更新後の shadows list (state["goal_shadows"] と同オブジェクト、test 用)
    """
    shadows = state.setdefault("goal_shadows", [])
    signals = _extract_signals(state, cycle_id, subj_count_prev)

    # signal の embedding を埋める (subj は既存、pending は必要なら新規生成)
    if embed_fn is not None and signals:
        need_embed_idx = [
            i for i, s in enumerate(signals)
            if not s.get("embedding") and s.get("text")
        ]
        if need_embed_idx:
            try:
                texts = [signals[i]["text"] for i in need_embed_idx]
                vecs = embed_fn(texts) or []
                for j, i in enumerate(need_embed_idx):
                    if j < len(vecs) and vecs[j]:
                        signals[i]["embedding"] = list(vecs[j])
            except Exception:
                pass  # embed 失敗時は label fallback で同定する

    # 各 signal を shadow に反映
    for sig in signals:
        emb = sig.get("embedding")
        match = find_matching_shadow(shadows, sig.get("label", ""), emb, cosine_fn)
        if match is None:
            match = _new_goal_shadow(
                label=sig.get("label", "") or "(unnamed)",
                origin=sig.get("origin", "intent"),
                cycle_id=cycle_id,
                embedding=_l2_normalize(emb) if emb else None,
            )
            shadows.append(match)
        else:
            if emb:
                match["embedding"] = update_centroid(
                    match.get("embedding") or [], emb, CENTROID_EMA_ALPHA
                )
        _apply_signal(match, sig, cycle_id)

    # files_written delta を非終端 shadow の persistence に加算
    if files_written_delta and files_written_delta > 0:
        for s in shadows:
            if s.get("status") in ("realized", "abandoned"):
                continue
            s["persistence"] = float(s.get("persistence", 0.0)) + float(files_written_delta)

    # 未言及 shadow の activation decay
    _decay_unmentioned(shadows, cycle_id)

    # status 遷移
    delta_int = int(files_written_delta or 0)
    for s in shadows:
        _apply_status_transition(s, cycle_id, delta_int)

    return shadows


# ============================================================
# Summary for metrics output (BP-3 判定基準を可視化)
# ============================================================

def summarize_for_metrics(shadows: list, top_n: int = 5) -> dict:
    """metrics_events.jsonl 用要約 dict (観測のみ、controller 不変)。

    BP-3 判定基準 (orchestration §4-1 BP-3 literal):
      - state["goal_shadows"] が成長 → count
      - status 遷移が起きる → status_distribution
      - evidence binding が raw/subjective id に正しく紐づく → total_evidence_refs
      - controller 挙動退行なし → 本 Slice では介入なし (構造的に no-op)

    top_active は activation 降順で top_n、smoke 観察で「どの label が育っているか」
    を 1 行で見るための要約 (label を 40 字で切詰)。
    """
    shadows = shadows or []
    total = len(shadows)
    status_dist: dict = {}
    for s in shadows:
        st = s.get("status", "latent")
        status_dist[st] = status_dist.get(st, 0) + 1
    total_refs = sum(len(s.get("evidence_refs", []) or []) for s in shadows)
    top_active = sorted(
        (
            {
                "id": s.get("id", ""),
                "label": (s.get("label", "") or "")[:40],
                "status": s.get("status", ""),
                "activation": round(float(s.get("activation", 0.0)), 6),
                "persistence": round(float(s.get("persistence", 0.0)), 6),
                "evidence_count": len(s.get("evidence_refs", []) or []),
            }
            for s in shadows
        ),
        key=lambda d: d["activation"],
        reverse=True,
    )[:top_n]
    return {
        "count": total,
        "status_distribution": status_dist,
        "total_evidence_refs": total_refs,
        "top_active": top_active,
    }
