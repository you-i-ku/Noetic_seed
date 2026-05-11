"""Slice 3 (orchestration §3 P1 #3): information_gain / model_resolution_gain 初期版。

orchestration §3 P1 #3 + §5 Slice 3。Slice 1 (subjective 永続化) + Slice 2
(metrics_events.jsonl + 靄 A-E) の上に、情報理論 / Bayesian 推論 / Active Inference
数学フレームワークの数式群を Noetic 6 項目に同型 mapping した情報利得の初期版。

設計原則 (memory `feedback_no_biological_mimicry` literal):
  Noetic は生命模倣を採らない。本 module の数式参照は **数学構造のみ** であり、
  脳のモデル / 認知科学的解釈としての参照は含まない。「脳が / 生物が こうだから
  Noetic も」という根拠は不採用 (Codex 諮問 2026-05-09 でも同制約で第三者確認済)。

設計確定事項 (2026-05-09 ゆう判断):
  ① 重み: PLAN literal どおり全部 1.0 (sub 係数も 1.0、マジックナンバー 0)
  ② 各項目独立記録: metrics_events.jsonl の info_gain dict に 6 項目 + 統合 +
     model_resolution_gain で別個に保存、後で重み調整可能 (BP-1 smoke 後)
  ③ controller 不変 (PLAN literal「最初は測定のみ」、Slice 6 まで温存)
  ④ prev_state handoff: state["_info_gain_prev"] で必要 field のみ snapshot
  ⑤ pe_drop 正規化: PE_NORMALIZATION_FACTOR=100.0 で /100 線形 (BP-1 hotfix
     2026-05-09、Codex Q1 推奨確定。reflection.py:82 既存 pattern と整合)

数学的裏付け (literal、情報理論 / 確率論 / Bayesian 推論フレームワーク限定):
  - Active Inference 数学フレームワーク (variational free energy minimization):
      G(a) = -E[ln p*(o')] - I[s'; o'|a]   (extrinsic + epistemic)
      epistemic value = E_q D_KL[q(s'|o',a) || q(s'|a)] = expected information gain
      (KL divergence 数学として参照、認知モデルとしての参照は不採用)
  - Mutual Information 情報理論 (scikit-learn implementation reference):
      MI[X;Y] = E[log(p(X,Y) / (p(X)·p(Y)))]、nats 単位、値域 [0, ∞)
      (saturation は数学的必然ではなくスコアリング設計の任意選択)
  - Curiosity-as-Information-Gain 数式 (CIG, clawRxiv 2603.00009 数式部分):
      r^CIG = N(s,a) · L(s,a) · C(s)
      N: ensemble mutual information (epistemic uncertainty)
      L: σ(α - mean H) noisy TV filter (aleatoric gate)
      C: exp(-β · Var[Q]) competence weight
      (掛け算式 = gating 意味論、Noetic は加算式 = additive utility 採用)
  - Bayesian 推論 (確率分布の posterior precision / KL divergence 数学):
      prediction error の絶対値減少 → posterior 分布の集中度上昇
      (確率論の数学構造として参照、脳 predictive coding の認知モデルとしてではない)
  - A-MEM zettelkasten link generation 数式 (arxiv 2502.12110, ICLR 2025):
      memory entry 間の semantic similarity に基づく link 生成 + evolution

Noetic 6 項目への mapping:

  | 項目 | 数学的近似 | Noetic 実装 |
  |---|---|---|
  | novelty_gain | epistemic value I[s';o'|a] (KL divergence) | 新 entry embedding の既存からの 1 - max cosine similarity |
  | effective_change_gain | pragmatic value -H[q,p*] (情報理論的 cross entropy) | last_e1 cycle 間差分 (clamp +) |
  | memory_link_gain | zettelkasten link generation 数式 | strength 合計 cycle 間 diff (新規 link の initial strength で捕捉) |
  | world_model_resolution_gain | Bayesian posterior precision 上昇 + 情報理論的 entropy 縮小 | last_prediction_error 減少 (/100 正規化) + 靄 B (local_density_mean) 増加 |
  | capability_gain | CIG C 項 (exp(-β · Var[Q]) 数式) | tool 別 success_rate variance 縮小 + tool 多様性 delta |
  | redundancy_penalty | CIG L 項 (noisy TV filter 数式) | 同 tool 連発 + embedding 中心固着 |

統合式 (PLAN §3 #3 literal):
  info_gain = novelty + effective_change + memory_link + world_model_resolution
            + capability - redundancy_penalty
  model_resolution_gain = world_model_resolution_gain (独立 metric)

たとえ:
  Slice 2 で観測台 (metrics) と靄計器 5 個を建てた。Slice 3 は観測台に
  「賢くなった度」計器 6 個を取り付ける工程。各計器は情報理論 / Bayesian 推論 /
  Active Inference の数学を Noetic embedding 空間用に近似した実装。森が
  広がった度 / 道が立ったか / 道が太くなったか / 靄が晴れた度 / 道具が
  増えた度 / 同じ場所をぐるぐるしてないか、を毎 cycle 1 行で読める。

正典 pointer:
  WORLD_MODEL_DESIGN/NOETIC_INTEGRATED_ORCHESTRATION_PLAN.md §3 P1 #3 + §5 Slice 3
  WORLD_MODEL_DESIGN/NOETIC_INTEGRATED_ORCHESTRATION_PLAN.md §5 Slice 6 (density_gain
  正規化整合の Slice 6 再検討負債、BP-1 由来)
"""
from __future__ import annotations

import statistics
from typing import Optional

# ============================================================
# 定数 (PLAN literal の自然定数 + smoke 観察起点の暫定閾値)
# ============================================================

NOVELTY_K_NEIGHBORS = 20         # 新 entry の比較対象 (recent N entry)
RECENT_TOOL_WINDOW = 5           # redundancy: 連発検知窓 (action_ledger 末尾)
RECENT_EMB_WINDOW = 5            # redundancy: 中心固着検知窓
CENTROID_VARIANCE_FLOOR = 0.3    # redundancy: emb mean dist 下限 (これ以下で固着)
PREDICTOR_MIN_ATTEMPTS = 3       # capability: variance 計算最低試行数
CYCLE_KEY = "_info_gain_prev"    # state snapshot key (state.py default に登録)
PE_NORMALIZATION_FACTOR = 100.0  # last_prediction_error 値域 0-100 を [0,1] に揃える
# Slice 6.5 Step 3a (PLAN §4.1): Lv3 数式 (log ratio) 用の 0 除算回避 ε。
# preference_distribution.EPS と数値同じだが、独立 module 整合性のため別定数。
EPS_LOG = 1e-3
# Slice 6.5 Step 4 (PLAN §4.3.1): _consecutive_penalty の log scale 参照点。
# RECENT_TOOL_WINDOW / 2 = 2.5、log(2.5) ≈ 0.916 nat が「多様」と「単発」の境界目安。
DIVERSITY_BASELINE = 2.5
                                 # (BP-1 hotfix 2026-05-09、reflection.py:82 既存 pattern 整合)


# ============================================================
# 公開 API
# ============================================================

def compute_efe_components(
    state: dict,
    prev_snapshot: dict,
    entries_with_embedding: list,
    links: list,
    fog_now: Optional[dict] = None,
) -> dict:
    """EFE 9 成分 + 3 カテゴリ + 統合 G を計算 (Slice 6.5 Step 4 公開 API、PLAN §4.4 + §4.5)。

    旧 compute_info_gain_components 置換 (案 α 確定、Codex Q-d-fix-3 ゆう確定、alias なし)。
    Step 4 commit 時点では旧公開 API も temporary 温存 (Step 5 で完全削除 + metrics.py 連動)。

    9 成分 (PLAN §4.1 + §4.2 + §4.3、Lv3 nat 単位):
      epistemic (4): novelty / pe_drop / density / memory_link
      pragmatic (3): effective_change / competence / tool_diversity
      regularization (2): consecutive_penalty / centroid_stuck

    3 カテゴリ:
      epistemic_gain = Σ epistemic 4 成分
      pragmatic_gain = Σ pragmatic 3 成分
      regularization = Σ regularization 2 成分

    統合 G (Active Inference literal、最小化対象):
      G_noetic = -pragmatic_gain - epistemic_gain + regularization

    output schema (§4.5 literal): efe dict + C 情報 (state["_efe_C"] 由来、Step 6 hook 配線済)。

    Args / Returns: §4.5 literal、controller には流さない (Slice 7 まで「計測のみ」継承)。
    """
    novelty = _novelty_gain(entries_with_embedding, prev_snapshot)
    pe_drop = _pe_drop_gain(state, prev_snapshot)
    density = _density_gain(prev_snapshot, fog_now)
    memory_link = _memory_link_gain(links, prev_snapshot)

    effective_change = _effective_change_gain(state, prev_snapshot, entries_with_embedding)
    competence = _competence_gain(state, prev_snapshot)
    tool_diversity = _tool_diversity_gain(state, prev_snapshot)

    consecutive_pen = _consecutive_penalty(state)
    centroid_stk = _centroid_stuck(entries_with_embedding)

    epistemic_gain = novelty + pe_drop + density + memory_link
    pragmatic_gain = effective_change + competence + tool_diversity
    regularization = consecutive_pen + centroid_stk

    G = -pragmatic_gain - epistemic_gain + regularization

    # C 情報 (state["_efe_C"] 由来、Step 6 update_self hook 配線済)
    C_data = state.get("_efe_C") or {}

    return {
        "G": round(G, 6),
        "pragmatic_gain": round(pragmatic_gain, 6),
        "epistemic_gain": round(epistemic_gain, 6),
        "regularization": round(regularization, 6),
        "novelty": round(novelty, 6),
        "pe_drop": round(pe_drop, 6),
        "density": round(density, 6),
        "memory_link": round(memory_link, 6),
        "effective_change": round(effective_change, 6),
        "competence": round(competence, 6),
        "tool_diversity": round(tool_diversity, 6),
        "consecutive_penalty": round(consecutive_pen, 6),
        "centroid_stuck": round(centroid_stk, 6),
        "C_source_self_keys": list(C_data.get("source_keys", [])),
        "C_per_key_confidence": dict(C_data.get("per_key_confidence", {})),
        "C_per_key_variance": dict(C_data.get("per_key_variance", {})),
        "C_entropy": round(float(C_data.get("C_entropy", 0.0)), 6),
        "C_update_cycle": int(state.get("_efe_C_update_cycle", -1)),
    }


def snapshot_for_next_cycle(
    state: dict,
    entries_with_embedding: list,
    links: list,
    fog_now: Optional[dict] = None,
) -> dict:
    """次 cycle の info_gain 計算用に最小 snapshot を組み立てる。

    cycle 末 emit 後に ``state[CYCLE_KEY] = snapshot_for_next_cycle(...)`` で更新する。

    含む field (diff 計算に必要なものだけ、軽量化):
      last_e1, last_prediction_error: scalar 比較用 (0.0 保持、Codex P2 fix)
      entries_count: 整数 diff (新 entry 検出、subj+mem 全 embedding 持ち合計)
      memory_links_strength_total: float (link gain は strength 合計 diff で捕捉)
      predictor_success_rate_var: capability gain diff 用
      predictor_tool_count: tool 多様性 diff 用
      fog_local_density_mean: density gain diff 用 (None 許容)
    """
    pc = state.get("predictor_confidence", {}) or {}
    success_rates = []
    tool_count = 0
    for tdata in pc.values():
        if not isinstance(tdata, dict):
            continue
        tool_count += 1
        succ = int(tdata.get("success", 0) or 0)
        fail = int(tdata.get("fail", 0) or 0)
        attempts = succ + fail
        if attempts >= PREDICTOR_MIN_ATTEMPTS:
            success_rates.append(succ / attempts)
    var_now = (
        statistics.variance(success_rates)
        if len(success_rates) >= 2 else 0.0
    )

    strength_total = sum(
        float(l.get("strength", 0.0) or 0.0)
        for l in (links or []) if isinstance(l, dict)
    )

    pe_now = state.get("last_prediction_error")
    fog_density = (fog_now or {}).get("local_density_mean")

    # Codex review 2026-05-09 P2 fix: legitimate 0.0 last_e1 を default で潰さない
    val_e1 = state.get("last_e1", 0.5)
    last_e1_snapshot = float(val_e1) if isinstance(val_e1, (int, float)) else 0.5

    # Slice 6.5 Step 3b (2026-05-11、PLAN §4.2.1 + §4.2.3): next cycle の
    # _effective_change_gain / _tool_diversity_gain で必要な前値を保存。
    # P2-1 fix (Codex review 2026-05-11): 新 entry 絞り込みに prev_snapshot
    # (= state["_info_gain_prev"]) が必要。snapshot_for_next_cycle は引数で
    # 受けてないため state から直接取得 (case b 採用、metrics.py signature 非破壊)。
    import math
    _prev_for_log_p = state.get(CYCLE_KEY, {}) or {}
    effective_change_log_p_now = _compute_log_p_now(
        state, _prev_for_log_p, entries_with_embedding
    )

    ledger = state.get("action_ledger", []) or []
    tool_counts_now: dict = {}
    for entry in ledger[-RECENT_TOOL_WINDOW:]:
        if isinstance(entry, dict) and entry.get("tool"):
            t = str(entry["tool"])
            tool_counts_now[t] = tool_counts_now.get(t, 0) + 1
    total_now = sum(tool_counts_now.values())
    tool_entropy_now = (
        -sum((c / total_now) * math.log((c / total_now) + EPS_LOG)
             for c in tool_counts_now.values())
        if total_now > 0 else 0.0
    )

    return {
        "last_e1": last_e1_snapshot,
        "last_prediction_error": (
            float(pe_now) if isinstance(pe_now, (int, float)) else None
        ),
        "entries_count": len(entries_with_embedding or []),
        "memory_links_strength_total": round(strength_total, 6),
        "predictor_success_rate_var": round(var_now, 6),
        "predictor_tool_count": tool_count,
        "fog_local_density_mean": (
            float(fog_density) if isinstance(fog_density, (int, float)) else None
        ),
        # Slice 6.5 Step 3b 追加: next cycle diff 計算用前値
        "effective_change_log_p": (
            float(effective_change_log_p_now)
            if isinstance(effective_change_log_p_now, (int, float))
            else None
        ),
        "tool_entropy": round(tool_entropy_now, 6),
    }


# ============================================================
# 内部 helper: 6 項目 (各々 1 関数 = literal 1 対応)
# ============================================================

def _novelty_gain(entries_with_embedding: list, prev_snapshot: dict) -> float:
    """epistemic surprise の Noetic 近似 (Lv3、PLAN §4.1.1、nat 単位)。

    Slice 6.5 Step 3a (2026-05-11) 数式変更:
        旧: 1 - max_cosine_similarity ∈ [0, 1]
        新: -log(max_cosine_similarity + ε) ∈ [0, -log(ε)] nat

    cosine → 1 (一致) → log(1+ε) ≈ ε ≈ 0 nat (新観測なし)
    cosine → 0 (直交) → -log(ε) ≈ 6.9 nat (= 大 surprise)

    **入力順序契約**: ``entries_with_embedding`` は newest first
    (``load_all_subjective_entries`` + ``load_all_memories`` が共に新しい順)。

    新 entry 0 件 → 0.0 (利得なし)
    初 cycle (existing 0 件) → 0.0 (比較不可、安全 default)

    識別力: cosine=0.5 で 旧 0.5 / 新 ≈ 0.693 nat、cosine=0 で 旧 1.0 / 新 ≈ 6.9 nat。
    cosine 値域は bge-m3 L2 正規化済で [0, 1] 範囲、負値の安全のため max(0, ·) clamp。
    """
    try:
        import numpy as np
    except ImportError:
        return 0.0

    embs = [
        e["embedding"] for e in (entries_with_embedding or [])
        if isinstance(e, dict) and isinstance(e.get("embedding"), list)
    ]
    prev_count = int(prev_snapshot.get("entries_count", 0) or 0)
    n_new = len(embs) - prev_count
    if n_new <= 0:
        return 0.0  # 新 entry なし

    # newest first: 先頭 n_new 件が新、続く NOVELTY_K_NEIGHBORS 件が直近既存
    new_embs = embs[:n_new]
    existing_embs = embs[n_new:n_new + NOVELTY_K_NEIGHBORS]
    if not existing_embs:
        return 0.0  # 比較不可 (初 cycle)

    new_arr = np.array(new_embs, dtype=np.float32)
    ex_arr = np.array(existing_embs, dtype=np.float32)
    # bge-m3 は L2 正規化済前提 (Phase 1 整合、metrics.py:fog_embedding_spread 同流儀)
    sims = new_arr @ ex_arr.T  # (N_new, N_existing)
    max_sims = sims.max(axis=1)  # 各 new に対する最大類似度
    # Lv3: -log(max_sim + ε) で nat 単位 surprise、cosine 値を [0, 1] clamp して log 引数安全
    max_sims_clamped = np.clip(max_sims, 0.0, 1.0)
    novelties = -np.log(max_sims_clamped + EPS_LOG)
    return float(novelties.mean())


def _compute_log_p_now(
    state: dict, prev_snapshot: dict, entries_with_embedding: list
) -> Optional[float]:
    """現 cycle の **新 entry** 群の平均 log_p(o | C_now) (Lv3 helper、PLAN §4.2.1)。

    Codex review breakpoint ① P2-1 fix (2026-05-11): 旧実装は entries_with_embedding 全体を
    平均化していて、新 entry なし cycle でも過去の全 log_p が混入する bug があった。
    `_novelty_gain` (line 251-252) と同じ pattern で prev_snapshot.entries_count を使い
    newest first 先頭 n_new 件だけを new 観測として絞る。

    現 cycle の preference C (state["_efe_C"]) で estimate_density を構築、newest n_new
    entry の log density を平均する。calculable な値があれば float、不能 (C 未構築 /
    新 entry なし / embedding なし) → None (_effective_change_gain と
    snapshot_for_next_cycle で共用)。
    """
    C_data = state.get("_efe_C")
    if not C_data or not C_data.get("components"):
        return None
    embs = [
        e["embedding"] for e in (entries_with_embedding or [])
        if isinstance(e, dict) and isinstance(e.get("embedding"), list)
    ]
    prev_count = int(prev_snapshot.get("entries_count", 0) or 0)
    n_new = len(embs) - prev_count
    if n_new <= 0:
        return None  # 新 entry なし (両実装一致 graceful)
    new_embs = embs[:n_new]  # newest first 先頭 n_new 件
    from core.preference_distribution import estimate_density
    log_p_fn = estimate_density(C_data["components"], method="vmf")
    return sum(log_p_fn(e) for e in new_embs) / len(new_embs)


def _effective_change_gain(
    state: dict, prev_snapshot: dict, entries_with_embedding: list
) -> float:
    """preference との cross-entropy 改善量 (Lv3、PLAN §4.2.1、nat 単位)。

    Slice 6.5 Step 3b (2026-05-11) 数式変更:
        旧: max(0, e1_now - e1_prev)  ← E1 [0,1] 値域の正部分
        新: log p*(o_t | C_t) - log p*(o_{t-1} | C_{t-1})  ← preference C 経由 cross-entropy diff

    現 cycle の newest entry 群を vMF mixture C で評価した log_p 平均と、
    前 cycle snapshot 値の差分を返す。preference に合う観測で正、離れる観測で負
    (旧 clamp で潰れない、Lv3 では離反方向も負値で捕捉)。

    C 未構築 (state._efe_C = None or empty) → 0 (両実装一致 graceful)
    新 entry なし or 初 cycle (prev 値なし) → 0 (両実装一致 graceful)

    識別力: 旧 e1 [0,1] 値域内の小さい diff vs 新 log scale で nat 単位 (規模差大)。
    """
    log_p_now = _compute_log_p_now(state, prev_snapshot, entries_with_embedding)
    if log_p_now is None:
        return 0.0
    log_p_prev = prev_snapshot.get("effective_change_log_p")
    if not isinstance(log_p_prev, (int, float)):
        return 0.0
    return float(log_p_now) - float(log_p_prev)


def _memory_link_gain(links: list, prev_snapshot: dict) -> float:
    """A-MEM link generation + memory evolution の Noetic 近似 (Lv3、PLAN §4.1.4、nat 単位)。

    Slice 6.5 Step 3a (2026-05-11) 数式変更:
        旧: max(0, strength_now - strength_prev)  ← raw diff、正値クランプ
        新: log((1 + strength_now) / (1 + strength_prev))  ← log ratio、負値保持

    新規 link 追加 / Physarum α 強化 で strength_now > strength_prev → 正
    decay で strength 減 → 負 (構造的情報減少、Lv3 では捕捉)
    不変 → 0 (両実装一致)

    +1 (Laplace 平滑化的) は 0 除算回避 + 小規模時の数値安定化 (1 + strength ≥ 1 で正)。

    識別力: 旧 max(0, diff)=4.0 vs 新 log(6/2)≈1.099 (例 strength_now=5, prev=1)、
    decay で 旧 0 (clamp) vs 新 負値 (識別、Lv3 では情報減少が見える)。
    """
    import math
    strength_now = sum(
        float(l.get("strength", 0.0) or 0.0)
        for l in (links or []) if isinstance(l, dict)
    )
    strength_prev = float(prev_snapshot.get("memory_links_strength_total", strength_now) or 0.0)
    return math.log((1.0 + strength_now) / (1.0 + strength_prev))


def _pe_drop_gain(state: dict, prev_snapshot: dict) -> float:
    """Bayesian posterior precision 上昇 (Lv3、PLAN §4.1.2、nat 単位)。

    Slice 6.5 Step 3a (2026-05-11、_world_model_resolution_gain から split):
        旧: max(0, (prev_pe - now_pe) / 100)  ← /100 線形 + 正値クランプ
        新: log((prev_pe + ε) / (now_pe + ε))  ← log ratio、負値保持

    prev_pe > now_pe → 正 (predictor 精度上昇)
    prev_pe < now_pe → 負 (predictor 精度低下、情報的に意味あり predictor 学習で重要)
    prev or now が None → 0 (graceful、初 cycle / history 不足)

    識別力: prev=80, now=10 で 旧 0.7 vs 新 log(80.001/10.001) ≈ 2.08 nat、
    prev=10, now=80 で 旧 0 (clamp) vs 新 ≈ -2.08 nat (識別、Lv3 では負値保持)。
    """
    import math
    pe_now = state.get("last_prediction_error")
    pe_prev = prev_snapshot.get("last_prediction_error")
    if not (isinstance(pe_now, (int, float)) and isinstance(pe_prev, (int, float))):
        return 0.0
    return math.log((float(pe_prev) + EPS_LOG) / (float(pe_now) + EPS_LOG))


def _density_gain(prev_snapshot: dict, fog_now: Optional[dict]) -> float:
    """memory graph 局所密度の MI delta 近似 (Lv3、PLAN §4.1.3、nat 単位)。

    Slice 6.5 Step 3a (2026-05-11、_world_model_resolution_gain から split):
        旧: max(0, now - prev)  ← raw diff、正値クランプ
        新: log((now + ε) / (prev + ε))  ← log ratio、負値保持

    now > prev → 正 (embedding 空間解像度向上)
    now < prev → 負 (解像度低下、memory_graph edge 削除等で構造的)
    どちらかが None → 0 (graceful)

    識別力: now=0.3, prev=0.1 で 旧 0.2 vs 新 log(0.301/0.101)≈1.092 nat、
    now=0.1, prev=0.3 で 旧 0 (clamp) vs 新 ≈ -1.092 nat (識別、Lv3 で負値捕捉)。
    """
    import math
    density_now = (fog_now or {}).get("local_density_mean")
    density_prev = prev_snapshot.get("fog_local_density_mean")
    if not (
        isinstance(density_now, (int, float))
        and isinstance(density_prev, (int, float))
    ):
        return 0.0
    return math.log((float(density_now) + EPS_LOG) / (float(density_prev) + EPS_LOG))


def _competence_gain(state: dict, prev_snapshot: dict) -> float:
    """CIG C 項 (tool 上達) の log 増分 (Lv3、PLAN §4.2.2、nat 単位)。

    Slice 6.5 Step 3b (2026-05-11、_capability_gain から split):
        旧: max(0, var_prev - var_now)  ← variance 差そのまま、正値クランプ
        新: β · (Var[Q]_prev - Var[Q]_now)  ← CIG C(s)=exp(-β·Var[Q]) の log 増分

    β = 1.0 初期 (PLAN literal「マジックナンバー 0」継承、smoke 観察後 tune 候補)。
    variance 拡大で負値保持 (predictor 学習で variance 増加もありえる、Lv3 で捕捉)。

    識別力: 旧 max(0, ·) clamp vs 新 β·diff (負値保持) で variance 拡大時 fail (誤実装)。
    """
    BETA = 1.0
    pc = state.get("predictor_confidence", {}) or {}
    success_rates = []
    for tdata in pc.values():
        if not isinstance(tdata, dict):
            continue
        succ = int(tdata.get("success", 0) or 0)
        fail = int(tdata.get("fail", 0) or 0)
        attempts = succ + fail
        if attempts >= PREDICTOR_MIN_ATTEMPTS:
            success_rates.append(succ / attempts)
    var_now = (
        statistics.variance(success_rates)
        if len(success_rates) >= 2 else 0.0
    )
    var_prev = float(prev_snapshot.get("predictor_success_rate_var", var_now) or 0.0)
    return BETA * (var_prev - var_now)


def _tool_diversity_gain(state: dict, prev_snapshot: dict) -> float:
    """tool 使用分布の Shannon entropy delta (Lv3、PLAN §4.2.3、nat 単位)。

    Slice 6.5 Step 3b (2026-05-11、_capability_gain から split):
        旧: max(0, tool_count_now - tool_count_prev)  ← count diff、正値クランプ
        新: H_now[tool] - H_prev[tool]  ← Shannon entropy delta (nat)
            H[tool] = -Σ_i p(tool_i) log p(tool_i)
            p_i は recent action_ledger 内 tool 使用比 (RECENT_TOOL_WINDOW 件)

    新 tool 試行 → entropy 増 → 正 (旧と類似)
    同 tool 連発 → entropy 減 → 負 (旧 max(0,·) clamp で 0、識別)
    分布不変 → 0 (両実装一致)

    識別力: 旧 count diff clamp vs 新 Shannon entropy diff、連発時の負値で誤実装 fail。
    """
    import math
    ledger = state.get("action_ledger", []) or []
    tool_counts: dict = {}
    for entry in ledger[-RECENT_TOOL_WINDOW:]:
        if isinstance(entry, dict) and entry.get("tool"):
            t = str(entry["tool"])
            tool_counts[t] = tool_counts.get(t, 0) + 1

    total = sum(tool_counts.values())
    if total == 0:
        H_now = 0.0
    else:
        H_now = -sum(
            (c / total) * math.log((c / total) + EPS_LOG)
            for c in tool_counts.values()
        )

    H_prev = float(prev_snapshot.get("tool_entropy", H_now) or 0.0)
    return H_now - H_prev


def _consecutive_penalty(state: dict) -> float:
    """同 tool 連発の log ratio (Lv3、PLAN §4.3.1、nat 単位)。

    Slice 6.5 Step 4 (2026-05-11、_redundancy_penalty から split):
        旧: max(0, consecutive_run - 1)  ← raw 連発件数 - 1
        新: log(1 + consecutive_run) - log(DIVERSITY_BASELINE)  ← log scale 相対

    DIVERSITY_BASELINE = 2.5 (= RECENT_TOOL_WINDOW / 2) で「多様」「単発」境界目安。
    - 単発 (run=1): log(2) - log(2.5) ≈ -0.223 nat (負、多様、識別)
    - 5 連発 (run=5): log(6) - log(2.5) ≈ 0.876 nat (正、固着、識別)
    - 空 ledger → 0 (両実装一致 graceful)
    """
    import math
    ledger = state.get("action_ledger", []) or []
    recent_tools = []
    for entry in ledger[-RECENT_TOOL_WINDOW:]:
        if isinstance(entry, dict) and entry.get("tool"):
            recent_tools.append(str(entry["tool"]))
    if not recent_tools:
        return 0.0
    last_tool = recent_tools[-1]
    consecutive_run = 1
    for t in reversed(recent_tools[:-1]):
        if t == last_tool:
            consecutive_run += 1
        else:
            break
    return math.log(1 + consecutive_run) - math.log(DIVERSITY_BASELINE)


def _centroid_stuck(entries_with_embedding: list) -> float:
    """embedding 中心固着の log ratio (Lv3、PLAN §4.3.2、nat 単位)。

    Slice 6.5 Step 4 (2026-05-11、_redundancy_penalty から split):
        旧: (FLOOR - mean_dist) / FLOOR  ← [0,1] 正規化、mean_dist < FLOOR で正
        新: log(FLOOR / max(mean_dist, ε))  ← log ratio、負値保持

    - mean_dist→0 (完全固着) → log(FLOOR/ε) ≈ log(300) ≈ 5.7 nat (大)
    - mean_dist=FLOOR (境界) → 0 nat
    - mean_dist > FLOOR (多様) → 負 (旧 max(0,·) clamp 撤去、Lv3 で識別)
    - 観察 5 件未満 → 0 (両実装一致 graceful)

    **入力順序契約**: entries_with_embedding は newest first、`[:RECENT_EMB_WINDOW]`
    で先頭 N 件 (newest 直近) を取る (_redundancy_penalty と同 pattern、Codex review
    2026-05-09 P2 #3 fix 整合)。
    """
    import math
    try:
        import numpy as np
    except ImportError:
        return 0.0
    embs = [
        e["embedding"] for e in (entries_with_embedding or [])[:RECENT_EMB_WINDOW]
        if isinstance(e, dict) and isinstance(e.get("embedding"), list)
    ]
    if len(embs) < RECENT_EMB_WINDOW:
        return 0.0  # 観察 5 件未満 graceful
    arr = np.array(embs, dtype=np.float32)
    centroid = arr.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 1e-9:
        return 0.0
    centroid = centroid / norm
    sims = arr @ centroid
    distances = 1.0 - sims
    mean_dist = float(distances.mean())
    return math.log(CENTROID_VARIANCE_FLOOR / max(mean_dist, EPS_LOG))


