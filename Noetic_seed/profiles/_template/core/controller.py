"""Controller（制御層）+ controller_select + intent-conditioned scoring"""
import re
import random
import time  # V07.5 commit 3: debug metric timing (hot path I/O 計測)
from core.config import WORLD_MODEL_CFG
from core.state import load_pref, merge_log_view
from core.embedding import is_vector_ready, _embed_sync, cosine_similarity
from core.eval import predict_result_novelty
from core.predictor import get_predictor
# V07.5 commit 3: argmin G(π) literal selection の literal 依存 module
# (関数内 import から module top-level に literal move、Codex 2 周目 P3 fix bundle)
from core.memory import load_all_subjective_entries, load_all_memories
from core.memory_links import list_links
from core.info_gain import compute_efe_components, CYCLE_KEY
from core.metrics import compute_fog_metrics, _sort_entries_global_newest_first, LINKS_SCAN_LIMIT
from core.jepa_runtime import predict_next_conditioned_batch

# 段階5: Predictor インスタンス (WM 設定に従って初期化、モジュールシングルトン)
_PREDICTOR = get_predictor(WORLD_MODEL_CFG.get("predictor_mode", "light"))


def controller(state: dict, tools_dict: dict, level_tools: dict) -> dict:
    """E値とenergyから構造的制約を導出。ツール解放レベルを判定。

    段階13 Phase 6.1 (2026-05-04): exec_code / create_tool / 旧 self_modify
    撤廃に伴い signature 簡素化 (ai_created_tools / dangerous_patterns /
    run_ai_tool_fn 引数撤去)、sandbox/tools/ 動的 load 機構 + Level 4-6
    昇格 logic を削除。Level 0-3 のみ残し、Level 3 = TOOLS 全 + bash + claw 系
    を最終解放とする。
    """
    energy = state.get("energy", 50)
    # 段階13 Phase 0.1.D: tool 別 e2 平均で raw (tool/result) と subjective (e2)
    # の両層 field を同時参照するため merge view を使う
    log = merge_log_view(state)

    # --- ツール順序: 各ツールの過去E2平均で並べる ---
    tool_e2 = {}
    for entry in log:
        tool = entry.get("tool", "")
        m = re.search(r'(\d+)%', str(entry.get("e2", "")))
        if m and tool in tools_dict:
            tool_e2.setdefault(tool, []).append(int(m.group(1)))
    tool_avg = {t: sum(vs) / len(vs) for t, vs in tool_e2.items() if vs}
    for t in tools_dict:
        if t not in tool_avg:
            tool_avg[t] = 50

    pref = load_pref()
    for t in tools_dict:
        if t in pref:
            tool_avg[t] = round(min(100, max(0, tool_avg[t] * (pref[t] / 50.0))), 1)

    ranked = sorted(tools_dict.keys(), key=lambda t: tool_avg[t], reverse=True)

    # --- tool_level による段階解放 (Level 0-3、Phase 6.1 で 0-6 から縮約) ---
    fr = set(state.get("files_read", []))
    fw = set(state.get("files_written", []))
    lv = state.get("tool_level", 0)
    new_lv = lv
    if lv == 0 and len(fr) >= 1:
        new_lv = 1
    elif lv == 1 and len(fr) >= 2:
        new_lv = 2
    elif lv == 2 and len(fr) >= 1 and len(fw) >= 1 and len(fr) + len(fw) >= 5:
        new_lv = 3

    allowed = set(level_tools[new_lv])

    # Xセッションがなければ X系ツールを除外
    from tools.x_tools import X_SESSION_PATH
    if not X_SESSION_PATH.exists():
        _x_tools = {"x_post", "x_reply", "x_timeline", "x_search", "x_quote", "x_like", "x_get_notifications"}
        allowed -= _x_tools

    # 段階11-D Phase 0 Step 0.4: memory_graph affordance ガード (B2)
    # 自発 memory_store ≥ 1 経験で candidate 解除 (Y1 確定: description は据え置き、
    # 候補に出ても allowed_tools フィルタで弾かれる "中間状態" を試行で気づく設計)
    #
    # Step 0.4 hotfix (2026-04-26): description は tools 一覧に残す + 選択候補からのみ
    # フィルタする PLAN §5 Step 0.4 literal を実装するため、displayed_tools (prompt 用、
    # affordance ガード前) と allowed_tools (parser 用、ガード後) を分離。逆 bootstrap
    # loop 回避 (description は LLM① に表示される、存在認知は可能、選択時のみ却下)。
    displayed_tools = set(allowed)
    if state.get("voluntary_memory_store_count", 0) < 1:
        allowed.discard("memory_graph")

    return {
        "allowed_tools": allowed,           # parser 用 (affordance 適用後)
        "displayed_tools": displayed_tools,  # prompt 用 (affordance 適用前)
        "tool_rank": {t: round(tool_avg[t], 1) for t in ranked},
        "tool_level": new_lv,
        "tool_level_prev": lv,
    }


# ============================================================
# 段階14 Step B: Attractor Cosine Redundancy Detection
# ============================================================
#
# STAGE14 PLAN §4 の実装。LLM① 候補中で「同 tool / 異 reason」を別経路
# attack と扱わず、attractor 空間の縮退として検出する。Spisak & Friston 2025
# の Self-orthogonalizing Attractor Networks 直交性 FEP 帰結を候補生成段階に
# 適用、cluster 縮退で weight 圧縮 = 罰でなく FEP 自然帰結。
#
# PLAN §4 literal の `_embedding` / `_cosine` は概念呼称、実装は controller.py
# 既存 pattern (core.embedding._embed_sync + cosine_similarity) を流用。

REDUNDANCY_THRESHOLD = 0.85  # PLAN §4-2 / §10-7 確定値 (entity_resolver SAME_THRESHOLD 整合)


def _detect_attractor_redundancy(
    candidates: list,
    threshold: float = REDUNDANCY_THRESHOLD,
) -> dict:
    """候補の reason embedding を cosine 比較し、redundancy cluster を返す。

    PLAN §4-2 literal: 同 tool / 異 reason を別経路と扱わず attractor 空間の
    縮退として検出。union-find で cluster 数算出、diversity_score = cluster
    数 / N で controller_select の weight 圧縮入力とする。

    Args:
        candidates: LLM① の候補 list、各 dict に "tool" / "reason" key 想定
        threshold: cosine 同一視の閾値 (default 0.85)

    Returns:
        {
            "redundancy_pairs": [(i, j, cosine), ...],  # 閾値超ペア
            "cluster_count": int,                        # 独立 cluster 数 (ideal=N)
            "redundant_tool_set": set[str],              # 縮退 tool 名
            "diversity_score": float,                    # cluster_count / N (0.2-1.0)
        }

    Embedding 未起動 / 失敗時は redundancy なし扱い (cluster_count=N,
    diversity=1.0) で fallback、controller_select の補正は発火しない。
    """
    n = len(candidates)
    if n == 0:
        return {
            "redundancy_pairs": [],
            "cluster_count": 0,
            "redundant_tool_set": set(),
            "diversity_score": 0.0,
        }

    # Embedding 未起動 → redundancy 検出 skip (構造 fallback、PLAN §1 LLM as
    # brain 整合)
    if not is_vector_ready():
        return {
            "redundancy_pairs": [],
            "cluster_count": n,
            "redundant_tool_set": set(),
            "diversity_score": 1.0,
        }

    reasons = [str(c.get("reason", "")) for c in candidates]
    vecs = _embed_sync(reasons)
    # Codex review P2-b fix: malformed (length 不一致) vector list でも fallback
    if not vecs or len(vecs) != n:
        return {
            "redundancy_pairs": [],
            "cluster_count": n,
            "redundant_tool_set": set(),
            "diversity_score": 1.0,
        }

    # 閾値超 cosine pair 抽出
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            cos = cosine_similarity(vecs[i], vecs[j])
            if cos > threshold:
                pairs.append((i, j, cos))

    # union-find ライト (PLAN §4-2 literal、5 候補で再帰深度安全)
    parent = list(range(n))

    def find(x):
        return x if parent[x] == x else find(parent[x])

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j, _ in pairs:
        union(i, j)
    cluster_count = len(set(find(x) for x in range(n)))

    # 縮退 tool 集合: 同 tool 名の pair が縮退してる場合を記録
    redundant_tools = set()
    for i, j, _ in pairs:
        if candidates[i].get("tool") == candidates[j].get("tool"):
            tool_name = candidates[i].get("tool")
            if tool_name:
                redundant_tools.add(tool_name)

    return {
        "redundancy_pairs": pairs,
        "cluster_count": cluster_count,
        "redundant_tool_set": redundant_tools,
        "diversity_score": cluster_count / max(1, n),
    }


def _intent_conditioned_scores(candidates: list, state: dict) -> list:
    """候補ごとに、過去の類似intent×同toolのE2加重平均を返す。"""
    # 段階13 Phase 0.1.D: intent (subj) + tool (raw) + e2 (subj) を同時参照のため merge
    log = merge_log_view(state)
    if not log or not is_vector_ready():
        return [50.0] * len(candidates)

    past = []
    for e in log:
        intent = e.get("intent", "")
        tool = e.get("tool", "")
        m = re.search(r'(\d+)%', str(e.get("e2", "")))
        if intent and tool and m:
            past.append({"intent": intent, "tool": tool, "e2": int(m.group(1))})
    if not past:
        return [50.0] * len(candidates)

    candidate_texts = [c.get("reason", "") or c.get("tool", "") for c in candidates]
    past_texts = [p["intent"] for p in past]
    all_texts = candidate_texts + past_texts
    all_vecs = _embed_sync(all_texts)
    if not all_vecs or len(all_vecs) != len(all_texts):
        return [50.0] * len(candidates)

    nc = len(candidates)
    cand_vecs = all_vecs[:nc]
    past_vecs = all_vecs[nc:]

    scores = []
    for i, c in enumerate(candidates):
        tool = c["tool"]
        weighted_sum = 0.0
        weight_total = 0.0
        for j, p in enumerate(past):
            if p["tool"] != tool:
                continue
            sim = cosine_similarity(cand_vecs[i], past_vecs[j])
            if sim > 0.3:
                weighted_sum += sim * p["e2"]
                weight_total += sim
        if weight_total > 0:
            scores.append(weighted_sum / weight_total)
        else:
            scores.append(50.0)
    return scores


def _pending_priority_boost(state: dict, candidate: dict) -> float:
    """candidate が未消化 UPS v2 pending を解消しそうなら boost 倍率を返す。

    UPS v2 (Phase 4 Step E-1): 未 observed な pending のうち、
      - source_action == candidate_tool (直接対応)
      - または output_display × device channel (対等協力者への応答)
    にマッチするものの priority 最大値から倍率を算出。

    Returns:
        1.0 〜 3.0 の倍率。該当 pending なしなら 1.0。
    """
    pending = state.get("pending", [])
    candidate_tool = candidate.get("tool", "")
    if not candidate_tool:
        return 1.0

    max_pri = 0.0
    for p in pending:
        if p.get("type") != "pending":
            continue
        if p.get("observed_content") is not None:
            continue
        source = p.get("source_action", "")
        direct_match = source == candidate_tool
        device_response_match = (
            candidate_tool == "output_display"
            and (p.get("observed_channel") == "device"
                 or p.get("expected_channel") == "device")
        )
        if direct_match or device_response_match:
            pri = float(p.get("priority", 0.0))
            if pri > max_pri:
                max_pri = pri

    if max_pri <= 0.0:
        return 1.0
    # priority 想定最大 ≈ 12.0 → [1.0, 3.0] に正規化
    return 1.0 + min(2.0, max_pri / 6.0)


# ============================================================
# 段階5: channel_mismatch + predicted_outcome ペナルティ
# (STAGE5_IMPLEMENTATION_PLAN.md §4)
# ============================================================

def _channel_mismatch_multiplier(candidate: dict, state: dict, cfg: dict) -> float:
    """候補 tool の channel が pending の channel と一致しないなら
    multiplier (<1.0) を返す。internal tool (channel=None) は影響なし (1.0)。
    pending が channel を持たなければ 1.0。
    副作用: 減点時に candidate["penalties"] に理由を追記。
    """
    # 循環 import 回避 (world_model は controller を import する経路がないため
    # 実害はないが、他の controller 拡張との対称性のため関数内 import)
    from core.world_model import get_tool_channel

    wm = state.get("world_model")
    tool_name = candidate.get("tool", "")
    tool_channel = get_tool_channel(wm, tool_name)
    if tool_channel is None:
        return 1.0
    pending_channels = set()
    for p in state.get("pending", []):
        if p.get("type") != "pending":
            continue
        if p.get("observed_content") is not None:
            continue
        for field in ("observed_channel", "expected_channel"):
            v = p.get(field)
            if v:
                pending_channels.add(v)
    if not pending_channels or tool_channel in pending_channels:
        return 1.0
    mult = float(cfg.get("channel_mismatch_multiplier", 0.5))
    candidate.setdefault("penalties", []).append(
        f"channel_mismatch: {tool_channel} not in {sorted(pending_channels)}"
    )
    return mult


def _predicted_outcome_multiplier(prediction: dict, candidate: dict,
                                  state: dict, cfg: dict) -> float:
    """predicted_e2 / predicted_ec を tool 別自己学習重みで正規化加重した
    連続値 multiplier を返す (段階9 → 段階10 柱 C 拡張)。

    段階9: predicted_e2 直接乗算 (pragmatic value of EFE)。
    段階10 柱 C: predicted_ec (0.0-1.0) 併用 + state.predictor_confidence の
    {e2_conf, ec_conf} で正規化加重。tool 別に「どの軸を信頼するか」が
    β+ で自己学習され、当たる軸の weight が自動で上がる。

    挙動:
    - predicted_ec あり + 正常 conf: (pe2*e2_conf + pec*ec_conf) / (e2_conf+ec_conf)
    - predicted_ec 欠如 (Light fallback 等): pe2_ratio のみ (段階9 挙動)
    - 両 conf ゼロ: pass-through 1.0 (predictor 層 bypass、PLAN v1 §3-C 確定)
    - combined < 0.4: candidate["penalties"] に理由追記 (解釈可能性)
    - cfg["predicted_e2_floor"] で下限 (default 0.05) 上書き可、不正値は 50 neutral
    """
    pe2_raw = prediction.get("predicted_e2", 50) if isinstance(prediction, dict) else 50
    try:
        pe2 = max(0, min(100, int(pe2_raw)))
    except (TypeError, ValueError):
        pe2 = 50
    pe2_ratio = pe2 / 100.0

    pec_raw = prediction.get("predicted_ec") if isinstance(prediction, dict) else None

    # 段階10 柱 C: tool 別自己学習重みを参照
    tool_name = str(candidate.get("tool", ""))
    pc_entry = state.get("predictor_confidence", {}).get(
        tool_name, {"e2_conf": 0.7, "ec_conf": 0.7}
    )
    pe2_w = float(pc_entry.get("e2_conf", 0.7))
    pec_w = float(pc_entry.get("ec_conf", 0.7))
    total_w = pe2_w + pec_w

    # 両 conf ゼロ fallback (PLAN v1 §3-C): predictor 層 bypass
    if total_w <= 0:
        return 1.0

    if pec_raw is not None:
        try:
            pec = max(0.0, min(1.0, float(pec_raw)))
        except (TypeError, ValueError):
            pec = 0.5
        combined = (pe2_ratio * pe2_w + pec * pec_w) / total_w
    else:
        # predicted_ec 欠如 (Light fallback や古い LLM 応答) は段階9 挙動維持
        combined = pe2_ratio

    # 段階14 Step C: β を pragmatic value に逆作用 (β 高 = pragmatic 抑制)、
    # Curiosity is Knowledge 2026 の lower bound 不等式直訳 (PLAN §5-2 literal)。
    from core.predictor import _compute_dynamic_beta
    beta = _compute_dynamic_beta(state, candidate)
    floor = float(cfg.get("predicted_e2_floor", 0.05))
    mult = max(floor, combined / beta)
    if combined < 0.4:
        candidate.setdefault("penalties", []).append(
            f"low_outcome={round(combined, 3)} beta={round(beta, 2)}"
        )
    return mult


def controller_select(candidates: list, ctrl: dict, state: dict) -> dict:
    """V07.5 commit 3: argmin_π G(π) literal selection (paradigm shift 核心)。

    旧 multiplicative chain (tool_rank / intent_scores / sharpness / novelty /
    _pending_priority_boost / _channel_mismatch_multiplier / _predicted_outcome_multiplier
    / attractor_redundancy 圧縮 / random.random sampling) を literal 撤去、
    EFE 9 成分 (commit 2.5 統一 signature) + JEPA predict_next_conditioned_batch (commit 1)
    で **argmin_π G(π)** literal selection 達成 (PLAN v1.4 §3-1 + §4-1 commit 3 literal)。

    撤去 5 経路 (tool_rank / intent_scores / _pending_priority_boost /
    _channel_mismatch_multiplier / attractor_redundancy 圧縮) の **機構自体は維持** =
    selection multiplier から外すのみ。各機構の state field 化 (telemetry) は保持、
    selection には literal 流さない (Codex 軸 1 OK + 軸 4 GAP literal 反映)。

    natural emergence: centroid_stuck → attractor_redundancy のみ literal 確認済
    (commit 2.5 `_centroid_stuck` 経由)、他 4 経路 (pending/channel/tool_rank/intent) は
    EFE 9 成分に未接続 = smoke 観察対象 (PLAN v1.4 §8-5 literal):
    - pending 解消遅延 / channel 応答外れ / tool 選択不安定化 を smoke で監視
    - 実害化時 hotfix (該当経路を G の項に literal 統合、別 commit)

    撤去 1 経路 (_predicted_outcome_multiplier、pressure 由来) は A3 literal で完全撤去
    (cycle 55 attractor 真因の中核解消、PLAN §1-3 literal)。
    pressure 機構自体は維持 (`main.py:1202, 1211` の calc_pressure_signals + 累積式、
    `reflection.py:64-82` 前倒し reflection、内発駆動 device、Codex AXIS 3 Blocker 死守)。

    Codex commit 2.5 P3 #1 literal 反映: shallow copy `dict(state)` で `_efe_*_view`
    一時 field の元 state pollution 回避。C None graceful skip (cycle 1 bootstrap
    依存ガード、commit 4 並走不要、Codex 軸 7 GAP literal 反映)。

    Args / Returns: 旧 signature 維持 (candidates / ctrl / state)、後方互換 wrapper 経路。
    """
    # 機構保持 (state field 化のみ、selection には流さない、telemetry 用途)
    intent_scores = _intent_conditioned_scores(candidates, state)
    for i, c in enumerate(candidates):
        c["_intent_score"] = intent_scores[i]  # telemetry only

    # 段階14 Step B: attractor redundancy 検出 + state field 化 (telemetry)、
    # selection 圧縮乗算は撤去 (natural emergence: centroid_stuck → attractor_redundancy literal)
    redundancy = _detect_attractor_redundancy(candidates)
    state["last_redundancy"] = {
        "redundancy_pairs": [list(p) for p in redundancy["redundancy_pairs"]],
        "cluster_count": redundancy["cluster_count"],
        "redundant_tool_set": sorted(redundancy["redundant_tool_set"]),
        "diversity_score": redundancy["diversity_score"],
    }

    # predictor.predict (state field 化保持、selection 経路の predicted_outcome_multiplier
    # は A3 literal 撤去、predicted_e2 / predicted_ec は main.py で prediction_error 計算継続)
    for c in candidates:
        prediction = _PREDICTOR.predict(c, state, state.get("world_model"))
        c["predicted_outcome"] = prediction
        c["_predicted_e2"] = (
            prediction.get("predicted_e2", 50) if isinstance(prediction, dict) else 50
        )
        c["_predicted_ec"] = (
            prediction.get("predicted_ec") if isinstance(prediction, dict) else None
        )

    # V07.5 commit 3: argmin_π G(π) literal selection
    # 1. entries / links / fog_now を取得 (metrics.py と同経路、newest first sort)
    #
    # V07.5 commit 3 debt track + observability (Codex review 1 周目 P2-2 reference、
    # commit 4/6 で cycle-level cache 経路追加予定): 本 hot path I/O は metrics.py
    # cycle 末計算と literal 重複 (両者で load_all_subjective_entries +
    # load_all_memories + list_links + compute_fog_metrics を別タイミングで実行)。
    # `_v07_5_io_elapsed_ms` で計測して smoke 観察可能化、実害化時 (cycle 時間延伸
    # literal 観察) は cycle-level cache に commit 4/6 で経路追加。
    _io_t0 = time.time()
    entries_with_embedding = [
        e for e in load_all_subjective_entries()
        if isinstance(e.get("embedding"), list)
    ] + [
        e for e in load_all_memories()
        if isinstance(e.get("embedding"), list)
    ]
    _sort_entries_global_newest_first(entries_with_embedding)
    links = list_links(limit=LINKS_SCAN_LIMIT)
    fog_now = compute_fog_metrics(state, entries_with_embedding, links)
    state["_v07_5_io_elapsed_ms"] = round((time.time() - _io_t0) * 1000, 2)
    prev_snapshot = state.get(CYCLE_KEY, {}) or {}

    # 2. JEPA predict_next_conditioned_batch (commit 1) で各 candidate の predicted_embedding 取得
    # graceful skip: torch 未 install / bge-m3 未起動 / sequence 不足で None
    predicted_embeddings = predict_next_conditioned_batch(state, candidates)

    # 3. 各 candidate の G(π) literal 計算
    # C None graceful skip: state["_efe_C"] 未構築 cycle 1 で compute_efe_components が
    # pragmatic value 経路を 0.0 で graceful return、G 計算継続 (commit 4 並走不要)
    g_values = []
    for i, c in enumerate(candidates):
        pred_emb = (
            predicted_embeddings[i]
            if (predicted_embeddings is not None and i < len(predicted_embeddings))
            else None
        )
        hypothetical_next = {
            "predicted_embedding": pred_emb,
            "candidate": c,
        }
        efe = compute_efe_components(
            state,
            prev_snapshot,
            entries_with_embedding=entries_with_embedding,
            links=links,
            fog_now=fog_now,
            hypothetical_next=hypothetical_next,
        )
        g_values.append(efe.get("G", 0.0))
        c["_efe_G"] = efe.get("G", 0.0)  # telemetry: 各 candidate の G 値を保持

    # 4. argmin_π G(π) literal selection (Friston FEP literal、最小化対象)
    if not g_values:
        return candidates[-1]
    min_idx = g_values.index(min(g_values))

    # V07.5 commit 3 P2-1 fix (Codex 1 周目 literal): 全 G 値タイ縮退検出
    # JEPA 未起動 + 全候補同 tool 等で各 helper が同一 graceful 0.0 を返す literal case では
    # g_values が全タイ → argmin が first-candidate 任意選択 literal 縮退。
    # smoke で degraded=True が頻発する場合は paradigm shift literal の前提が崩れてる
    # signal (= candidates 生成側 / JEPA / EFE 計算経路の literal 不全)、hotfix 必要。
    g_min = min(g_values)
    g_max = max(g_values)
    g_spread = g_max - g_min
    g_degraded = g_spread < 1e-6  # tie 検出 threshold (numerical noise level literal)

    # selection log 雛形 (commit 5 で C_version / policy_snapshot_id / basin_snapshot_id 拡張)
    state["last_selection_log"] = {
        "g_values": [round(g, 6) for g in g_values],
        "selected_idx": min_idx,
        "selected_tool": candidates[min_idx].get("tool", ""),
        "selection_method": "argmin_G",  # V07.5 paradigm shift literal marker
        "degraded": g_degraded,  # P2-1 fix: 全 G タイ縮退 literal observability
        "g_spread": round(g_spread, 6),  # G 値の literal な散らばり (smoke 観察用)
    }

    return candidates[min_idx]
