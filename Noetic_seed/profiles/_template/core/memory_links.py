"""memory_links — 段階11-B Phase 4 (A-MEM NeurIPS 2025 準拠)。

memory entry 間に関係性 link を LLM judge で生成、Zettelkasten 形式の graph 化。
既存 entity facts (段階4) とは別 layer として並立、データ重複なし。

link_type 候補:
  既存 5 type (11-B Phase 4): similar / contradict / elaborate / causal / temporal
  追加 3 type (11-D Phase 2): co_activation / semantic / supporting
    - semantic は本 Phase 2 から LLM judge 経由で生成可能
    - co_activation / supporting は LINK_TYPES 受け入れのみ、実 generation hook
      は Phase 4 (predictive coding 接続時) で実装
confidence 閾値: 0.7 以上のみ保存 (link 爆発防止、escape hatch で tune 可)
link 生成タイミング: memory_store 同期 + top-K=5 近傍のみ LLM judge
(Phase 3 keywords 同期と一貫性)

11-D Phase 2 schema 拡張: link_entry に strength / last_used / usage_count
3 field 追加 (Phase 3 Physarum update の前準備、§-1 v1「initial strength =
confidence」自然 migration)。
"""
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from core.config import MEMORY_DIR


LINK_FILE_NAME = "memory_links.jsonl"
LINK_CONFIDENCE_THRESHOLD = 0.7      # Phase 4 Step 4.2: これ未満は discard
LINK_GENERATION_TOP_K = 5            # Phase 4 Step 4.3: 近傍 top-K のみ judge
LINK_TYPES = (
    # 既存 5 type (11-B Phase 4)
    "similar", "contradict", "elaborate", "causal", "temporal",
    # 11-D Phase 2 追加 3 type (storage 受け入れ + semantic は LLM judge 候補)
    "co_activation", "semantic", "supporting",
)

# 11-D Phase 3 (Physarum strength update、Session W v1 確定):
# 数式 strength[t+1] = strength[t] * (1 - β) + α * usage[t] (EMA with decay 同型)
# 根拠: Tero et al. 2007 + Sun 2017 review + RL/EMA 典型値
PHYSARUM_ALPHA = 0.1                 # up rate (Tero 2007 + EMA 典型値 0.01-0.1 上限)
PHYSARUM_BETA = 0.05                 # decay rate (Tero 2007 + 半減期 14 cycle)
STRENGTH_CAP = 1.0                   # 数学派生 saturate ceiling (α=0.1 で 10 連続 hit で saturate)
PRUNING_STRENGTH_RATIO = 0.15        # pruning threshold = initial × 0.15 (動的相対基準)
# graph 全走査時の link 取得上限 (default list_links limit=200 では不足)。
# 段階11-C G-lite Phase 1 (commit 40867f2c) で確立した慣習値。隠れ根拠は
# 本 module の pruning 機構 (PHYSARUM_BETA + PRUNING_STRENGTH_RATIO) で
# idle link が ≒ 56 cycle で削除される設計 = 実運用で 10000 行頭打ち想定。
# smoke 観察で実値判明したら本定数を更新、全 5 利用箇所 (dynamic_composition /
# tag_emergence_monitor / memory_graph_tool / reflection / 本 module) に一括反映。
LINK_SCAN_LIMIT = 10000


def _compute_pruning_idle_cycles() -> int:
    """pruning idle cycle 閾値 = β 半減期 × 4 (動的派生、literal 定数追加なし)。

    β=0.05 で ≒ 56 cycle。β を変えれば自動再計算 (ゆう ① マジックナンバー回避精神)。
    """
    import math
    return int(round(math.log(0.5) / math.log(1.0 - PHYSARUM_BETA) * 4))


def _link_strength(link: dict) -> float:
    """link の現在 strength を取得 (Phase 2 以前 link の backward compat)。

    Phase 2 以前 (11-B Phase 4 で作られた link) は strength field 欠落、
    confidence 値にフォールバック (initial strength = confidence の自然 migration)。
    """
    return float(link.get("strength", link.get("confidence", 0.0)))


def _apply_lazy_decay(link: dict, current_cycle: int) -> float:
    """case Q lazy decay: last_used_cycle から経過 cycle 分の decay を計算 (read-only)。

    数式: strength_decayed = strength * (1 - β)^elapsed_cycles
    既存 link (last_used_cycle 欠落、Phase 2 以前) は decay skip (current cycle 扱い)、
    Phase 3 以降の新 link は最初の access で last_used_cycle が設定される。

    Returns:
        decay 適用後の strength (link 自体は変更しない)
    """
    strength = _link_strength(link)
    last_cycle = link.get("last_used_cycle")
    if last_cycle is None:
        return strength
    try:
        elapsed = max(0, int(current_cycle) - int(last_cycle))
    except (TypeError, ValueError):
        return strength
    if elapsed == 0:
        return strength
    return strength * ((1.0 - PHYSARUM_BETA) ** elapsed)


def get_link_current_strength(link: dict,
                              current_cycle: Optional[int] = None) -> float:
    """link の現在 strength を取得 (current_cycle 指定時は lazy decay 適用後)。

    v0.5 Phase 5 Slice 4 F-003: memory_graph view 層で decayed_strength を
    露出するための公開 API。私的関数 (_link_strength / _apply_lazy_decay) の
    呼び分けを集約、view-layer から module 越境せず利用可能にする。

    Args:
        link: link entry dict
        current_cycle: 現 cycle 番号。None なら raw strength を返す。

    Returns:
        current_cycle 指定時は lazy decay 適用後の strength (read-only、link 不変)、
        None 時は raw strength。
    """
    if current_cycle is not None:
        try:
            return _apply_lazy_decay(link, current_cycle)
        except Exception:
            pass
    return _link_strength(link)


def _link_file() -> Path:
    MEMORY_DIR.mkdir(exist_ok=True)
    return MEMORY_DIR / LINK_FILE_NAME


def _build_link_prompt(entry_a: dict, entry_b: dict) -> str:
    """link 判定用 LLM prompt (PLAN §5 Phase 4 Step 4.2 + 11-D Phase 2 Step 2.2 準拠、軽量 JSON 出力)。

    11-D Phase 2: semantic (embedding 類似ベース、5 type の generic 化) を
    LLM judge 候補に追加。co_activation / supporting は Phase 4 hook で生成、
    LLM judge では出さない (構造誘導の混乱回避)。
    """
    a_kws = entry_a.get("keywords", []) or []
    b_kws = entry_b.get("keywords", []) or []
    return (
        "以下 2 つの記憶 entry の関係性を判定してください:\n"
        f"\nEntry A: {entry_a.get('content', '')[:300]}\n"
        f"  tag: {entry_a.get('network', '')}, keywords: {a_kws}\n"
        f"Entry B: {entry_b.get('content', '')[:300]}\n"
        f"  tag: {entry_b.get('network', '')}, keywords: {b_kws}\n"
        "\nlink_type 候補:\n"
        "- similar: 類似内容 (重複に近い)\n"
        "- semantic: 意味的に関連する一般概念 (similar より broader)\n"
        "- contradict: 矛盾 (Phase 3 reconciliation と補完関係)\n"
        "- elaborate: 片方が他方を詳述 / 具体化\n"
        "- causal: 原因-結果 / 行動-観察\n"
        "- temporal: 時系列的連続\n"
        "- (none): 関係薄い → link 作らない\n"
        "\n出力は JSON のみ (他の文字を含めない):\n"
        '{"link_type": str, "confidence": float, "reason": str}\n'
        '- link_type は上記 6 種 or "none"\n'
        "- confidence 0.0-1.0、reason は 1 文"
    )


def _parse_link_response(response: str) -> dict:
    """LLM 応答から {link_type, confidence, reason} を抽出 (robust)。

    失敗時 / none 時 / 閾値未満は link 作らない扱い (link_type="none")。
    """
    default = {"link_type": "none", "confidence": 0.0, "reason": ""}
    try:
        m = re.search(r'\{.*\}', response, re.DOTALL)
        if not m:
            return default
        data = json.loads(m.group(0))
        lt = str(data.get("link_type", "none")).strip().lower()
        if lt not in LINK_TYPES:
            lt = "none"
        conf = float(data.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        return {
            "link_type": lt,
            "confidence": conf,
            "reason": str(data.get("reason", ""))[:200],
        }
    except Exception:
        return default


def _llm_judge_link(entry_a: dict, entry_b: dict,
                    llm_call_fn: Optional[Callable] = None) -> dict:
    """2 entry 間の link 判定 (LLM mock 可能、error で graceful fallback)。"""
    if llm_call_fn is None:
        from core.llm import call_llm
        llm_call_fn = call_llm
    prompt = _build_link_prompt(entry_a, entry_b)
    try:
        response = llm_call_fn(prompt, max_tokens=200, temperature=0.2)
        return _parse_link_response(response)
    except Exception as e:
        print(f"  [memory_links] judge skip (error: {e})")
        return {"link_type": "none", "confidence": 0.0, "reason": ""}


def _build_link_entry(from_entry: dict, to_entry: dict, verdict: dict) -> dict:
    """link entry dict 生成 (storage 用)。11-A perspective 属性を付与。

    11-D Phase 2 schema 拡張 (Phase 3 Physarum update 前準備):
    - strength: 初期値 = confidence (PLAN §-1 v1「initial strength = confidence」
      自然 migration、Phase 3 Step 3.2 で動的更新対象)
    - last_used: 初期値 = created_at (Phase 3 Step 3.3 で retrieval 毎に update)
    - usage_count: 初期値 = 0 (Phase 3 Step 3.3 で retrieval 毎に increment)
    """
    from core.perspective import default_self_perspective
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    confidence = float(verdict.get("confidence", 0.0))
    return {
        "id": f"link_{uuid.uuid4().hex[:12]}",
        "from_id": from_entry.get("id", ""),
        "to_id": to_entry.get("id", ""),
        "link_type": verdict.get("link_type", "none"),
        "confidence": confidence,
        "strength": confidence,            # 11-D Phase 2: 初期値 = confidence
        "perspective": default_self_perspective(),
        "created_at": now,
        "last_used": now,                  # 11-D Phase 2: 初期値 = created_at
        "usage_count": 0,                  # 11-D Phase 2: 初期値 = 0
        "reason": verdict.get("reason", ""),
    }


def _append_link(link_entry: dict) -> None:
    """memory_links.jsonl に atomic append。"""
    fpath = _link_file()
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(link_entry, ensure_ascii=False) + "\n")


def update_link_strength_used(link_id: str,
                              current_cycle: Optional[int] = None,
                              prediction_error: Optional[float] = None) -> Optional[dict]:
    """retrieval で使われた link の strength を up + last_used / usage_count update。

    11-D Phase 3 (Physarum rule、案 Q lazy decay):
    1. last_used_cycle から経過 cycle 分の decay を適用 (case Q)
    2. strength += α、上限 STRENGTH_CAP (1.0) で clip
    3. last_used / last_used_cycle / usage_count を update
    4. memory_links.jsonl に書き戻し

    11-D Phase 4 (Predictive coding modulator) + Slice 4 F-001 production wiring:
    - prediction_error (0.0-1.0) が指定されたら strength up を modulate:
      * strength_delta = α * max(0.0, 1.0 - prediction_error)
      * 予測誤差小 (成功) → modulator 1.0 → α そのまま
      * 予測誤差大 (失敗) → modulator 0.0 → strength up ゼロ
    - 段階10 経路 (entropy.record_ec_prediction_error) との接続点。
      values は 0.0-1.0 severity スケールでそのまま使える (clamp あり)
    - production timing = **前 cycle modulation** (Slice 4 設計確定):
      retrieval は cycle 序盤、prediction_error history 書込は cycle 末 (main.py)
      のため、retrieval 時の history[-1] は前 cycle の error。RL/Active Inference
      の TD error / FEP 学習則 (= 過去誤差で現在 policy 更新) の標準パターンと
      整合的。同 cycle modulation 化は cycle 構造改修必要のため採用せず。

    Args:
        link_id: 対象 link の id
        current_cycle: 現在の cycle 番号 (state["cycle_id"])。None なら decay skip
        prediction_error: 0.0-1.0 の severity (段階10 magnitude)。None なら modulator なし

    Returns:
        更新後の link entry (見つからない場合 None)
    """
    fpath = _link_file()
    if not fpath.exists():
        return None
    try:
        lines = fpath.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    # 11-D Phase 4: prediction_error modulator 計算 (段階10 接続)
    if prediction_error is not None:
        try:
            err = max(0.0, min(1.0, float(prediction_error)))
        except (TypeError, ValueError):
            err = 0.0
        strength_delta = PHYSARUM_ALPHA * (1.0 - err)
    else:
        strength_delta = PHYSARUM_ALPHA
    updated_entry = None
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        try:
            link = json.loads(line)
        except Exception:
            new_lines.append(line)
            continue
        if link.get("id") == link_id:
            # 1. lazy decay (案 Q)
            if current_cycle is not None:
                strength = _apply_lazy_decay(link, current_cycle)
            else:
                strength = _link_strength(link)
            # 2. strength up + cap (Phase 4 modulator 適用済の delta を使う)
            strength = min(STRENGTH_CAP, strength + strength_delta)
            link["strength"] = strength
            link["last_used"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if current_cycle is not None:
                link["last_used_cycle"] = int(current_cycle)
            link["usage_count"] = int(link.get("usage_count", 0)) + 1
            updated_entry = link
        new_lines.append(json.dumps(link, ensure_ascii=False))
    if new_lines:
        fpath.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return updated_entry


# ============================================================
# 段階11-D Phase 4 Step 4.2: 新 link 探索 trigger (動的 percentile threshold)
# ============================================================

NEW_LINK_EXPLORATION_PERCENTILE = 90        # 90% percentile (Active Inference epistemic value 系)
NEW_LINK_EXPLORATION_HISTORY_N = 20         # 過去 N cycle のサンプル
NEW_LINK_EXPLORATION_MIN_SAMPLES = 5        # サンプル不足判定
NEW_LINK_EXPLORATION_FALLBACK = 0.7         # サンプル不足時の初期 fallback


def should_explore_new_links(state: dict, current_error: float) -> bool:
    """予測誤差大時に新 link 探索を起動するか判定 (Phase 4 Step 4.2、動的 percentile)。

    Session W v1 確定 (動的派生、literal 定数なし):
    過去 N=20 cycle の prediction_error 分布の 90% percentile を threshold とする。
    サンプル不足 (< 5) は initial fallback 0.7 (Active Inference epistemic value 系
    + anomaly detection 分野の確立手法、固定値より状況適応性が高い)。

    実 generation 発火 (LLM judge で新 link 探索) は呼び手側で判断、本関数は
    trigger の boolean のみ返す。reflect 継続原則。

    Args:
        state: state dict (state["prediction_error_history_ec"] を参照)
        current_error: 現在の prediction error (0.0-1.0 severity)

    Returns:
        True: 探索 trigger (current_error > 動的 threshold)
    """
    history = state.get("prediction_error_history_ec", []) if isinstance(state, dict) else []
    recent = list(history)[-NEW_LINK_EXPLORATION_HISTORY_N:]
    if len(recent) < NEW_LINK_EXPLORATION_MIN_SAMPLES:
        threshold = NEW_LINK_EXPLORATION_FALLBACK
    else:
        sorted_recent = sorted(float(x) for x in recent)
        # 90% percentile (linear interpolation 不要、近似で十分)
        idx = int(len(sorted_recent) * (NEW_LINK_EXPLORATION_PERCENTILE / 100.0))
        idx = min(idx, len(sorted_recent) - 1)
        threshold = sorted_recent[idx]
    try:
        cur = max(0.0, min(1.0, float(current_error)))
    except (TypeError, ValueError):
        return False
    return cur > threshold


def prune_weak_links(current_cycle: int) -> int:
    """全 link を走査、低 strength + idle 長 の link を削除。

    11-D Phase 3 Step 3.5 (Session W v1 確定、両方とも動的派生):
    - strength threshold = confidence × 0.15 (initial strength = confidence の relative ratio)
    - idle threshold = β 半減期 × 4 ≒ 56 cycle (β からの完全派生)

    呼出タイミング: cycle 終端 batch (案 P 系の sweep 用、lazy decay 案 Q と並存)。
    main.py の cycle loop で `pending_prune` と並んで呼ぶ運用想定。

    Args:
        current_cycle: 現在の cycle 番号 (state["cycle_id"])

    Returns:
        削除した link 件数
    """
    fpath = _link_file()
    if not fpath.exists():
        return 0
    try:
        lines = fpath.read_text(encoding="utf-8").splitlines()
    except Exception:
        return 0
    idle_threshold = _compute_pruning_idle_cycles()
    kept = []
    removed = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            link = json.loads(line)
        except Exception:
            kept.append(line)
            continue
        # initial strength = confidence (PLAN §-1 v1 自然 migration)
        initial = float(link.get("confidence", 0.0))
        strength_threshold = initial * PRUNING_STRENGTH_RATIO
        # 現在の有効 strength (lazy decay 込み)
        current_strength = _apply_lazy_decay(link, current_cycle)
        last_cycle = link.get("last_used_cycle")
        if last_cycle is None:
            idle = 0   # last_used_cycle 欠落 link は idle 0 扱い (Phase 2 以前 / 新規 link 保護)
        else:
            try:
                idle = max(0, int(current_cycle) - int(last_cycle))
            except (TypeError, ValueError):
                idle = 0
        if current_strength < strength_threshold and idle >= idle_threshold:
            removed += 1
            continue
        kept.append(json.dumps(link, ensure_ascii=False))
    if removed:
        fpath.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    return removed


def list_links(limit: int = 200) -> list:
    """memory_links.jsonl から新しい順に limit 件読む。"""
    fpath = _link_file()
    if not fpath.exists():
        return []
    try:
        lines = fpath.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def generate_links_for(new_entry: dict, *,
                       top_k: int = LINK_GENERATION_TOP_K,
                       embed_fn: Optional[Callable] = None,
                       cosine_fn: Optional[Callable] = None,
                       llm_call_fn: Optional[Callable] = None,
                       confidence_threshold: float = LINK_CONFIDENCE_THRESHOLD,
                       candidate_limit: int = 50) -> list:
    """新 memory entry の近傍 top-K に対して link 生成 (memory_store 同期呼出想定)。

    PLAN §5 Phase 4 Step 4.3: memory_store 同期呼出で embedding 近傍 top-K を
    LLM judge、confidence >= threshold のみ memory_links.jsonl に append。

    Args:
        new_entry: 新規 memory entry
        top_k: 近傍数 (PLAN 推奨 5、cost / 密度バランス)
        embed_fn / cosine_fn: None で近傍取得 skip (= link 生成 skip)
        llm_call_fn: None で core.llm.call_llm 使用
        confidence_threshold: 閾値 (PLAN 推奨 0.7、smoke 後 tune 可能)
        candidate_limit: 同 network から走査する候補上限

    Returns:
        [link_entry, ...] 生成した link の list (smoke 分析用、実運用は副作用)
    """
    from core.memory import list_records

    network = new_entry.get("network", "")
    new_id = new_entry.get("id", "")
    if not network:
        return []

    # embed_fn 未指定なら近傍取得不可 → link 生成 skip
    if embed_fn is None or cosine_fn is None:
        return []

    all_records = list_records(network, limit=candidate_limit)
    candidates = [r for r in all_records if r.get("id") != new_id]
    if not candidates:
        return []

    new_content = new_entry.get("content", "")
    if not new_content:
        return []

    # embedding で近傍 top-K 取得
    try:
        vecs = embed_fn([new_content] + [c.get("content", "") for c in candidates])
    except Exception:
        return []
    if not vecs or len(vecs) != 1 + len(candidates):
        return []

    query_vec = vecs[0]
    sims = []
    for i, c in enumerate(candidates):
        try:
            sim = float(cosine_fn(query_vec, vecs[i + 1]))
        except Exception:
            continue
        sims.append((c, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    near = sims[:top_k]

    # 各近傍に対して LLM judge
    created = []
    for cand, _sim in near:
        verdict = _llm_judge_link(new_entry, cand, llm_call_fn=llm_call_fn)
        if verdict.get("link_type", "none") == "none":
            continue
        if verdict.get("confidence", 0.0) < confidence_threshold:
            continue
        link_entry = _build_link_entry(new_entry, cand, verdict)
        _append_link(link_entry)
        created.append(link_entry)
    return created


# ============================================================
# v0.5 Phase 5 Slice 4 F-002: co_activation 構造的生成 hook
# ============================================================
# 11-D Phase 4 で should_explore_new_links (trigger boolean のみ) を実装、
# 本 Slice で「実発火 = retrieved memory pair から co_activation link を
# 構造的生成」を配線。LLM 通さず、「同時 retrieval された」事実を構造的に記録。
# semantic / supporting 自動発火は次 pass (PLAN_CODE_TRACEABILITY §3.1
# F-002 conservative first pass literal)。

CO_ACTIVATION_INITIAL_CONFIDENCE = 0.7      # LINK_CONFIDENCE_THRESHOLD と整合 (Codex P1 fix
                                              # 2026-05-10): 構造的事実 (同時 retrieval) は LLM
                                              # judge 推測より信頼性高、0.7 = follow_links 走査
                                              # 閾値以上で retrieval 経路から実際に参照される。
                                              # Physarum decay 余地 (1.0 cap まで 0.3) 確保。
CO_ACTIVATION_MAX_PAIRS_PER_CALL = 10        # 1 回呼出で生成する link 上限 (cost 制御)
CO_ACTIVATION_TOP_N_MEMORIES = 5             # pair 候補 memory 上位件数 (5C2 = 10 pair)


def generate_co_activation_links(memories: list,
                                 current_cycle: Optional[int] = None,
                                 *,
                                 max_pairs: int = CO_ACTIVATION_MAX_PAIRS_PER_CALL,
                                 top_n: int = CO_ACTIVATION_TOP_N_MEMORIES) -> list:
    """retrieval で同時取得された memory 群から co_activation link を構造的生成。

    11-D Phase 4 → Slice 4 F-002: should_explore_new_links が True の時 (high
    prediction error) に呼ばれる hook。LLM 通さず、構造的事実 (= 同時 retrieval
    されたペア) として co_activation link を memory_links.jsonl に append。

    co_activation は対称的関係 (= 同時 retrieval された事実は方向不変)、ただし
    既存 follow_links は outgoing しか辿らないため、生成時に **双方向 2 本**
    (A→B + B→A) を同時 append してデータ層で対称性を表現する (Codex Slice 4
    review P1-3 ゆう案採用、2026-05-10)。dedup は frozenset で対称的に集約、
    1 pair = 2 link として扱う (max_pairs は pair 単位 cap)。strength up は
    別経路 update_link_strength_used で各 link 個別に走る (対称関係の対称強化)。

    Args:
        memories: retrieval 結果の memory entry list (network_mems 上位想定)
        current_cycle: 現 cycle (link last_used_cycle / reason に記録)
        max_pairs: 本呼出で生成する pair 上限 (cost 制御、1 pair = 2 link)
        top_n: 生成対象とする memory 上位件数 (top_n × (top_n-1) / 2 pair)

    Returns:
        生成した link entry list (1 pair につき 2 link)
    """
    if not memories or len(memories) < 2:
        return []
    candidate_mems = memories[:top_n]
    existing_pairs = set()
    for l in list_links(limit=LINK_SCAN_LIMIT):
        if l.get("link_type") == "co_activation":
            f = l.get("from_id", "")
            t = l.get("to_id", "")
            if f and t:
                existing_pairs.add(frozenset((f, t)))
    from core.perspective import default_self_perspective
    created = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    perspective = default_self_perspective()
    pairs_created = 0
    for i in range(len(candidate_mems)):
        if pairs_created >= max_pairs:
            break
        for j in range(i + 1, len(candidate_mems)):
            if pairs_created >= max_pairs:
                break
            id_a = candidate_mems[i].get("id", "")
            id_b = candidate_mems[j].get("id", "")
            if not id_a or not id_b or id_a == id_b:
                continue
            pair_key = frozenset((id_a, id_b))
            if pair_key in existing_pairs:
                continue
            # 双方向 2 本 (A→B + B→A) 同時 append (P1-3 ゆう案、データ層対称性)
            for from_id, to_id in ((id_a, id_b), (id_b, id_a)):
                link_entry = {
                    "id": f"link_{uuid.uuid4().hex[:12]}",
                    "from_id": from_id,
                    "to_id": to_id,
                    "link_type": "co_activation",
                    "confidence": CO_ACTIVATION_INITIAL_CONFIDENCE,
                    "strength": CO_ACTIVATION_INITIAL_CONFIDENCE,
                    "perspective": perspective,
                    "created_at": now,
                    "last_used": now,
                    "last_used_cycle": int(current_cycle) if current_cycle is not None else None,
                    "usage_count": 0,
                    "reason": (f"co-retrieved at cycle {current_cycle}"
                               if current_cycle is not None else "co-retrieved"),
                }
                _append_link(link_entry)
                created.append(link_entry)
            existing_pairs.add(pair_key)
            pairs_created += 1
    return created


# ============================================================
# 段階11-C G-lite Phase 1: retrieval 拡張 (follow_links)
# ============================================================

LINK_TRAVERSAL_MAX_DEPTH_DEFAULT = 1   # G-lite 推奨 depth=1、smoke 後 tune 余地
LINK_TRAVERSAL_TOP_N_DEFAULT = 3       # depth 毎の上位 N 件 (cost / 密度 balance)


def follow_links(
    node_id: str,
    *,
    depth: int = LINK_TRAVERSAL_MAX_DEPTH_DEFAULT,
    link_types: Optional[tuple] = None,
    min_confidence: float = LINK_CONFIDENCE_THRESHOLD,
    top_n_per_depth: int = LINK_TRAVERSAL_TOP_N_DEFAULT,
    visited: Optional[set] = None,
) -> list:
    """指定 memory entry から link graph を traverse して近傍 entry を返す.

    段階11-C G-lite Phase 1: 既存 `memory_links.jsonl` (storage、Phase 4 実装)
    を retrieval 経路で活用、A-MEM の top-k similarity + link traversal 戦略。

    Args:
        node_id: 起点 memory entry の id
        depth: traversal 最大深さ (G-lite 推奨 1)
        link_types: 追従する link_type tuple (None で全 type)
        min_confidence: 未満の link は辿らない (G-lite 推奨 0.7 = 既存 storage 閾値)
        top_n_per_depth: 各 depth で confidence 上位 N 件のみ展開
        visited: 循環防止の visited memory id set (外部呼出は None)

    Returns:
        [{"memory_entry": dict, "via_link": dict,
          "depth": int, "strength_hint": float}, ...]

    API 契約 (full 見据え):
        strength_hint は G-lite では via_link.confidence を流用、
        11-D Phase 3 (Physarum) で strength field に差し替え可能。
        呼び手は「retrieval 順序付けに使う float」として扱うだけ、
        source を知らない設計で後付け拡張に壊れない。
    """
    if not node_id or depth <= 0:
        return []
    visited = set(visited) if visited else set()
    visited.add(node_id)
    all_links = list_links(limit=LINK_SCAN_LIMIT)
    return _traverse_depth(
        node_id, 1, depth, all_links, visited,
        link_types, min_confidence, top_n_per_depth,
    )


def _traverse_depth(
    node_id: str,
    current_depth: int,
    max_depth: int,
    all_links: list,
    visited: set,
    link_types: Optional[tuple],
    min_confidence: float,
    top_n_per_depth: int,
) -> list:
    """follow_links の再帰実装 (current_depth 1 始まり).

    11-D Phase 3 Step 3.2: strength_hint を confidence → strength に置換。
    sort key も strength を優先 (Phase 2 以前 link は confidence にフォールバック)。
    API 契約は変更なし (呼び手は strength_hint を float として扱うだけ)。
    """
    if current_depth > max_depth:
        return []

    outgoing = [l for l in all_links if l.get("from_id") == node_id]
    filtered = []
    for l in outgoing:
        lt = l.get("link_type", "none")
        if lt == "none":
            continue
        if link_types is not None and lt not in link_types:
            continue
        if float(l.get("confidence", 0.0)) < min_confidence:
            continue
        filtered.append(l)
    # 11-D Phase 3 Step 3.2: sort key を strength に変更 (Phase 2 以前は confidence)
    filtered.sort(key=lambda l: _link_strength(l), reverse=True)
    near = filtered[:top_n_per_depth]

    results = []
    for link in near:
        to_id = link.get("to_id")
        if not to_id or to_id in visited:
            continue
        target_entry = _find_memory_entry_by_id(to_id)
        if target_entry is None:
            continue
        visited.add(to_id)
        results.append({
            "memory_entry": target_entry,
            "via_link": link,
            "depth": current_depth,
            # 11-D Phase 3 Step 3.2: strength_hint を strength に置換 (fallback あり)
            "strength_hint": _link_strength(link),
        })
        if current_depth < max_depth:
            child = _traverse_depth(
                to_id, current_depth + 1, max_depth,
                all_links, visited,
                link_types, min_confidence, top_n_per_depth,
            )
            results.extend(child)
    return results


def _find_memory_entry_by_id(entry_id: str) -> Optional[dict]:
    """登録済 network を走査して entry_id を持つ memory entry を見つける.

    G-lite では coarse (全 tag 走査、network n 増加で線形)。
    11-D Phase 7 migration で索引化を検討可。

    11-D Phase 2 hotfix (Phase 1 untagged 対応の補完): UNTAGGED_NETWORK も
    走査対象に追加。Phase 1 で memory_update / memory_forget /
    memory_network_search / list_records は untagged 対応済だったが、
    本関数の見落としで untagged memory が link traverse から見えなかった
    (link 生成は走るが follow_links から hit しない非対称) のを修復。
    """
    from core.memory import list_records, UNTAGGED_NETWORK
    from core.tag_registry import list_registered_tags
    for tag in list(list_registered_tags()) + [UNTAGGED_NETWORK]:
        try:
            recs = list_records(tag, limit=10000)
        except Exception:
            continue
        for r in recs:
            if r.get("id") == entry_id:
                return r
    return None
