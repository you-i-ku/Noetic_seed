"""memory fact 類似検索 (reconciliation の矛盾検出候補取得用)。

段階11-D Phase 0 Step 0.1b で段階4 Entity Resolver (resolve_or_create_entity +
Tier 1/2/3 helper) を削除。段階11-B Phase 3 Step 3.3 で追加された
find_similar_facts のみ残した。memory fact レイヤの 3 段類似検索:
- Tier 1 (exact): 同 network + metadata.entity_name 一致
- Tier 2 (embedding): 埋め込みベクトル cosine 類似度 >= 0.85 (SAME_THRESHOLD)
- Tier 3 (embedding ambiguous): 0.70 <= cosine < 0.85 (LLM judge 推奨領域)

caller: core/reconciliation.py `reconcile_memory_entry` の矛盾検出候補取得。
"""
from typing import Callable, Optional, Tuple

EMBEDDING_SAME_THRESHOLD = 0.85
EMBEDDING_DIFFERENT_THRESHOLD = 0.70


# ============================================================
# 段階11-B Phase 3 Step 3.3: memory fact 類似検索
# ============================================================

def find_similar_facts(new_entry: dict, *,
                       tiers: Tuple[int, ...] = (1, 2, 3),
                       embed_fn: Optional[Callable] = None,
                       cosine_fn: Optional[Callable] = None,
                       limit: int = 50) -> list:
    """memory fact 間の類似検索 (reconciliation の矛盾検出候補取得用)。

    - Tier 1: exact match (同 network + metadata.entity_name 一致)
    - Tier 2: embedding >= 0.85 (濃厚類似、矛盾候補)
    - Tier 3: 0.70 <= embedding < 0.85 (潜在類似、LLM judge 推奨)

    Args:
        new_entry: 比較元の memory entry dict (network / id / content / metadata 必須)
        tiers: 走査する tier の subset。例: (1,) で Tier 1 のみ
        embed_fn: (list[str]) -> list[list[float]]。None で Tier 2/3 skip
        cosine_fn: (vec, vec) -> float。None で Tier 2/3 skip
        limit: 同 network 内の走査対象 memory 件数上限 (cost 抑制)

    Returns:
        [(candidate_entry, tier), ...] — tier 昇順ソート
    """
    from core.memory import list_records

    network = new_entry.get("network", "")
    new_id = new_entry.get("id", "")
    if not network:
        return []

    all_records = list_records(network, limit=limit)
    candidates = [r for r in all_records if r.get("id") != new_id]
    if not candidates:
        return []

    results: list = []
    tier1_ids: set = set()

    # Tier 1: exact match (metadata.entity_name 基準)
    if 1 in tiers:
        new_ent_name = new_entry.get("metadata", {}).get("entity_name", "")
        if new_ent_name:
            for c in candidates:
                if c.get("metadata", {}).get("entity_name", "") == new_ent_name:
                    results.append((c, 1))
                    tier1_ids.add(c.get("id"))

    # Tier 2/3: embedding 類似
    if (2 in tiers or 3 in tiers) and embed_fn is not None and cosine_fn is not None:
        new_content = new_entry.get("content", "")
        if new_content:
            t23_candidates = [c for c in candidates if c.get("id") not in tier1_ids]
            if t23_candidates:
                try:
                    vecs = embed_fn([new_content] + [c.get("content", "") for c in t23_candidates])
                except Exception:
                    vecs = None
                if vecs and len(vecs) == 1 + len(t23_candidates):
                    query_vec = vecs[0]
                    for i, c in enumerate(t23_candidates):
                        try:
                            sim = float(cosine_fn(query_vec, vecs[i + 1]))
                        except Exception:
                            continue
                        if 2 in tiers and sim >= EMBEDDING_SAME_THRESHOLD:
                            results.append((c, 2))
                        elif 3 in tiers and EMBEDDING_DIFFERENT_THRESHOLD <= sim < EMBEDDING_SAME_THRESHOLD:
                            results.append((c, 3))

    return sorted(results, key=lambda x: x[1])


# ============================================================
# 段階13 Phase 1: subjective entry 類似検索 (Layer B sibling)
# ============================================================

def find_similar_subjective(new_entry: dict, *,
                            embed_fn: Optional[Callable] = None,
                            cosine_fn: Optional[Callable] = None,
                            limit: int = 50) -> list:
    """段階13 Phase 1: subjective_entries.jsonl 内の類似 entry を embedding で検索。

    PLAN §18-A4 (iv') literal「既存基盤フル流用」+ §21-6 (c) literal「signature 拡張
    じゃなく新 sibling 推奨」整合。`find_similar_facts` (memory entry 専用、network
    必須、Tier 1 metadata.entity_name) とは構造が違うため別関数化:
      - subjective entry は network / metadata.entity_name 概念なし
      - text 組み立ては intent + expect (Phase 0.4 / Phase 1 enrichment と同 text)
      - Tier は embedding only (SAME / DIFFERENT 同 threshold 流用)、Tier 1 metadata
        相当の素直な exact match 軸が無い

    Tier 構造 (find_similar_facts と同 threshold):
      - Tier 2: cosine >= EMBEDDING_SAME_THRESHOLD (0.85) — 濃厚類似
      - Tier 3: EMBEDDING_DIFFERENT_THRESHOLD <= cosine < SAME (0.70-0.85) — 潜在類似

    Args:
        new_entry: 比較元の subjective entry dict (intent / expect 必須)
        embed_fn: (list[str]) -> list[list[float]]。None で skip
        cosine_fn: (vec, vec) -> float。None で skip
        limit: 走査対象 subjective entry 件数上限 (cost 抑制)

    Returns:
        [(candidate_entry, tier), ...] — tier 昇順ソート
    """
    if embed_fn is None or cosine_fn is None:
        return []

    new_intent = new_entry.get("intent", "") or ""
    new_expect = new_entry.get("expect", "") or ""
    if not new_intent and not new_expect:
        return []
    new_text = f"intent: {new_intent}\nexpect: {new_expect}"
    new_id = new_entry.get("id", "")

    # 段階13 Phase 1: filter 順序 (filter → limit) を non_empty_subjective_entries
    # 経由で集約 (Codex review 2 周目指摘)。limit 前に blank filter しないと、
    # 先頭 limit 件が raw event 起源 (id-only) の場合に real subj が embedding 計算
    # 対象外になる + 意味なし `intent: \nexpect: ` text を embed する 2 種の bug。
    from core.memory import load_all_subjective_entries, non_empty_subjective_entries
    all_entries = non_empty_subjective_entries(load_all_subjective_entries())
    candidates = [e for e in all_entries if e.get("id") != new_id][:limit]
    if not candidates:
        return []

    cand_texts = []
    for c in candidates:
        ci = c.get("intent", "") or ""
        ce = c.get("expect", "") or ""
        cand_texts.append(f"intent: {ci}\nexpect: {ce}")

    try:
        vecs = embed_fn([new_text] + cand_texts)
    except Exception:
        return []
    if not vecs or len(vecs) != 1 + len(candidates):
        return []

    query_vec = vecs[0]
    results: list = []
    for i, c in enumerate(candidates):
        try:
            sim = float(cosine_fn(query_vec, vecs[i + 1]))
        except Exception:
            continue
        if sim >= EMBEDDING_SAME_THRESHOLD:
            results.append((c, 2))
        elif EMBEDDING_DIFFERENT_THRESHOLD <= sim < EMBEDDING_SAME_THRESHOLD:
            results.append((c, 3))

    return sorted(results, key=lambda x: x[1])
