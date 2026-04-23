"""Entity Resolver — 名前から WM entity を解決する 3 段マッチング。

WORLD_MODEL.md §6 段階4 実装。

段階構成:
- Tier 1 (exact): name が既存 entity の name または aliases と完全一致
- Tier 2 (embedding): 埋め込みベクトル cosine 類似度による判定
    - >= EMBEDDING_SAME_THRESHOLD (0.85): 同一とみなして既存に merge、name を aliases 追加
    - <  EMBEDDING_DIFFERENT_THRESHOLD (0.70): 別 entity として新規作成
    - 中間 (0.70 - 0.85): Tier 3 へ
- Tier 3 (LLM tiebreaker): llm_call_fn 渡されていれば LLM に "同一か?" を問う
    - True: merge
    - False または llm_call_fn=None: 新規作成 (安全側: 統合しすぎより分けすぎの方がマシ)

依存注入:
- embed_fn / cosine_fn / llm_call_fn は caller 側から渡す (テスト容易性 + 段階5+ での差替可)
- いずれも None なら該当 Tier を skip し、Tier 1 のみで判定 → 未マッチなら新規作成

段階5+ で予定される拡張:
- aliases の semantic 重複マージ (入れすぎ防止)
- 決定キャッシュ (同じ pair を毎回判定しない)
- LLM コール回数のレート制限
"""
from typing import Callable, Optional, Tuple

from core.world_model import _now, _slugify, ensure_entity

EMBEDDING_SAME_THRESHOLD = 0.85
EMBEDDING_DIFFERENT_THRESHOLD = 0.70


# ============================================================
# 内部ヘルパ
# ============================================================

def _all_names(entity: dict) -> list:
    """entity の primary name + aliases を 1 つのリストで返す (空文字除外)。"""
    names = [entity.get("name", "")]
    names.extend(entity.get("aliases", []))
    return [n for n in names if n]


def _add_alias(entity: dict, new_name: str) -> None:
    """alias を追加 (重複 skip、primary name と同じなら skip)。in-place。"""
    if not new_name:
        return
    if new_name == entity.get("name"):
        return
    aliases = entity.setdefault("aliases", [])
    if new_name in aliases:
        return
    aliases.append(new_name)
    entity["updated_at"] = _now()


# ============================================================
# Tier 1: exact match
# ============================================================

def _resolve_exact(wm: dict, name: str) -> Optional[dict]:
    """name が entity.name または aliases と完全一致するなら返す。"""
    for entity in wm.get("entities", {}).values():
        if name in _all_names(entity):
            return entity
    return None


# ============================================================
# Tier 2: embedding
# ============================================================

def _resolve_embedding(wm: dict, name: str,
                       embed_fn: Callable, cosine_fn: Callable
                       ) -> Tuple[Optional[dict], float]:
    """embedding cosine 類似度で best match を返す。

    戻り値: (best_entity, best_similarity)。計算不能や候補なしは (None, 0.0)。
    """
    entities = list(wm.get("entities", {}).values())
    if not entities:
        return None, 0.0

    # 各 entity の全 name (primary + aliases) を平坦化
    flat_names: list = []
    flat_refs: list = []
    for e in entities:
        for n in _all_names(e):
            flat_names.append(n)
            flat_refs.append(e)
    if not flat_names:
        return None, 0.0

    try:
        vecs = embed_fn([name] + flat_names)
    except Exception:
        return None, 0.0
    if not vecs or len(vecs) != 1 + len(flat_names):
        return None, 0.0

    query_vec = vecs[0]
    best_sim = 0.0
    best_entity: Optional[dict] = None
    for i, _ in enumerate(flat_names):
        try:
            sim = cosine_fn(query_vec, vecs[i + 1])
        except Exception:
            continue
        if sim > best_sim:
            best_sim = float(sim)
            best_entity = flat_refs[i]
    return best_entity, best_sim


# ============================================================
# Tier 3: LLM tiebreaker
# ============================================================

def _resolve_llm(candidate_entity: dict, name: str,
                 llm_call_fn: Callable) -> bool:
    """LLM に「同一か?」を問う。True なら merge 推奨、False なら別 entity。
    例外時は False (安全側)。
    """
    aliases = candidate_entity.get("aliases", [])
    prompt = (
        "以下の 2 つの名前は、同じ存在を指しますか？ yes または no で 1 単語で答えてください。\n"
        f"- 候補A: {candidate_entity.get('name', '')}\n"
    )
    if aliases:
        prompt += f"  (別名: {', '.join(aliases)})\n"
    prompt += f"- 候補B: {name}\n\n回答 (yes/no のみ):"
    try:
        resp = llm_call_fn(prompt, max_tokens=10)
        return str(resp).strip().lower().startswith("y")
    except Exception:
        return False


# ============================================================
# 公開 API
# ============================================================

def resolve_or_create_entity(wm: dict, name: str,
                             embed_fn: Optional[Callable] = None,
                             cosine_fn: Optional[Callable] = None,
                             llm_call_fn: Optional[Callable] = None
                             ) -> Tuple[Optional[dict], bool]:
    """名前から WM entity を解決。存在すれば既存を返し、なければ新規作成する。

    - Tier 1 (exact) で見つかれば即返す (is_new=False)
    - Tier 2 (embedding) で >= SAME_THRESHOLD なら既存に merge + alias 追加
    - ambiguous なら Tier 3 (LLM) で判定
    - いずれも match しなければ新規作成 (is_new=True)

    Args:
        wm: world_model dict
        name: 解決対象の名前
        embed_fn: (list[str]) -> list[list[float]] を返す関数。None で Tier 2 skip
        cosine_fn: (vec, vec) -> float。None で Tier 2 skip
        llm_call_fn: (prompt, max_tokens=N) -> str。None で Tier 3 skip

    Returns:
        (entity, is_new)。wm/name 不正時は (None, False)。
    """
    if not wm or not name:
        return None, False
    name = name.strip()
    if not name:
        return None, False

    # Tier 1: exact
    match = _resolve_exact(wm, name)
    if match is not None:
        return match, False

    # Tier 2: embedding
    best_entity: Optional[dict] = None
    best_sim = 0.0
    if embed_fn is not None and cosine_fn is not None:
        best_entity, best_sim = _resolve_embedding(wm, name, embed_fn, cosine_fn)

    if best_entity is not None and best_sim >= EMBEDDING_SAME_THRESHOLD:
        _add_alias(best_entity, name)
        return best_entity, False

    # Tier 3: LLM (ambiguous 領域のみ)
    if (best_entity is not None
            and best_sim >= EMBEDDING_DIFFERENT_THRESHOLD
            and llm_call_fn is not None):
        if _resolve_llm(best_entity, name, llm_call_fn):
            _add_alias(best_entity, name)
            return best_entity, False

    # No match → create with unique entity_id
    base_id = f"ent_{_slugify(name)}"
    entity_id = base_id
    counter = 0
    existing_entities = wm.get("entities", {})
    while entity_id in existing_entities:
        counter += 1
        entity_id = f"{base_id}_{counter}"
    new_entity = ensure_entity(wm, entity_id, name)
    return new_entity, True


# ============================================================
# 段階11-B Phase 3 Step 3.3: memory fact 類似検索
# ============================================================

def find_similar_facts(new_entry: dict, *,
                       tiers: Tuple[int, ...] = (1, 2, 3),
                       embed_fn: Optional[Callable] = None,
                       cosine_fn: Optional[Callable] = None,
                       limit: int = 50) -> list:
    """memory fact 間の類似検索 (reconciliation の矛盾検出候補取得用)。

    段階4 Entity Resolver の 3 段概念を memory fact レイヤに流用、閾値は
    EMBEDDING_SAME_THRESHOLD (0.85) / EMBEDDING_DIFFERENT_THRESHOLD (0.70) 共通。

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
