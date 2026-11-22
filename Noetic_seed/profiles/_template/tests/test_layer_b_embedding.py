"""段階13 Phase 1 (Layer B embedding 保存 + 検索経路 hybrid 化) — 検証 test。

検証対象 (commit bf2145c):
  - core.memory._enrich_subjective_inline: entry mutate で subj に keywords /
    contextual_description / embedding を inline 注入、graceful skip
  - core.memory.non_empty_subjective_entries: subj 消費者の filter 順序契約
    (raw/subj 二層化で blank subj が日常的に存在する Phase 0.1.A 構造への対応)
  - core.memory.get_relevant_memories: subj entry append (kind='subjective' marker)、
    案 b concept-perception alignment、filter 順序で real subj 取得保証
  - core.memory.format_memories_for_prompt: subj 用 render branch
    (`[subjective] intent={x} | expect={y}` 形式)
  - core.entity_resolver.find_similar_subjective: 新 sibling、Tier 同 threshold
    (EMBEDDING_SAME 0.85 / DIFFERENT 0.70) 流用、limit 前 blank filter

CLAUDE.md §5 (識別力) + §6 (docstring 同期) literal 適用:
  - 想定誤実装 10 件カバー (enrich blank skip / graceful / populates × 3、
    filter 順序 × 2、kind marker 排他性 × 2、format branch × 2、tier 分類 × 1)
  - fixture 4 種混在 (blank / intent only / expect only / both)
  - assert は count + content (id 検証 / kind 検証 / 排他性検証)

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_layer_b_embedding.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.memory as cm
from core.memory import (
    UNTAGGED_NETWORK,
    _enrich_subjective_inline,
    non_empty_subjective_entries,
    get_relevant_memories,
    format_memories_for_prompt,
)
from core.entity_resolver import find_similar_subjective


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# _enrich_subjective_inline (entry mutate 経由 enrichment)
# ============================================================

def test_enrich_skips_blank_entry():
    """intent / expect 両方空 → enrichment skip (entry mutate なし、Phase 0.1.A
    raw event 起源 entry 等)。想定誤実装: blank に空 keywords 注入してしまう (汚染)。"""
    print("== _enrich_subjective_inline: blank skip ==")
    entry = {"id": "raw1", "tool": "search", "result": "ok"}
    _enrich_subjective_inline(entry)
    ok = True
    ok &= _assert("keywords" not in entry, "blank entry に keywords 注入なし")
    ok &= _assert("contextual_description" not in entry, "blank entry に contextual_description 注入なし")
    ok &= _assert("embedding" not in entry, "blank entry に embedding 注入なし")
    return ok


def test_enrich_graceful_on_llm_failure():
    """LLM 呼出失敗 → graceful skip (空 fields でも entry は壊れない、reflect 継続原則)。
    想定誤実装: LLM 失敗時に例外伝播 → entry 永続化 path が止まる。"""
    print("== _enrich_subjective_inline: graceful on LLM failure ==")
    original_gen = cm._generate_memory_metadata
    original_embed = cm._embed_sync
    original_ready = cm.is_vector_ready

    def raise_llm(*a, **kw):
        raise RuntimeError("LLM down")
    def raise_embed(*a, **kw):
        raise RuntimeError("embed down")

    cm._generate_memory_metadata = raise_llm
    cm.is_vector_ready = lambda: True
    cm._embed_sync = raise_embed
    try:
        entry = {"id": "subj1", "intent": "test intent", "expect": "test expect"}
        _enrich_subjective_inline(entry)
        ok = True
        ok &= _assert(entry.get("keywords") == [], f"LLM 失敗時 keywords 空 list: {entry.get('keywords')!r}")
        ok &= _assert(entry.get("contextual_description") == "",
                      f"LLM 失敗時 contextual_description 空 str: {entry.get('contextual_description')!r}")
        ok &= _assert("embedding" not in entry, "embedding 失敗時 entry に追加なし")
        return ok
    finally:
        cm._generate_memory_metadata = original_gen
        cm._embed_sync = original_embed
        cm.is_vector_ready = original_ready


def test_enrich_populates_via_mocks():
    """LLM mock + bge-m3 mock 成功 → entry に keywords/contextual_description/embedding 注入。
    想定誤実装: enrichment が entry に書き込まれない (空のまま jsonl に永続化される)。"""
    print("== _enrich_subjective_inline: populates via mocks ==")
    original_gen = cm._generate_memory_metadata
    original_embed = cm._embed_sync
    original_ready = cm.is_vector_ready

    cm._generate_memory_metadata = lambda text, network: {
        "keywords": ["k1", "k2", "k3"],
        "contextual_description": "test context",
    }
    cm.is_vector_ready = lambda: True
    cm._embed_sync = lambda texts: [[0.1] * 1024 for _ in texts]
    try:
        entry = {"id": "subj1", "intent": "test intent", "expect": "test expect"}
        _enrich_subjective_inline(entry)
        ok = True
        ok &= _assert(entry.get("keywords") == ["k1", "k2", "k3"],
                      f"keywords 注入: {entry.get('keywords')!r}")
        ok &= _assert(entry.get("contextual_description") == "test context",
                      f"contextual_description 注入: {entry.get('contextual_description')!r}")
        ok &= _assert(len(entry.get("embedding", [])) == 1024,
                      f"embedding 1024D 注入: len={len(entry.get('embedding', []))}")
        return ok
    finally:
        cm._generate_memory_metadata = original_gen
        cm._embed_sync = original_embed
        cm.is_vector_ready = original_ready


# ============================================================
# non_empty_subjective_entries (filter 順序契約 helper)
# ============================================================

def test_non_empty_filter_and_order():
    """non_empty_subjective_entries: 4 種混在 fixture (CLAUDE.md §5)、blank 除外 +
    元順序維持。想定誤実装: 順序破壊 (後半 entry 先頭にくる) / 過剰 filter
    (intent only / expect only も除外)。"""
    print("== non_empty_subjective_entries: 4 種混在 + 元順序 ==")
    # 4 種混在 fixture: 両方空 / intent のみ / expect のみ / 両方あり
    entries = [
        {"id": "blank1"},                                # 両方なし
        {"id": "intent_only", "intent": "x"},            # intent のみ (含む)
        {"id": "expect_only", "expect": "y"},            # expect のみ (含む)
        {"id": "both", "intent": "a", "expect": "b"},    # 両方あり (含む)
        {"id": "blank2"},                                # 両方なし (順序確認用 tail)
    ]
    result = non_empty_subjective_entries(entries)
    ok = True
    ok &= _assert(len(result) == 3, f"3 件残る (blank 2 件除外): actual={len(result)}")
    ids = [e["id"] for e in result]
    ok &= _assert(ids == ["intent_only", "expect_only", "both"], f"元順序維持: {ids}")
    return ok


# ============================================================
# get_relevant_memories (subj entry append + filter 順序)
# ============================================================

def test_get_relevant_filter_order_preserves_real_subj():
    """直近 N 件全部 raw event 起源 (blank) でも、それ以前の real subj が拾える
    (filter → slice 順序、Codex review 2 周目指摘 fix の検証)。
    想定誤実装: slice 先 → 中で skip = 末尾 3 件 全部 blank の case で real subj 0 件。"""
    print("== get_relevant_memories: filter 順序で real subj 取得 ==")
    state = {
        "subjective_entries": [
            {"id": "subj_real_1", "intent": "real intent A", "expect": "real expect A"},
            {"id": "subj_real_2", "intent": "real intent B", "expect": "real expect B"},
            {"id": "subj_real_3", "intent": "real intent C", "expect": "real expect C"},
            {"id": "raw_1"},  # ★ 末尾 3 件 全部 raw event 起源 (blank)
            {"id": "raw_2"},
            {"id": "raw_3"},
        ],
    }
    # is_vector_ready False で memory 検索 path skip、archive externals 不在
    original_ready = cm.is_vector_ready
    cm.is_vector_ready = lambda: False
    try:
        result = get_relevant_memories(state, limit=8)
        subj_results = [m for m in result if m.get("kind") == "subjective"]
        ok = True
        ok &= _assert(len(subj_results) == 3,
                      f"real subj 3 件取得 (blank 末尾 3 件 skip): actual={len(subj_results)}")
        subj_ids = {m["id"] for m in subj_results}
        ok &= _assert(subj_ids == {"subj_real_1", "subj_real_2", "subj_real_3"},
                      f"real subj IDs 一致: {subj_ids}")
        ok &= _assert(not any(m["id"].startswith("raw_") for m in subj_results),
                      "raw_* IDs は subj 結果に混入してない (排他性)")
        return ok
    finally:
        cm.is_vector_ready = original_ready


def test_get_relevant_does_not_mark_memory_as_subjective():
    """memory entry には kind='subjective' marker が付与されない (排他性、汚染なし)。
    想定誤実装: subj append 時に memory entry も marker 付与してしまう (concept 混線)。"""
    print("== get_relevant_memories: memory に subj marker 汚染なし ==")
    state = {
        "subjective_entries": [
            {"id": "subj1", "intent": "test", "expect": "ok"},
        ],
    }
    original_ready = cm.is_vector_ready
    cm.is_vector_ready = lambda: False
    try:
        result = get_relevant_memories(state, limit=8)
        subj_results = [m for m in result if m.get("kind") == "subjective"]
        non_subj = [m for m in result if m.get("kind") != "subjective"]
        ok = True
        ok &= _assert(len(subj_results) == 1, f"subj 1 件 (id=subj1): actual={len(subj_results)}")
        # subj 以外 (memory 候補があれば) には kind='subjective' なし
        ok &= _assert(not any(m.get("kind") == "subjective" for m in non_subj),
                      "memory 候補に subj marker 汚染なし (排他性)")
        return ok
    finally:
        cm.is_vector_ready = original_ready


# ============================================================
# format_memories_for_prompt (subj 用 render branch、案 b)
# ============================================================

def test_format_memories_subj_branch_render():
    """subj entry が `[subjective] intent={x} | expect={y}` 形式で render される
    (案 b concept-perception alignment、ゆう判断 2026-05-03)。
    想定誤実装: subj branch 不在 → memory 形式で表示 → ラベル `[?]` 等になる。"""
    print("== format_memories_for_prompt: subj 用 render branch ==")
    memories = [
        {"id": "subj1", "kind": "subjective", "intent": "test intent", "expect": "test expect"},
    ]
    output = format_memories_for_prompt(memories)
    ok = True
    ok &= _assert("[subjective]" in output, f"[subjective] ラベル: {output!r}")
    ok &= _assert("intent=test intent" in output, f"intent 表示: {output!r}")
    ok &= _assert("expect=test expect" in output, f"expect 表示: {output!r}")
    return ok


def test_format_memories_no_subj_branch_for_memory_entry():
    """memory entry には subj branch 適用されない (排他性、memory 形式維持)。
    想定誤実装: kind 判定が memory entry にも適用 → ラベル汚染。"""
    print("== format_memories_for_prompt: memory に subj branch 不適用 ==")
    memories = [
        {"id": "mem1", "network": UNTAGGED_NETWORK, "content": "memory content"},
    ]
    output = format_memories_for_prompt(memories)
    ok = True
    ok &= _assert("[untagged]" in output, f"[untagged] ラベル維持: {output!r}")
    ok &= _assert("[subjective]" not in output, "[subjective] ラベル混入なし (排他性)")
    return ok


# ============================================================
# find_similar_subjective (新 sibling、limit 前 filter + Tier 同 threshold)
# ============================================================

def test_find_similar_subjective_excludes_blank_before_limit():
    """find_similar_subjective: limit 前 blank filter で real subj が embedding 計算
    対象になる (Codex review 2 周目指摘 fix の検証)。
    想定誤実装: limit 先 → 先頭 limit 件が blank だと real subj が candidates 対象外。"""
    print("== find_similar_subjective: limit 前 blank filter ==")
    original_load = cm.load_all_subjective_entries
    blank_50 = [{"id": f"raw_{i}"} for i in range(50)]
    real_subj = [{"id": "real1", "intent": "real intent", "expect": "real expect"}]
    cm.load_all_subjective_entries = lambda: blank_50 + real_subj
    try:
        embed_fn = lambda texts: [[0.1] * 1024 for _ in texts]
        cosine_fn = lambda v1, v2: 0.9  # SAME_THRESHOLD 超え (Tier 2)
        new_entry = {"id": "query", "intent": "query intent", "expect": "query expect"}
        result = find_similar_subjective(new_entry, embed_fn=embed_fn, cosine_fn=cosine_fn, limit=10)
        ok = True
        result_ids = {c.get("id") for c, tier in result}
        ok &= _assert("real1" in result_ids,
                      f"real subj が結果に含まれる (limit 前 filter で blank skip): {result_ids}")
        # blank entry が結果に混入してないこと
        ok &= _assert(not any(rid.startswith("raw_") for rid in result_ids),
                      "blank candidates が結果に含まれない (filter 効いてる)")
        return ok
    finally:
        cm.load_all_subjective_entries = original_load


def test_find_similar_subjective_tier_threshold():
    """SAME_THRESHOLD 0.85 / DIFFERENT_THRESHOLD 0.70 流用、Tier 2/3 分類正確性。
    想定誤実装: threshold 別物 (新 magic number) / Tier 境界判定間違い。"""
    print("== find_similar_subjective: Tier threshold 流用 ==")
    original_load = cm.load_all_subjective_entries
    cands = [
        {"id": "high", "intent": "h", "expect": "h"},   # cosine = 0.9 → Tier 2
        {"id": "mid", "intent": "m", "expect": "m"},    # cosine = 0.75 → Tier 3
        {"id": "low", "intent": "l", "expect": "l"},    # cosine = 0.5 → 除外
    ]
    cm.load_all_subjective_entries = lambda: cands
    try:
        embed_fn = lambda texts: [[0.1] * 1024 for _ in texts]
        sim_iter = iter([0.9, 0.75, 0.5])
        cosine_fn = lambda v1, v2: next(sim_iter)
        new_entry = {"id": "q", "intent": "q", "expect": "q"}
        result = find_similar_subjective(new_entry, embed_fn=embed_fn, cosine_fn=cosine_fn, limit=10)
        ok = True
        by_tier = {tier: c["id"] for c, tier in result}
        ok &= _assert(by_tier.get(2) == "high", f"Tier 2 = high (>=0.85): {by_tier}")
        ok &= _assert(by_tier.get(3) == "mid", f"Tier 3 = mid (0.70-0.85): {by_tier}")
        ok &= _assert("low" not in [c["id"] for c, _ in result],
                      f"low (<0.70) は除外: result={[c['id'] for c, _ in result]}")
        return ok
    finally:
        cm.load_all_subjective_entries = original_load


# ============================================================
# main runner
# ============================================================

def main():
    results = [
        test_enrich_skips_blank_entry(),
        test_enrich_graceful_on_llm_failure(),
        test_enrich_populates_via_mocks(),
        test_non_empty_filter_and_order(),
        test_get_relevant_filter_order_preserves_real_subj(),
        test_get_relevant_does_not_mark_memory_as_subjective(),
        test_format_memories_subj_branch_render(),
        test_format_memories_no_subj_branch_for_memory_entry(),
        test_find_similar_subjective_excludes_blank_before_limit(),
        test_find_similar_subjective_tier_threshold(),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} test groups passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
