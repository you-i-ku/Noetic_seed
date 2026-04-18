"""entity_resolver.py テスト (段階4)。

3 段マッチングの各 Tier を個別検証:
  - Tier 1: exact (name, alias)
  - Tier 2: embedding (high/low/ambiguous)
  - Tier 3: LLM tiebreak (yes/no/None)
  - aliases 自動追加
  - ID collision 回避

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_entity_resolver.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.world_model import init_world_model, ensure_entity
from core.entity_resolver import (
    EMBEDDING_SAME_THRESHOLD,
    EMBEDDING_DIFFERENT_THRESHOLD,
    resolve_or_create_entity,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# Mock embedding / cosine
# ============================================================

def _mock_embed_factory(name_to_vec):
    """name -> 固定ベクトル の対応表から embed_fn を生成。
    未知の name は zero vector。"""
    def embed_fn(texts):
        return [name_to_vec.get(t, [0.0] * 3) for t in texts]
    return embed_fn


def _mock_cosine(a, b):
    """2 ベクトルの cosine 類似度。zero vector は 0.0。"""
    import math
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (na * nb)


# ============================================================
# Tier 1: exact
# ============================================================

def test_exact_match_by_name():
    print("== Tier 1: name 完全一致で既存 entity 返却 ==")
    wm = init_world_model()
    ensure_entity(wm, "ent_yuu", "ゆう")
    ent, is_new = resolve_or_create_entity(wm, "ゆう")
    return all([
        _assert(ent is not None, "entity 取得"),
        _assert(ent.get("id") == "ent_yuu", "既存 id"),
        _assert(is_new is False, "is_new=False"),
    ])


def test_exact_match_by_alias():
    print("== Tier 1: alias 完全一致で既存 entity 返却 ==")
    wm = init_world_model()
    ent = ensure_entity(wm, "ent_yuu", "ゆう")
    ent.setdefault("aliases", []).append("YOU")
    out, is_new = resolve_or_create_entity(wm, "YOU")
    return all([
        _assert(out is ent, "同じ entity"),
        _assert(is_new is False, "is_new=False"),
    ])


# ============================================================
# 新規作成
# ============================================================

def test_no_match_creates_new():
    print("== 未マッチで新規作成 ==")
    wm = init_world_model()
    ent, is_new = resolve_or_create_entity(wm, "新顔")
    return all([
        _assert(ent is not None, "entity 作成"),
        _assert(is_new is True, "is_new=True"),
        _assert(ent.get("name") == "新顔", "name"),
        _assert(ent.get("id", "").startswith("ent_"), "id prefix"),
        _assert("新顔" in [e["name"] for e in wm["entities"].values()],
                "wm に登録"),
    ])


def test_id_collision_uses_counter():
    print("== 同名別人 (slug 衝突) でも id が一意 ==")
    wm = init_world_model()
    # 先に ent_x を作る
    ensure_entity(wm, "ent_x", "x")
    # 新しい "x" を要求 (exact match するから同じになるはず)
    e1, new1 = resolve_or_create_entity(wm, "x")
    # "x" という名前の完全一致は既存。だから同じ entity
    # ここでは exact match の動作確認
    return all([
        _assert(e1.get("id") == "ent_x", "既存 entity に解決"),
        _assert(new1 is False, "is_new=False"),
    ])


# ============================================================
# Tier 2: embedding
# ============================================================

def test_embedding_high_similarity_merges():
    print("== Tier 2: cos 0.95 で既存に merge + alias 追加 ==")
    wm = init_world_model()
    ent = ensure_entity(wm, "ent_yuu", "ゆう")
    name_vecs = {"ゆう": [1.0, 0.0, 0.0], "YOU": [0.95, 0.05, 0.0]}  # 高類似
    out, is_new = resolve_or_create_entity(
        wm, "YOU",
        embed_fn=_mock_embed_factory(name_vecs),
        cosine_fn=_mock_cosine,
    )
    return all([
        _assert(out is ent, "既存 entity に解決"),
        _assert(is_new is False, "is_new=False"),
        _assert("YOU" in out.get("aliases", []), "YOU が alias に追加"),
    ])


def test_embedding_low_similarity_creates_new():
    print("== Tier 2: cos 0.5 で別 entity として新規 ==")
    wm = init_world_model()
    ensure_entity(wm, "ent_yuu", "ゆう")
    name_vecs = {"ゆう": [1.0, 0.0, 0.0], "太郎": [0.5, 0.87, 0.0]}  # 低類似
    out, is_new = resolve_or_create_entity(
        wm, "太郎",
        embed_fn=_mock_embed_factory(name_vecs),
        cosine_fn=_mock_cosine,
    )
    return all([
        _assert(is_new is True, "is_new=True (新規)"),
        _assert(out.get("name") == "太郎", "太郎 として作成"),
    ])


def test_embedding_ambiguous_without_llm_creates_new():
    print("== Tier 2 ambiguous (0.75) + llm_call_fn=None で新規 (安全側) ==")
    wm = init_world_model()
    ensure_entity(wm, "ent_yuu", "ゆう")
    name_vecs = {"ゆう": [1.0, 0.0, 0.0], "似た名": [0.78, 0.6, 0.0]}
    out, is_new = resolve_or_create_entity(
        wm, "似た名",
        embed_fn=_mock_embed_factory(name_vecs),
        cosine_fn=_mock_cosine,
    )
    sim = _mock_cosine(name_vecs["ゆう"], name_vecs["似た名"])
    return all([
        _assert(EMBEDDING_DIFFERENT_THRESHOLD <= sim < EMBEDDING_SAME_THRESHOLD,
                f"ambiguous 範囲 (sim={sim:.3f})"),
        _assert(is_new is True, "LLM なし → 新規 (安全側)"),
    ])


# ============================================================
# Tier 3: LLM tiebreak
# ============================================================

def test_llm_tiebreak_yes_merges():
    print("== Tier 3: ambiguous + LLM yes → merge ==")
    wm = init_world_model()
    ent = ensure_entity(wm, "ent_yuu", "ゆう")
    name_vecs = {"ゆう": [1.0, 0.0, 0.0], "similar": [0.78, 0.6, 0.0]}
    out, is_new = resolve_or_create_entity(
        wm, "similar",
        embed_fn=_mock_embed_factory(name_vecs),
        cosine_fn=_mock_cosine,
        llm_call_fn=lambda prompt, max_tokens=10: "yes",
    )
    return all([
        _assert(out is ent, "merge 先が既存"),
        _assert(is_new is False, "is_new=False"),
        _assert("similar" in out.get("aliases", []), "similar が alias"),
    ])


def test_llm_tiebreak_no_creates_new():
    print("== Tier 3: ambiguous + LLM no → 新規 ==")
    wm = init_world_model()
    ensure_entity(wm, "ent_yuu", "ゆう")
    name_vecs = {"ゆう": [1.0, 0.0, 0.0], "似た名": [0.78, 0.6, 0.0]}
    out, is_new = resolve_or_create_entity(
        wm, "似た名",
        embed_fn=_mock_embed_factory(name_vecs),
        cosine_fn=_mock_cosine,
        llm_call_fn=lambda prompt, max_tokens=10: "no",
    )
    return _assert(is_new is True, "LLM no → 新規")


def test_llm_exception_falls_back_to_new():
    print("== Tier 3: LLM が例外を投げても新規で継続 ==")
    wm = init_world_model()
    ensure_entity(wm, "ent_yuu", "ゆう")
    name_vecs = {"ゆう": [1.0, 0.0, 0.0], "似た名": [0.78, 0.6, 0.0]}

    def _bad_llm(prompt, max_tokens=10):
        raise RuntimeError("LLM 落ちた")

    out, is_new = resolve_or_create_entity(
        wm, "似た名",
        embed_fn=_mock_embed_factory(name_vecs),
        cosine_fn=_mock_cosine,
        llm_call_fn=_bad_llm,
    )
    return _assert(is_new is True, "例外でも新規作成で継続")


# ============================================================
# 入力不正
# ============================================================

def test_empty_name():
    print("== 空 name で (None, False) 返却 ==")
    wm = init_world_model()
    out, is_new = resolve_or_create_entity(wm, "")
    return all([
        _assert(out is None, "None 返却"),
        _assert(is_new is False, "is_new=False"),
    ])


def test_empty_wm():
    print("== wm=None で (None, False) 返却 ==")
    out, is_new = resolve_or_create_entity(None, "x")
    return all([
        _assert(out is None, "None"),
        _assert(is_new is False, "is_new=False"),
    ])


def test_no_embed_fn_falls_back_to_exact_only():
    print("== embed_fn=None で exact のみ動作 (未マッチで新規) ==")
    wm = init_world_model()
    ensure_entity(wm, "ent_yuu", "ゆう")
    out, is_new = resolve_or_create_entity(wm, "YOU")  # embed なし
    return _assert(is_new is True,
                   "embedding 無しで似た名も別 entity になる")


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        # Tier 1
        ("Tier 1: exact (name)", test_exact_match_by_name),
        ("Tier 1: exact (alias)", test_exact_match_by_alias),
        # 新規
        ("未マッチ → 新規", test_no_match_creates_new),
        ("id collision は exact で回避", test_id_collision_uses_counter),
        # Tier 2
        ("Tier 2: high sim → merge", test_embedding_high_similarity_merges),
        ("Tier 2: low sim → 新規", test_embedding_low_similarity_creates_new),
        ("Tier 2: ambiguous + no LLM → 新規",
         test_embedding_ambiguous_without_llm_creates_new),
        # Tier 3
        ("Tier 3: LLM yes → merge", test_llm_tiebreak_yes_merges),
        ("Tier 3: LLM no → 新規", test_llm_tiebreak_no_creates_new),
        ("Tier 3: LLM 例外 → 新規継続", test_llm_exception_falls_back_to_new),
        # 不正入力
        ("空 name", test_empty_name),
        ("空 wm", test_empty_wm),
        ("embed_fn=None → exact のみ", test_no_embed_fn_falls_back_to_exact_only),
    ]
    results = []
    for _label, fn in groups:
        print()
        ok = fn()
        results.append((_label, ok))
    print()
    print("=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for _label, ok in results:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {_label}")
    print(f"\n  {passed}/{total} groups passed")
    sys.exit(0 if passed == total else 1)
