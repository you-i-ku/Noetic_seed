"""test_entity_find_similar_facts.py — 段階11-B Phase 3 Step 3.3 副次実装。

検証対象 (find_similar_facts):
  - Tier 1: 同 network + metadata.entity_name 一致で hit
  - Tier 2: embedding >= 0.85 (EMBEDDING_SAME_THRESHOLD) で hit (mock embed)
  - Tier 3: 0.70 <= embedding < 0.85 (EMBEDDING_DIFFERENT_THRESHOLD 間) で hit
  - 0.70 未満は hit しない
  - embed_fn/cosine_fn None で Tier 2/3 skip (Tier 1 のみ有効)
  - new_entry 自身 (同 id) は候補から除外
  - tiers subset (例: (1,)) で Tier フィルタ
  - limit で走査上限

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_entity_find_similar_facts.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_find_similar_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

from core.entity_resolver import (
    EMBEDDING_DIFFERENT_THRESHOLD,
    EMBEDDING_SAME_THRESHOLD,
    find_similar_facts,
)
from core.memory import memory_store
from core.perspective import default_self_perspective
from core.tag_registry import register_standard_tags


register_standard_tags()


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


# =========================================================================
# fixture: 既存 entity fact を複数作成 (_auto_metadata=False で LLM skip)
# =========================================================================
print("=== fixture: 既存 entity fact 作成 ===")

e_yuu_1 = memory_store(
    "entity", "好きな食べ物はチョコレート", {"entity_name": "ゆう"},
    origin="fixture", perspective=default_self_perspective(),
    _auto_metadata=False,
)
e_yuu_2 = memory_store(
    "entity", "Noetic の開発者", {"entity_name": "ゆう"},
    origin="fixture", perspective=default_self_perspective(),
    _auto_metadata=False,
)
e_ik_1 = memory_store(
    "entity", "AI として存在する", {"entity_name": "iku"},
    origin="fixture", perspective=default_self_perspective(),
    _auto_metadata=False,
)


# =========================================================================
# Section A: Tier 1 exact match (entity_name)
# =========================================================================
print("=== Section A: Tier 1 exact match ===")

new_yuu = memory_store(
    "entity", "ゆうについての新しい観察", {"entity_name": "ゆう"},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)
r_a = find_similar_facts(new_yuu, tiers=(1,))
tier1_ids = {c.get("id") for c, t in r_a if t == 1}
_assert(
    {e_yuu_1["id"], e_yuu_2["id"]} <= tier1_ids,
    f"A-1 entity_name='ゆう' で既存 2 件が Tier 1 hit (got {len(tier1_ids)} 件)",
)
_assert(
    e_ik_1["id"] not in tier1_ids,
    "A-2 異 entity_name (iku) は含まれない",
)
_assert(
    new_yuu["id"] not in tier1_ids,
    "A-3 new_entry 自身は除外",
)


# =========================================================================
# Section B: Tier 1 mismatch
# =========================================================================
print("=== Section B: entity_name 未指定 → Tier 1 hit なし ===")

new_nameless = memory_store(
    "entity", "名もなき観察", {},  # entity_name 未指定
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)
r_b = find_similar_facts(new_nameless, tiers=(1,))
_assert(r_b == [], "B-1 entity_name 未指定で Tier 1 hit ゼロ")


# =========================================================================
# Section C: Tier 2 embedding >= 0.85 (mock)
# =========================================================================
print("=== Section C: Tier 2 (embedding >= 0.85) ===")


def _mock_embed_uniform(texts):
    """全テキストに同一ベクトル → cosine 類似度 1.0 → Tier 2 判定"""
    return [[1.0, 0.0, 0.0] for _ in texts]


def _mock_cosine(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


new_for_t2 = memory_store(
    "experience", "何か起きた A", {},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)
_ = memory_store(
    "experience", "何か起きた B", {},
    origin="fixture", perspective=default_self_perspective(),
    _auto_metadata=False,
)

r_c = find_similar_facts(
    new_for_t2,
    tiers=(1, 2, 3),
    embed_fn=_mock_embed_uniform,
    cosine_fn=_mock_cosine,
)
tier2_hits = [c for c, t in r_c if t == 2]
_assert(len(tier2_hits) >= 1, f"C-1 cosine=1.0 で Tier 2 hit ({len(tier2_hits)} 件)")


# =========================================================================
# Section D: Tier 3 (0.70 <= embedding < 0.85)
# =========================================================================
print("=== Section D: Tier 3 (0.70-0.85) ===")


def _mock_embed_tier3(texts):
    """1 番目と 2 番目で cosine ≒ 0.75 になるベクトル返却 (Tier 3 判定)"""
    # query: [1, 0]、候補: [cos(θ), sin(θ)] θ=arccos(0.75) ≒ 41.4°
    import math
    theta = math.acos(0.75)
    return [
        [1.0, 0.0] if i == 0 else [math.cos(theta), math.sin(theta)]
        for i, _ in enumerate(texts)
    ]


# 新プロファイル相当: fresh experience で 1 件目
_new_for_t3 = memory_store(
    "opinion", "懐疑的な主張", {"confidence": 0.5},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)
_ = memory_store(
    "opinion", "肯定的な主張", {"confidence": 0.6},
    origin="fixture", perspective=default_self_perspective(),
    _auto_metadata=False,
)

r_d = find_similar_facts(
    _new_for_t3,
    tiers=(2, 3),
    embed_fn=_mock_embed_tier3,
    cosine_fn=_mock_cosine,
)
tier3_hits = [c for c, t in r_d if t == 3]
tier2_hits_d = [c for c, t in r_d if t == 2]
_assert(len(tier3_hits) >= 1, f"D-1 cosine≒0.75 で Tier 3 hit ({len(tier3_hits)} 件)")
_assert(len(tier2_hits_d) == 0, "D-2 Tier 2 には入らない (0.85 未満)")


# =========================================================================
# Section E: embed_fn None → Tier 2/3 skip (Tier 1 のみ)
# =========================================================================
print("=== Section E: embed_fn None で Tier 2/3 skip ===")

r_e = find_similar_facts(new_yuu, tiers=(1, 2, 3))  # embed_fn 未指定
_e_tiers = {t for _, t in r_e}
_assert(
    _e_tiers <= {1},
    f"E-1 embed_fn None で Tier 2/3 は含まれない (got tiers {_e_tiers})",
)


# =========================================================================
# Section F: tiers subset
# =========================================================================
print("=== Section F: tiers=(1,) で Tier 1 のみ ===")

r_f = find_similar_facts(
    new_for_t2, tiers=(1,),
    embed_fn=_mock_embed_uniform, cosine_fn=_mock_cosine,
)
_f_tiers = {t for _, t in r_f}
_assert(_f_tiers <= {1}, f"F-1 tiers=(1,) で Tier 1 のみ (got {_f_tiers})")


# =========================================================================
# Section G: limit 効く (走査件数上限)
# =========================================================================
print("=== Section G: limit ===")

# experience に大量追加
for i in range(15):
    memory_store(
        "experience", f"bulk entry {i}", {},
        origin="bulk", perspective=default_self_perspective(),
        _auto_metadata=False,
    )
# limit=5 で list_records が 5 件しか返さない → candidate 最大 5 件
# (new_for_t2 自身は list の古側にいて 5 件に含まれない場合がある、厳密な上限のみ確認)
r_g = find_similar_facts(
    new_for_t2, tiers=(1, 2, 3),
    embed_fn=_mock_embed_uniform, cosine_fn=_mock_cosine,
    limit=5,
)
_assert(len(r_g) <= 5, f"G-1 limit=5 で candidate <= 5 (got {len(r_g)})")


# =========================================================================
# Section H: 空 network / new_entry 不正
# =========================================================================
print("=== Section H: robustness ===")

_assert(
    find_similar_facts({}, tiers=(1,)) == [],
    "H-1 空 entry (network 無し) で [] 返却",
)
_assert(
    find_similar_facts({"network": "nonexistent_network"}, tiers=(1,)) == [],
    "H-2 未登録 network で [] 返却 (list_records が [] 返すため)",
)


# =========================================================================
print("=" * 60)
_pass = sum(1 for r, _ in results if r)
_total = len(results)
print(f"結果: {_pass}/{_total} passed")
for ok, msg in results:
    if not ok:
        print(f"  FAIL: {msg}")
print("=" * 60)

try:
    shutil.rmtree(_tmp_root, ignore_errors=True)
except Exception:
    pass

sys.exit(0 if _pass == _total else 1)
