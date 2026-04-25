"""test_memory_links.py — 段階11-B Phase 4 Step 4.7 (統合 β 方針)。

検証対象:
  [A] _parse_link_response: JSON robust parse + link_type validation + clamp
  [B] _llm_judge_link: LLM mock 経由の判定
  [C] link_type 候補 (similar/contradict/elaborate/causal/temporal/none)
  [D] confidence < 0.7 は discard
  [E] top_k=5 遵守 (近傍 top-K のみ judge)
  [F] _build_link_entry: 11-A default_self_perspective 付与
  [G] atomic append + list_links で読取
  [H] memory_links.jsonl が memory/*.jsonl とファイル分離
  [I] bidirectional traversal (from_id / to_id 両方向検索可能)
  [J] LLM error graceful fallback (link 生成 skip)
  [K] memory_store hook 経由で link 生成発火 (_state 渡し)
  [L] _link_generation_enabled=False で skip

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_memory_links.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_links_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

import core.memory_links as _ml
_ml.MEMORY_DIR = _tmp_memory  # link file も temp へ

from core.memory import list_records, memory_store
from core.memory_links import (
    LINK_CONFIDENCE_THRESHOLD,
    LINK_GENERATION_TOP_K,
    LINK_TYPES,
    _build_link_entry,
    _llm_judge_link,
    _parse_link_response,
    generate_links_for,
    list_links,
)
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
# Section A: _parse_link_response robustness
# =========================================================================
print("=== Section A: _parse_link_response ===")

p1 = _parse_link_response('{"link_type": "similar", "confidence": 0.85, "reason": "関連"}')
_assert(p1["link_type"] == "similar" and p1["confidence"] == 0.85, "A-1 similar 正常抽出")

p2 = _parse_link_response('{"link_type": "unknown_type", "confidence": 0.9, "reason": "x"}')
_assert(p2["link_type"] == "none", f"A-2 候補外 link_type は none に正規化 (got {p2['link_type']})")

p3 = _parse_link_response('not json')
_assert(
    p3["link_type"] == "none" and p3["confidence"] == 0.0,
    "A-3 非 JSON で graceful none 返却",
)

p4 = _parse_link_response('{"link_type": "causal", "confidence": 1.5, "reason": "clamp"}')
_assert(p4["confidence"] == 1.0, f"A-4 confidence clamp 1.0 (got {p4['confidence']})")

p5 = _parse_link_response('{"link_type": "temporal", "confidence": -0.2, "reason": "neg"}')
_assert(p5["confidence"] == 0.0, f"A-5 confidence clamp 0.0 (got {p5['confidence']})")


# =========================================================================
# Section B: _llm_judge_link with mock
# =========================================================================
print("=== Section B: _llm_judge_link ===")


def _mock_llm_elab(prompt, max_tokens=200, temperature=0.2):
    return '{"link_type": "elaborate", "confidence": 0.8, "reason": "entry A 詳述"}'


a_entry = {"content": "食べた", "network": "experience", "keywords": ["食事"]}
b_entry = {"content": "夕飯はオムライス", "network": "experience", "keywords": ["夕飯", "オムライス"]}
v_b = _llm_judge_link(a_entry, b_entry, llm_call_fn=_mock_llm_elab)
_assert(v_b["link_type"] == "elaborate", "B-1 LLM mock 応答で elaborate 判定")
_assert(v_b["confidence"] == 0.8, "B-2 confidence 抽出")


def _mock_llm_err(prompt, max_tokens=200, temperature=0.2):
    raise RuntimeError("mock fail")


v_e = _llm_judge_link(a_entry, b_entry, llm_call_fn=_mock_llm_err)
_assert(
    v_e["link_type"] == "none" and v_e["confidence"] == 0.0,
    "B-3 LLM error で graceful none fallback",
)


# =========================================================================
# Section C: LINK_TYPES 定数と閾値
# =========================================================================
print("=== Section C: link_type 候補と閾値 ===")

# 11-D Phase 2 で LINK_TYPES を 5 → 8 type に拡張 (co_activation / semantic /
# supporting 追加)。詳細は test_link_types_extended.py で個別検証、本 test では
# 既存 5 type が引き続き含まれることのみ回帰確認。
_assert(
    {"similar", "contradict", "elaborate", "causal", "temporal"}.issubset(set(LINK_TYPES)),
    f"C-1 LINK_TYPES に既存 5 type 含む (回帰確認、got {LINK_TYPES})",
)
_assert(LINK_CONFIDENCE_THRESHOLD == 0.7, f"C-2 confidence threshold 0.7 (PLAN 準拠)")
_assert(LINK_GENERATION_TOP_K == 5, f"C-3 top_k 5 (PLAN 準拠)")


# =========================================================================
# Section D: confidence < threshold は discard
# =========================================================================
print("=== Section D: confidence 閾値 discard ===")

_ = memory_store(
    "experience", "既存 entry D", {},
    origin="fixture", perspective=default_self_perspective(),
    _auto_metadata=False,
)
new_d = memory_store(
    "experience", "新 entry D", {},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)


def _mock_embed_uniform(texts):
    return [[1.0, 0.0] for _ in texts]


def _mock_cos(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _mock_llm_low_conf(prompt, max_tokens=200, temperature=0.2):
    return '{"link_type": "similar", "confidence": 0.5, "reason": "low"}'


gen_d = generate_links_for(
    new_d,
    embed_fn=_mock_embed_uniform, cosine_fn=_mock_cos,
    llm_call_fn=_mock_llm_low_conf,
)
_assert(len(gen_d) == 0, f"D-1 confidence < 0.7 で link 作らず (got {len(gen_d)})")


# =========================================================================
# Section E: top_k 遵守 (近傍上位のみ judge)
# =========================================================================
print("=== Section E: top_k 遵守 ===")

# experience に 10 件追加
for i in range(10):
    memory_store(
        "experience", f"bulk E {i}", {},
        origin="bulk", perspective=default_self_perspective(),
        _auto_metadata=False,
    )
new_e = memory_store(
    "experience", "新 entry E", {},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)

_judge_calls = []


def _mock_llm_count(prompt, max_tokens=200, temperature=0.2):
    _judge_calls.append(True)
    return '{"link_type": "similar", "confidence": 0.9, "reason": "near"}'


gen_e = generate_links_for(
    new_e,
    top_k=3,
    embed_fn=_mock_embed_uniform, cosine_fn=_mock_cos,
    llm_call_fn=_mock_llm_count,
)
_assert(
    len(_judge_calls) == 3,
    f"E-1 top_k=3 で judge は 3 回のみ (got {len(_judge_calls)})",
)
_assert(len(gen_e) == 3, f"E-2 link 3 件生成 (got {len(gen_e)})")


# =========================================================================
# Section F: _build_link_entry — 11-A perspective 付与
# =========================================================================
print("=== Section F: perspective 付与 ===")

le = _build_link_entry(
    {"id": "mem_from"},
    {"id": "mem_to"},
    {"link_type": "similar", "confidence": 0.8, "reason": "test"},
)
_assert(le["from_id"] == "mem_from", "F-1 from_id 保存")
_assert(le["to_id"] == "mem_to", "F-2 to_id 保存")
_assert(le["link_type"] == "similar", "F-3 link_type 保存")
_assert(le["confidence"] == 0.8, "F-4 confidence 保存")
_assert(le["id"].startswith("link_"), f"F-5 link_ prefix id (got {le['id']})")
_assert(
    isinstance(le.get("perspective"), dict)
    and le["perspective"].get("viewer") == "self",
    f"F-6 default_self_perspective 付与 (got {le.get('perspective')})",
)
_assert("created_at" in le, "F-7 created_at 付与")


# =========================================================================
# Section G: atomic append + list_links
# =========================================================================
print("=== Section G: append + list ===")

# 生成済 (Section E で 3 件) を list_links で読取確認
all_links = list_links(limit=100)
_assert(len(all_links) >= 3, f"G-1 生成済 link を list で読取 ({len(all_links)} 件)")
# 新しい順
if len(all_links) >= 2:
    t0 = all_links[0].get("created_at", "")
    t1 = all_links[1].get("created_at", "")
    _assert(t0 >= t1, "G-2 list_links 新しい順 (created_at 降順)")


# =========================================================================
# Section H: memory_links.jsonl がファイル分離
# =========================================================================
print("=== Section H: ファイル分離 ===")

link_fpath = _tmp_memory / "memory_links.jsonl"
ent_fpath = _tmp_memory / "experience.jsonl"
_assert(link_fpath.exists(), "H-1 memory_links.jsonl 存在")
_assert(ent_fpath.exists(), "H-2 experience.jsonl 存在 (fixture)")
# 相互に別ファイル
_assert(link_fpath != ent_fpath, "H-3 別ファイル")
# experience.jsonl に link entry が混入しない
ent_lines = ent_fpath.read_text(encoding="utf-8").splitlines()
for line in ent_lines:
    if not line.strip():
        continue
    try:
        e = json.loads(line)
        _assert(
            e.get("id", "").startswith("mem_"),
            f"H-4 experience.jsonl に mem_ id のみ (got {e.get('id', '')[:10]})",
        )
    except Exception:
        pass


# =========================================================================
# Section I: bidirectional traversal
# =========================================================================
print("=== Section I: bidirectional traversal ===")

# Section E で new_e → 近傍 3 件への link が生成済
# from_id == new_e.id で抽出
new_e_id = new_e.get("id", "")
from_e = [l for l in all_links if l.get("from_id") == new_e_id]
_assert(len(from_e) >= 3, f"I-1 from_id 検索で new_e の link を取得 ({len(from_e)})")

# 任意の link の to_id で reverse lookup
if from_e:
    to_id = from_e[0].get("to_id")
    reverse = [l for l in all_links if l.get("to_id") == to_id]
    _assert(len(reverse) >= 1, "I-2 to_id で reverse lookup 可能")


# =========================================================================
# Section J: LLM error graceful
# =========================================================================
print("=== Section J: LLM error graceful ===")

new_j = memory_store(
    "opinion", "J test", {"confidence": 0.5},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)
_existing_j = memory_store(
    "opinion", "J fixture", {"confidence": 0.5},
    origin="fixture", perspective=default_self_perspective(),
    _auto_metadata=False,
)

gen_j = generate_links_for(
    new_j,
    embed_fn=_mock_embed_uniform, cosine_fn=_mock_cos,
    llm_call_fn=_mock_llm_err,  # 全 judge が raise
)
_assert(len(gen_j) == 0, "J-1 LLM error で link 生成ゼロ")


# =========================================================================
# Section K: memory_store hook 経由で link 生成発火 (_state 渡し)
# =========================================================================
print("=== Section K: memory_store hook 経由 ===")

_orig_gen = _ml.generate_links_for
_hook_called = []


def _mock_gen(new_entry, *, top_k=5, embed_fn=None, cosine_fn=None,
              llm_call_fn=None, confidence_threshold=0.7, candidate_limit=50):
    _hook_called.append(new_entry.get("id", ""))
    return []


_ml.generate_links_for = _mock_gen

try:
    state_k = {}
    e_k = memory_store(
        "experience", "K hook test", {},
        origin="test", perspective=default_self_perspective(),
        _auto_metadata=False,
        _state=state_k,  # hook 発火条件
    )
    _assert(len(_hook_called) == 1, f"K-1 _state 渡しで generate_links_for 1 回呼出 ({len(_hook_called)})")
    if _hook_called:
        _assert(_hook_called[0] == e_k.get("id"), "K-2 hook に new_entry が渡される")
finally:
    _ml.generate_links_for = _orig_gen


# =========================================================================
# Section L: _link_generation_enabled=False で skip
# =========================================================================
print("=== Section L: _link_generation_enabled=False ===")

_hook_called2 = []


def _mock_gen2(new_entry, **kwargs):
    _hook_called2.append(True)
    return []


_ml.generate_links_for = _mock_gen2

try:
    state_l = {}
    _e_l = memory_store(
        "experience", "L skip test", {},
        origin="test", perspective=default_self_perspective(),
        _auto_metadata=False,
        _state=state_l,
        _link_generation_enabled=False,  # hook skip
    )
    _assert(len(_hook_called2) == 0, f"L-1 _link_generation_enabled=False で skip (called {len(_hook_called2)})")
finally:
    _ml.generate_links_for = _orig_gen


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
