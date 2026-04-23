"""test_reflection_phase1.py — 段階11-B Phase 1 Step 1.6。

検証対象 (挙動変化ゼロ保証):
  - reflection.py:297 の rule 駆動化 (get_tags_with_rule("c_gradual_source")) が
    旧ハードコード networks=["entity"] と同一 entry を返す (entity lookup)
  - reflection.py:393 の rule 駆動化 (get_tags_with_rule で records 収集) が
    旧ハードコード list_records("entity", limit=20) と同一件数 / 同一 id 列を返す
  - registry 空 (Phase 5 白紙想定) で rule 駆動化 pathway が空 result を返す
    = reflect の entity 抽出 / C-gradual WM sync が自然に skip される前提

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_reflection_phase1.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_phase1_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

from core.memory import list_records, memory_network_search, memory_store
from core.perspective import default_self_perspective
from core.tag_registry import (
    get_tags_with_rule,
    register_standard_tags,
    register_tag,
)


register_standard_tags()


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


# =========================================================================
# Section A: Step 1.4 — entity lookup の rule 駆動化が旧挙動と等価
# =========================================================================
print("=== Section A: entity lookup equivalence (L297) ===")

_p = default_self_perspective()
memory_store("entity", "好きな食べ物はチョコ", {"entity_name": "ゆう"},
             origin="test", perspective=_p)
memory_store("entity", "Noetic の開発者", {"entity_name": "ゆう"},
             origin="test", perspective=_p)
memory_store("opinion", "テストは書くべき", {"confidence": 0.8},
             origin="test", perspective=_p)
memory_store("experience", "朝から実装を続けた",
             origin="test", perspective=_p)

_hardcode_existing = memory_network_search("ゆう", networks=["entity"], limit=3)
_rule_tags = get_tags_with_rule("c_gradual_source")
_rule_existing = memory_network_search("ゆう", networks=_rule_tags, limit=3)

_assert(_rule_tags == ["entity"], f"A-1 rule_tags == ['entity'] (got {_rule_tags})")
_assert(
    len(_hardcode_existing) == len(_rule_existing),
    f"A-2 件数一致 (hardcode={len(_hardcode_existing)} rule={len(_rule_existing)})",
)
_hard_ids = sorted(e.get("id", "") for e in _hardcode_existing)
_rule_ids = sorted(e.get("id", "") for e in _rule_existing)
_assert(_hard_ids == _rule_ids, f"A-3 id 列一致 ({_hard_ids})")

# opinion / experience が entity lookup に漏れ込まない
_assert(
    all(e.get("network") == "entity" for e in _rule_existing),
    "A-4 rule 駆動 lookup が entity tag のみ返す",
)


# =========================================================================
# Section B: Step 1.5 — C-gradual sync source の rule 駆動化が旧挙動と等価
# =========================================================================
print("=== Section B: C-gradual records equivalence (L393) ===")

_hardcode_records = list_records("entity", limit=20)
_rule_records = []
for tag in get_tags_with_rule("c_gradual_source"):
    _rule_records.extend(list_records(tag, limit=20))

_assert(
    len(_hardcode_records) == len(_rule_records),
    f"B-1 件数一致 (hardcode={len(_hardcode_records)} rule={len(_rule_records)})",
)
_hard_ids_b = sorted(r.get("id", "") for r in _hardcode_records)
_rule_ids_b = sorted(r.get("id", "") for r in _rule_records)
_assert(_hard_ids_b == _rule_ids_b, f"B-2 id 列一致 (n={len(_hard_ids_b)})")

# 非 c_gradual_source tag (opinion / experience) は漏れない
_assert(
    all(r.get("network") == "entity" for r in _rule_records),
    "B-3 rule 駆動 records が entity のみ (opinion / experience 混入なし)",
)


# =========================================================================
# Section C: 白紙 registry で rule 駆動が空 result (Phase 5 前提整合)
# =========================================================================
print("=== Section C: empty registry gives empty rule_tags ===")

# 別 temp registry で _reset
_tmp_empty = _tmp_memory / "empty_registered_tags.json"
_tr._reset_for_testing(registry_file=_tmp_empty)

_empty_tags = get_tags_with_rule("c_gradual_source")
_assert(_empty_tags == [], f"C-1 空 registry → rule_tags == [] (got {_empty_tags})")

# reflection L297 の `if entity_tags:` 分岐で空 → existing=[] に落ちる前提を示す
_fallback_existing = [] if not _empty_tags else memory_network_search(
    "ゆう", networks=_empty_tags, limit=3)
_assert(_fallback_existing == [], "C-2 空 rule_tags → existing=[] (L297 else 分岐)")

# reflection L393 の for loop 空 iteration で records=[] に落ちる前提を示す
_fallback_records = []
for tag in _empty_tags:
    _fallback_records.extend(list_records(tag, limit=20))
_assert(_fallback_records == [], "C-3 空 rule_tags → records=[] (L393 for-loop 空)")


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
