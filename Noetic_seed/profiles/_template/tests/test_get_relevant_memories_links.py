"""test_get_relevant_memories_links.py — 段階11-C G-lite Phase 1 hook 統合.

検証対象: core.memory.get_relevant_memories の use_links 引数

  [A] use_links=False (デフォルト) で既存挙動 = link merge なし (後方互換)
  [B] use_links=True で link 由来 memory が merge される
  [C] _retrieval_via="link" / _retrieval_depth / _retrieval_strength_hint 付与
  [D] link 無し環境で use_links=True でも error 出ない (graceful)

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_get_relevant_memories_links.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_getrel_"))
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
_ml.MEMORY_DIR = _tmp_memory

from core.memory import memory_store, get_relevant_memories
from core.memory_links import _append_link, _build_link_entry
from core.tag_registry import register_standard_tags


register_standard_tags()


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _make_state_with_intent(intent: str) -> dict:
    """log に直近 intent を 1 件入れた最小 state."""
    return {
        "log": [{"intent": intent, "time": "2026-04-24 00:00:00"}],
    }


def _link(src: dict, dst: dict, confidence: float = 0.9) -> None:
    """手動 link 作成 (LLM 呼出 skip)."""
    verdict = {"link_type": "similar", "confidence": confidence, "reason": "test"}
    _append_link(_build_link_entry(src, dst, verdict))


# ============================================================
# [A] use_links=False 後方互換
# ============================================================

def test_default_use_links_false_preserves_behavior():
    print("== use_links=False デフォルトで link merge なし ==")
    a = memory_store("opinion", "curiosity_expression_marker_a", {})
    b = memory_store("opinion", "silent_neighbor_b", {})
    _link(a, b, confidence=0.9)

    state = _make_state_with_intent("curiosity_expression_marker_a")
    default_result = get_relevant_memories(state, limit=8)
    vias = [m.get("_retrieval_via") for m in default_result]
    return all([
        _assert("link" not in vias,
                "link 由来 entry なし (_retrieval_via=link 不在)"),
    ])


# ============================================================
# [B] use_links=True で link merge
# ============================================================

def test_use_links_true_merges_link_neighbors():
    print("== use_links=True で link 由来 merge ==")
    a = memory_store("opinion", "ocean_view_marker_b", {})
    b = memory_store("opinion", "sky_reflection_b", {})
    _link(a, b, confidence=0.9)

    state = _make_state_with_intent("ocean_view_marker_b")
    result = get_relevant_memories(state, limit=8, use_links=True)
    link_entries = [m for m in result if m.get("_retrieval_via") == "link"]
    link_ids = [m.get("id") for m in link_entries]
    return all([
        _assert(len(link_entries) >= 1, "link 由来 entry >= 1"),
        _assert(b["id"] in link_ids, "B が link 経由で merge"),
    ])


# ============================================================
# [C] retrieval marker 属性
# ============================================================

def test_retrieval_marker_attributes():
    print("== _retrieval_via / _retrieval_depth / _retrieval_strength_hint 付与 ==")
    a = memory_store("opinion", "forest_path_marker_c", {})
    b = memory_store("opinion", "trees_neighbor_c", {})
    _link(a, b, confidence=0.85)

    state = _make_state_with_intent("forest_path_marker_c")
    result = get_relevant_memories(state, limit=8, use_links=True)
    link_entries = [m for m in result if m.get("_retrieval_via") == "link"]
    if not link_entries:
        return _assert(False, "link entry 見つからない (前提崩壊)")
    e = link_entries[0]
    return all([
        _assert(e.get("_retrieval_via") == "link", "_retrieval_via='link'"),
        _assert(e.get("_retrieval_depth", 0) >= 1, "_retrieval_depth >= 1"),
        _assert(isinstance(e.get("_retrieval_strength_hint"), float),
                "_retrieval_strength_hint is float"),
    ])


# ============================================================
# [D] link 無し環境での use_links=True
# ============================================================

def test_use_links_no_links_no_error():
    print("== link 無し + use_links=True: error 出ず空 merge ==")
    memory_store("opinion", "lonely_marker_d", {})
    memory_store("opinion", "other_lonely_d", {})
    # link なし

    state = _make_state_with_intent("lonely_marker_d")
    result = []
    ok = True
    try:
        result = get_relevant_memories(state, limit=8, use_links=True)
    except Exception as e:
        print(f"  exception: {e}")
        ok = False
    link_entries = [m for m in result if m.get("_retrieval_via") == "link"] if ok else []
    return all([
        _assert(ok, "exception 出ない"),
        _assert(link_entries == [], "link 由来 entry なし"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    try:
        groups = [
            ("use_links=False 後方互換",
             test_default_use_links_false_preserves_behavior),
            ("use_links=True merge", test_use_links_true_merges_link_neighbors),
            ("retrieval marker 属性", test_retrieval_marker_attributes),
            ("link 無しで no error", test_use_links_no_links_no_error),
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
    finally:
        shutil.rmtree(_tmp_root, ignore_errors=True)
