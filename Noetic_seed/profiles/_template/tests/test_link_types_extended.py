"""link type 拡張 + schema 拡張テスト (段階11-D Phase 2 Step 2.1-2.4).

成功条件:
  - LINK_TYPES が 8 type (既存 5 + 追加 3 = co_activation/semantic/supporting)
  - _build_link_prompt に "semantic" が候補として含まれる (LLM judge 経由)
  - _build_link_prompt から "co_activation" / "supporting" は除外 (Phase 4 hook)
  - _build_link_entry が strength / last_used / usage_count を含む
  - strength 初期値 = confidence、last_used 初期値 = created_at、usage_count = 0
  - _parse_link_response が新 type (co_activation/semantic/supporting) を受け入れる
  - 既存 5 type (similar 等) の挙動保持 (回帰確認)
  - _find_memory_entry_by_id が UNTAGGED_NETWORK を走査対象に含む (Phase 1 hotfix)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_link_types_extended.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.memory as memory_mod
import core.tag_registry as tr
from core.memory import memory_store, UNTAGGED_NETWORK
from core.memory_links import (
    LINK_TYPES,
    _build_link_prompt,
    _parse_link_response,
    _build_link_entry,
    _find_memory_entry_by_id,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup(tmp_path: Path):
    memory_mod.MEMORY_DIR = tmp_path
    reg_file = tmp_path / "registered_tags.json"
    tr._reset_for_testing(registry_file=reg_file)
    tr.register_standard_tags()
    return reg_file


def test_link_types_count_and_members():
    print("== LINK_TYPES が 8 type 含む (既存 5 + 追加 3) ==")
    expected = {
        "similar", "contradict", "elaborate", "causal", "temporal",
        "co_activation", "semantic", "supporting",
    }
    return all([
        _assert(len(LINK_TYPES) == 8, f"LINK_TYPES = 8 type (got: {len(LINK_TYPES)})"),
        _assert(set(LINK_TYPES) == expected,
                f"LINK_TYPES に期待 type 全て含む (got: {set(LINK_TYPES)})"),
    ])


def test_prompt_contains_semantic():
    print("== _build_link_prompt に semantic が含まれる ==")
    a = {"id": "a", "content": "test A", "network": "opinion", "keywords": ["x"]}
    b = {"id": "b", "content": "test B", "network": "opinion", "keywords": ["y"]}
    prompt = _build_link_prompt(a, b)
    return all([
        _assert("semantic" in prompt, "semantic が prompt に含まれる"),
        _assert("similar" in prompt, "既存 similar も含まれる (回帰)"),
        _assert("temporal" in prompt, "既存 temporal も含まれる (回帰)"),
    ])


def test_prompt_excludes_phase4_types():
    print("== _build_link_prompt から co_activation / supporting は除外 (Phase 4 hook) ==")
    a = {"id": "a", "content": "test A", "network": "opinion", "keywords": []}
    b = {"id": "b", "content": "test B", "network": "opinion", "keywords": []}
    prompt = _build_link_prompt(a, b)
    return all([
        _assert("co_activation" not in prompt, "co_activation は LLM judge 候補外"),
        _assert("supporting" not in prompt, "supporting は LLM judge 候補外"),
    ])


def test_link_entry_new_fields():
    print("== _build_link_entry に strength / last_used / usage_count 含む ==")
    a = {"id": "mem_aaa", "content": "A"}
    b = {"id": "mem_bbb", "content": "B"}
    verdict = {"link_type": "similar", "confidence": 0.85, "reason": "test"}
    entry = _build_link_entry(a, b, verdict)
    return all([
        _assert("strength" in entry, "strength field 存在"),
        _assert("last_used" in entry, "last_used field 存在"),
        _assert("usage_count" in entry, "usage_count field 存在"),
        _assert(entry["strength"] == 0.85,
                f"strength 初期値 = confidence (got: {entry['strength']})"),
        _assert(entry["last_used"] == entry["created_at"],
                f"last_used 初期値 = created_at"),
        _assert(entry["usage_count"] == 0,
                f"usage_count 初期値 = 0 (got: {entry['usage_count']})"),
    ])


def test_link_entry_backward_compat():
    print("== _build_link_entry の既存 field も保持 (回帰) ==")
    a = {"id": "mem_aaa", "content": "A"}
    b = {"id": "mem_bbb", "content": "B"}
    verdict = {"link_type": "causal", "confidence": 0.75, "reason": "r"}
    entry = _build_link_entry(a, b, verdict)
    return all([
        _assert(entry["link_type"] == "causal", "link_type 保持"),
        _assert(entry["confidence"] == 0.75, "confidence 保持"),
        _assert(entry["from_id"] == "mem_aaa", "from_id 保持"),
        _assert(entry["to_id"] == "mem_bbb", "to_id 保持"),
        _assert("perspective" in entry, "perspective 保持 (11-A)"),
        _assert(entry["reason"] == "r", "reason 保持"),
        _assert(entry["id"].startswith("link_"), "id 採番保持"),
    ])


def test_parse_accepts_new_types():
    print("== _parse_link_response が新 type 受け入れ ==")
    # co_activation
    p1 = _parse_link_response('{"link_type": "co_activation", "confidence": 0.8, "reason": "x"}')
    # semantic
    p2 = _parse_link_response('{"link_type": "semantic", "confidence": 0.7, "reason": "y"}')
    # supporting
    p3 = _parse_link_response('{"link_type": "supporting", "confidence": 0.9, "reason": "z"}')
    return all([
        _assert(p1["link_type"] == "co_activation",
                f"co_activation 受け入れ (got: {p1['link_type']})"),
        _assert(p2["link_type"] == "semantic",
                f"semantic 受け入れ (got: {p2['link_type']})"),
        _assert(p3["link_type"] == "supporting",
                f"supporting 受け入れ (got: {p3['link_type']})"),
    ])


def test_parse_existing_types_regression():
    print("== _parse_link_response が既存 5 type 保持 (回帰) ==")
    results = []
    for t in ("similar", "contradict", "elaborate", "causal", "temporal"):
        p = _parse_link_response(f'{{"link_type": "{t}", "confidence": 0.8, "reason": "r"}}')
        results.append(_assert(p["link_type"] == t, f"{t} 保持"))
    return all(results)


def test_parse_unknown_type_to_none():
    print("== _parse_link_response が未知 type を 'none' にフォールバック ==")
    p = _parse_link_response('{"link_type": "weird_type", "confidence": 0.8, "reason": "r"}')
    return _assert(p["link_type"] == "none", f"未知 type → none (got: {p['link_type']})")


def test_find_memory_entry_includes_untagged(tmp_path: Path):
    print("== _find_memory_entry_by_id が untagged memory を見つける (Phase 1 hotfix) ==")
    _setup(tmp_path)
    untagged_entry = memory_store(network=None, content="untagged target",
                                   _auto_metadata=False)
    tagged_entry = memory_store(network="opinion", content="tagged target",
                                 _auto_metadata=False)
    found_untagged = _find_memory_entry_by_id(untagged_entry["id"])
    found_tagged = _find_memory_entry_by_id(tagged_entry["id"])
    return all([
        _assert(found_untagged is not None,
                "untagged memory が _find_memory_entry_by_id で見つかる"),
        _assert(found_untagged["network"] == UNTAGGED_NETWORK,
                f"取得 entry の network = {UNTAGGED_NETWORK}"),
        _assert(found_tagged is not None,
                "tagged memory も継続して見つかる (回帰)"),
        _assert(found_tagged["network"] == "opinion", "tagged network 保持"),
    ])


def run_all():
    print("=" * 60)
    print("test_link_types_extended.py (段階11-D Phase 2 Step 2.1-2.4)")
    print("=" * 60)
    results = []
    # Section A: LINK_TYPES / prompt 構造 (tmp_path 不要)
    results.append(test_link_types_count_and_members())
    results.append(test_prompt_contains_semantic())
    results.append(test_prompt_excludes_phase4_types())
    # Section B: schema 拡張 (tmp_path 不要、純粋関数)
    results.append(test_link_entry_new_fields())
    results.append(test_link_entry_backward_compat())
    # Section C: parse 挙動 (tmp_path 不要)
    results.append(test_parse_accepts_new_types())
    results.append(test_parse_existing_types_regression())
    results.append(test_parse_unknown_type_to_none())
    # Section D: Phase 1 hotfix (tmp_path 必要)
    with tempfile.TemporaryDirectory() as td:
        sub = Path(td) / "find_untagged"
        sub.mkdir(exist_ok=True)
        results.append(test_find_memory_entry_includes_untagged(sub))
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"結果: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
