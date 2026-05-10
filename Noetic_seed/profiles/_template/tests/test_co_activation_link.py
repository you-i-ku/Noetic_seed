"""co_activation 構造的生成 helper テスト (v0.5 Phase 5 Slice 4 F-002).

設計: co_activation は対称的関係 (= 同時 retrieval された事実) のため、
**生成時に双方向 2 本 (A→B + B→A) を同時 append** してデータ層で対称性を
表現する (Codex Slice 4 review P1-3 ゆう案、2026-05-10 確定)。1 pair = 2 link、
max_pairs は pair 単位 cap。

成功条件:
  - memories >= 2 で pair 生成 (top_n × (top_n-1) / 2 pair = ×2 link)
  - memories < 2 で早期 return (生成 0)
  - 既存 co_activation link あれば dedup (新規生成 0)
  - 既存 co_activation の方向逆 (b→a) でも dedup (frozenset 対称)
  - max_pairs 上限で打ち止め (pair 単位)
  - link_type="co_activation" / strength == confidence == 0.5 (initial)
  - last_used_cycle が current_cycle で記録される
  - LLM 通さない (構造的事実生成、reason に "co-retrieved" 文字列)
  - 双方向化: 1 pair で from_id/to_id が逆向き 2 link 生成される

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_co_activation_link.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.memory as memory_mod
import core.memory_links as ml_mod
import core.tag_registry as tr
from core.memory_links import (
    generate_co_activation_links,
    list_links,
    LINK_SCAN_LIMIT,
    CO_ACTIVATION_INITIAL_CONFIDENCE,
    CO_ACTIVATION_MAX_PAIRS_PER_CALL,
    CO_ACTIVATION_TOP_N_MEMORIES,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup(tmp_path: Path):
    from core import config as cfg
    cfg.MEMORY_DIR = tmp_path
    memory_mod.MEMORY_DIR = tmp_path
    ml_mod.MEMORY_DIR = tmp_path
    reg_file = tmp_path / "registered_tags.json"
    tr._reset_for_testing(registry_file=reg_file)
    tr.register_standard_tags()


def _make_memories(n: int) -> list:
    """test 用 memory entry list (id 'm1', 'm2', ...)。"""
    return [{"id": f"m{i}", "content": f"c{i}", "network": "default"}
            for i in range(1, n + 1)]


# ============================================================
# Section A: 基本生成
# ============================================================

def test_a1_pair_count_from_top_n_memories(tmp_path: Path):
    print("== top_n=5 memory で 5C2 = 10 pair × 2 = 20 link 生成 (双方向化) ==")
    _setup(tmp_path)
    mems = _make_memories(5)
    created = generate_co_activation_links(mems, current_cycle=10)
    return _assert(len(created) == 20,
                   f"5C2 = 10 pair × 2 = 20 link (got: {len(created)})")


def test_a2_link_type_and_initial_strength(tmp_path: Path):
    print("== link_type='co_activation' / strength == confidence == 0.5 ==")
    _setup(tmp_path)
    mems = _make_memories(2)
    created = generate_co_activation_links(mems, current_cycle=10)
    if len(created) != 2:
        return _assert(False, f"1 pair = 2 link 生成期待 (got: {len(created)})")
    return all([
        _assert(all(e["link_type"] == "co_activation" for e in created),
                f"両 link 共 link_type='co_activation' (got: {[e['link_type'] for e in created]})"),
        _assert(all(abs(e["confidence"] - CO_ACTIVATION_INITIAL_CONFIDENCE) < 1e-9
                    for e in created),
                f"両 link 共 confidence = {CO_ACTIVATION_INITIAL_CONFIDENCE}"),
        _assert(all(abs(e["strength"] - CO_ACTIVATION_INITIAL_CONFIDENCE) < 1e-9
                    for e in created),
                "両 link 共 strength = confidence (initial)"),
        _assert(all(e["last_used_cycle"] == 10 for e in created),
                "両 link 共 last_used_cycle = current_cycle"),
        _assert(all("co-retrieved" in e["reason"] for e in created),
                "両 link 共 reason に 'co-retrieved'"),
    ])


def test_a2b_bidirectional_edges(tmp_path: Path):
    print("== 1 pair で双方向 2 link (A→B + B→A) 生成 ==")
    _setup(tmp_path)
    mems = _make_memories(2)
    created = generate_co_activation_links(mems, current_cycle=10)
    if len(created) != 2:
        return _assert(False, f"双方向 2 link 期待 (got: {len(created)})")
    pairs = {(e["from_id"], e["to_id"]) for e in created}
    return all([
        _assert(("m1", "m2") in pairs, f"A→B (m1→m2) 存在 (got: {pairs})"),
        _assert(("m2", "m1") in pairs, f"B→A (m2→m1) 存在 (got: {pairs})"),
        _assert(len({e["id"] for e in created}) == 2,
                "2 link は別 id (uuid 独立)"),
    ])


def test_a3_jsonl_persisted(tmp_path: Path):
    print("== memory_links.jsonl に双方向 2 行 append される ==")
    _setup(tmp_path)
    mems = _make_memories(2)
    generate_co_activation_links(mems, current_cycle=10)
    fpath = tmp_path / "memory_links.jsonl"
    if not fpath.exists():
        return _assert(False, "jsonl 作成されてない")
    lines = [l for l in fpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    return _assert(len(lines) == 2,
                   f"jsonl に 2 行 (双方向 2 link、got: {len(lines)})")


# ============================================================
# Section B: 識別力 (CLAUDE.md §5)
# ============================================================

def test_b1_lt_two_memories_zero_created(tmp_path: Path):
    print("== memories < 2 で生成 0 (早期 return) ==")
    _setup(tmp_path)
    return all([
        _assert(generate_co_activation_links([], current_cycle=10) == [],
                "空 list で生成 0"),
        _assert(generate_co_activation_links(_make_memories(1), current_cycle=10) == [],
                "1 件で生成 0"),
    ])


def test_b2_dedup_existing_co_activation(tmp_path: Path):
    print("== 既存 co_activation link あれば dedup (新規 0) ==")
    _setup(tmp_path)
    mems = _make_memories(2)
    first = generate_co_activation_links(mems, current_cycle=10)
    second = generate_co_activation_links(mems, current_cycle=11)
    return all([
        _assert(len(first) == 2, f"初回 1 pair = 2 link (got: {len(first)})"),
        _assert(len(second) == 0, f"2 回目 dedup で 0 (got: {len(second)})"),
    ])


def test_b3_dedup_symmetric_frozenset(tmp_path: Path):
    print("== (a, b) と (b, a) は同一 pair (frozenset 対称) ==")
    _setup(tmp_path)
    mems_ab = _make_memories(2)             # m1, m2
    mems_ba = list(reversed(mems_ab))        # m2, m1
    first = generate_co_activation_links(mems_ab, current_cycle=10)
    second = generate_co_activation_links(mems_ba, current_cycle=11)
    return all([
        _assert(len(first) == 2, f"初回 (a, b) 1 pair = 2 link (got: {len(first)})"),
        _assert(len(second) == 0,
                f"(b, a) 順でも dedup (got: {len(second)})"),
    ])


def test_b4_max_pairs_cap(tmp_path: Path):
    print("== max_pairs cap で打ち止め (pair 単位) ==")
    _setup(tmp_path)
    mems = _make_memories(10)   # 10C2 = 45 pair 候補
    created = generate_co_activation_links(mems, current_cycle=10, max_pairs=3)
    return _assert(len(created) == 6,   # 3 pair × 2 link = 6
                   f"max_pairs=3 pair = 6 link (got: {len(created)})")


def test_b5_top_n_limits_candidates(tmp_path: Path):
    print("== top_n でペア候補絞込 ==")
    _setup(tmp_path)
    mems = _make_memories(10)
    created = generate_co_activation_links(mems, current_cycle=10, top_n=3)
    return _assert(len(created) == 6,   # 3C2 = 3 pair × 2 link = 6
                   f"top_n=3 で 3C2=3 pair × 2 = 6 link (got: {len(created)})")


def test_b6_no_self_pair(tmp_path: Path):
    print("== from_id == to_id (自己 pair) は skip ==")
    _setup(tmp_path)
    # 同 id を含む list (実運用で起きえない、識別力強化)
    mems = [
        {"id": "m1", "content": "x", "network": "default"},
        {"id": "m1", "content": "y", "network": "default"},
    ]
    created = generate_co_activation_links(mems, current_cycle=10)
    return _assert(len(created) == 0,
                   f"同 id pair は skip (got: {len(created)})")


# ============================================================
# Section C: 定数値 (Slice 4 確定)
# ============================================================

def test_c1_constants():
    print("== Slice 4 定数値 ==")
    from core.memory_links import LINK_CONFIDENCE_THRESHOLD
    return all([
        _assert(CO_ACTIVATION_INITIAL_CONFIDENCE == 0.7,
                f"INITIAL_CONFIDENCE = 0.7 (got: {CO_ACTIVATION_INITIAL_CONFIDENCE})"),
        _assert(CO_ACTIVATION_INITIAL_CONFIDENCE >= LINK_CONFIDENCE_THRESHOLD,
                f"INITIAL_CONFIDENCE ({CO_ACTIVATION_INITIAL_CONFIDENCE}) >= "
                f"follow_links 走査閾値 ({LINK_CONFIDENCE_THRESHOLD}) "
                f"= retrieval 経路から実際に参照される (Codex 2 周目 P1 fix)"),
        _assert(CO_ACTIVATION_MAX_PAIRS_PER_CALL == 10,
                f"MAX_PAIRS = 10 (got: {CO_ACTIVATION_MAX_PAIRS_PER_CALL})"),
        _assert(CO_ACTIVATION_TOP_N_MEMORIES == 5,
                f"TOP_N = 5 (got: {CO_ACTIVATION_TOP_N_MEMORIES})"),
    ])


def run_all():
    print("=" * 60)
    print("test_co_activation_link.py (v0.5 Phase 5 Slice 4 F-002)")
    print("=" * 60)
    results = []
    with tempfile.TemporaryDirectory() as td:
        for fn in [
            test_a1_pair_count_from_top_n_memories,
            test_a2_link_type_and_initial_strength,
            test_a2b_bidirectional_edges,
            test_a3_jsonl_persisted,
            test_b1_lt_two_memories_zero_created,
            test_b2_dedup_existing_co_activation,
            test_b3_dedup_symmetric_frozenset,
            test_b4_max_pairs_cap,
            test_b5_top_n_limits_candidates,
            test_b6_no_self_pair,
        ]:
            sub = Path(td) / fn.__name__
            sub.mkdir(exist_ok=True)
            results.append(fn(sub))
    results.append(test_c1_constants())
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"結果: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
