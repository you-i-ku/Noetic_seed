"""段階11-D Phase 5 Step 5.2: reflect の cluster section + NOTES 経路 integration test.

検証対象:
  Section A: _build_cluster_sections (純粋関数)
    - 空 clusters → 空文字列
    - label + 代表 sample content が prompt に出る
    - sample は 2 件まで (cost 抑制)
    - label 空 → "(未分類)" 表示
  Section B: _parse_reflection NOTES (Phase 5 統一枠)
    - NOTES 行が memory_store(network=None) で保存される
    - reconciliation hook (_state=state) が memory_store kwargs に継承
    - 旧 OPINIONS / ENTITIES section は無視 (parse されない)
    - confidence parse + 範囲外 clamp は disposition 経路で維持
  Section C: 戻り値 dict 構造 (Phase 5 統一)
    - "notes" キー (旧 opinions / entities 撤去)
    - self_disp_delta / attr_disp_delta は維持 (段階11-A 設計)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" \\
      tests/test_reflect_cluster_integration.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.reflection as refl
from core.reflection import _build_cluster_sections, _parse_reflection


def _assert(cond, msg):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


# ============================================================
# Section A: _build_cluster_sections (純粋関数)
# ============================================================

def test_cluster_sections_empty():
    print("== _build_cluster_sections: 空 clusters → 空文字列 ==")
    return _assert(_build_cluster_sections([], {}) == "", "空 clusters で空文字列")


def test_cluster_sections_with_label_and_samples():
    print("== _build_cluster_sections: label + sample content 表示 ==")
    clusters = [
        {"cluster_id": "c1", "label": "動物観察",
         "memory_ids": ["m1", "m2", "m3"], "method": "hybrid"},
        {"cluster_id": "c2", "label": "",
         "memory_ids": ["m4"], "method": "hybrid"},
    ]
    memory_index = {
        "m1": {"id": "m1", "content": "猫が寝てる"},
        "m2": {"id": "m2", "content": "犬と散歩"},
        "m3": {"id": "m3", "content": "鳥が鳴いてる"},
        "m4": {"id": "m4", "content": "コードを書く"},
    }
    result = _build_cluster_sections(clusters, memory_index)
    return all([
        _assert("動物観察" in result, "label が prompt 内"),
        _assert("(未分類)" in result, "label 空 → (未分類)"),
        _assert("猫が寝てる" in result, "代表 sample 1 件目 content"),
        _assert("コードを書く" in result, "別 cluster の content も"),
        _assert("3 件" in result, "件数表示"),
        _assert("posterior" in result, "posterior の説明文付き (永続化しない注記)"),
    ])


def test_cluster_sections_sample_limit():
    print("== _build_cluster_sections: sample 上限 2 件 (cost 抑制) ==")
    clusters = [
        {"cluster_id": "c1", "label": "x",
         "memory_ids": ["m1", "m2", "m3", "m4"], "method": "hybrid"},
    ]
    memory_index = {
        f"m{i}": {"id": f"m{i}", "content": f"content_{i}"}
        for i in range(1, 5)
    }
    result = _build_cluster_sections(clusters, memory_index)
    return all([
        _assert("content_1" in result, "1 件目表示"),
        _assert("content_2" in result, "2 件目表示"),
        _assert("content_3" not in result, "3 件目以降は省略 (cost 抑制)"),
    ])


def test_cluster_sections_missing_memory_index():
    """memory_id が memory_index に無い場合は skip (graceful)"""
    print("== _build_cluster_sections: memory_index 欠落で graceful ==")
    clusters = [
        {"cluster_id": "c1", "label": "test",
         "memory_ids": ["nonexistent"], "method": "hybrid"},
    ]
    result = _build_cluster_sections(clusters, {})
    return all([
        _assert("test" in result, "label は表示される"),
        _assert("nonexistent" not in result, "missing memory id は本文に出ない"),
    ])


# ============================================================
# Section B: _parse_reflection NOTES (reconciliation hook 維持)
# ============================================================

def test_parse_notes_to_untagged_with_state_hook():
    """NOTES 行は network=None で memory_store、_state=state も継承。"""
    print("== _parse_reflection: NOTES → memory_store(network=None) + _state hook ==")
    captured = []
    orig_ms = refl.memory_store

    def mock_memory_store(*args, **kwargs):
        captured.append({"args": args, "kwargs": kwargs})
        return {"id": f"mock_{len(captured)}", "content": kwargs.get("content")}

    refl.memory_store = mock_memory_store
    try:
        state = {"log": [], "self": {"name": "iku"}}
        result = _parse_reflection("""
NOTES:
- 観察には視点がある (confidence: 0.8)
- 整理することで気づきが生まれる (confidence: 0.6)

SELF_DISPOSITION:
- curiosity_delta: +0.05
""", state)

        return all([
            _assert(len(captured) == 2, f"NOTES 2 行 → memory_store 2 回 (got {len(captured)})"),
            _assert(
                all(c["kwargs"].get("network") is None for c in captured),
                "全 memory_store で network=None (untagged path)",
            ),
            _assert(
                all(c["kwargs"].get("_state") is state for c in captured),
                "全 memory_store で _state=state 継承 (reconciliation hook)",
            ),
            _assert(
                "観察には視点" in (captured[0]["kwargs"].get("content") or ""),
                "1 件目の content が NOTES 1 行目から",
            ),
            _assert(
                captured[0]["kwargs"].get("metadata", {}).get("confidence") == 0.8,
                "confidence parse (0.8)",
            ),
            _assert(
                captured[1]["kwargs"].get("metadata", {}).get("confidence") == 0.6,
                "2 件目 confidence (0.6)",
            ),
            _assert(
                captured[0]["kwargs"].get("origin") == "reflection",
                "origin=reflection",
            ),
            _assert(
                len(result.get("notes", [])) == 2,
                "戻り値 notes に 2 件",
            ),
            _assert(
                result.get("self_disp_delta", {}).get("curiosity") == 0.05,
                "SELF_DISPOSITION も並列 parse (touch しない経路)",
            ),
        ])
    finally:
        refl.memory_store = orig_ms


def test_parse_ignores_old_opinions_entities():
    """旧 OPINIONS / ENTITIES section は parse されない (撤去確認)。"""
    print("== _parse_reflection: 旧 OPINIONS/ENTITIES section は無視 ==")
    captured = []
    orig_ms = refl.memory_store

    def mock_memory_store(*args, **kwargs):
        captured.append(kwargs.get("content", ""))
        return {"id": "mock_id"}

    refl.memory_store = mock_memory_store
    try:
        state = {"log": [], "self": {}}
        result = _parse_reflection("""
OPINIONS:
- 旧形式の主張 (confidence: 0.7)

ENTITIES:
- name: ゆう, content: 開発者
""", state)
        return all([
            _assert(len(captured) == 0, f"memory_store は呼ばれない (got {len(captured)})"),
            _assert(result.get("notes") == [], "notes 空"),
            _assert("opinions" not in result, "戻り値に opinions キーなし"),
            _assert("entities" not in result, "戻り値に entities キーなし"),
        ])
    finally:
        refl.memory_store = orig_ms


# ============================================================
# Section C: 戻り値 dict 構造
# ============================================================

def test_return_dict_structure():
    """戻り値は notes / self_disp_delta / attr_disp_delta の 3 キー。"""
    print("== _parse_reflection: 戻り値 3 キー構造 ==")
    orig_ms = refl.memory_store
    refl.memory_store = lambda *a, **kw: {"id": "mock"}
    try:
        state = {"log": [], "self": {}}
        result = _parse_reflection("NOTES:\n- 気づき\n", state)
        return all([
            _assert("notes" in result, "notes キー存在"),
            _assert("self_disp_delta" in result, "self_disp_delta キー存在"),
            _assert("attr_disp_delta" in result, "attr_disp_delta キー存在"),
            _assert("opinions" not in result, "opinions キー撤去確認"),
            _assert("entities" not in result, "entities キー撤去確認"),
        ])
    finally:
        refl.memory_store = orig_ms


# ============================================================
# Runner
# ============================================================

def run_all():
    print("=" * 60)
    print("test_reflect_cluster_integration.py (段階11-D Phase 5 Step 5.2)")
    print("=" * 60)
    results = [
        test_cluster_sections_empty(),
        test_cluster_sections_with_label_and_samples(),
        test_cluster_sections_sample_limit(),
        test_cluster_sections_missing_memory_index(),
        test_parse_notes_to_untagged_with_state_hook(),
        test_parse_ignores_old_opinions_entities(),
        test_return_dict_structure(),
    ]
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"結果: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
