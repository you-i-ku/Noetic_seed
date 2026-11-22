"""段階13 Phase 0.4 — check_raw_subjective_gap テスト。

PLAN §18-C.5: tool 実行直後の raw event と subjective entry の意味的 gap を
bge-m3 cosine 距離で検出、check_on_write の sibling として段階10 EC 経路に流す。

閾値は entity_resolver の Tier 流用 (新規マジックナンバー 0):
  - record_threshold    = 1 - EMBEDDING_DIFFERENT_THRESHOLD ≒ 0.30
  - ambiguous_threshold = 1 - EMBEDDING_SAME_THRESHOLD      ≒ 0.15

挙動マトリクス (CLAUDE.md §6 docstring 約束網羅):
  | 状態 | 期待戻り値 | record? |
  | id 不在 | None | No |
  | raw_part 不在 (subj_part のみ) | None | No |
  | subj_part 不在 (raw_part のみ) | None | No |
  | raw_text 空 (result+args 空) | None | No |
  | subj_text 空 (intent+expect 空) | None | No |
  | is_vector_ready False | None | No |
  | embed_fn 例外 | None | No |
  | gap < ambiguous (0.15) | None | No |
  | ambiguous (0.15) <= gap < record (0.30) | None | No (Tier 3 lean skip) |
  | gap >= record (0.30) | verdict dict | Yes |

CLAUDE.md §5 識別力: cosine_fn を mock で固定値返す → gap が controlled に
出るため、誤実装 (例: 閾値判定逆) でも test fail で検出可。

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_raw_subjective_gap.py
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reconciliation import check_raw_subjective_gap
from core.entity_resolver import (
    EMBEDDING_SAME_THRESHOLD,
    EMBEDDING_DIFFERENT_THRESHOLD,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _make_state(entry_id="e1",
                raw_fields=None,
                subj_fields=None) -> dict:
    """raw_events / subjective_entries に id 共有 entry を入れた最小 state。"""
    if raw_fields is None:
        raw_fields = {"result": "OK", "args": {"x": 1}}
    if subj_fields is None:
        subj_fields = {"intent": "発話", "expect": "応答"}
    return {
        "raw_events": [{"id": entry_id, **raw_fields}],
        "subjective_entries": [{"id": entry_id, **subj_fields}],
    }


def _make_embed_fn():
    """embed_fn mock: 2 件入力なら 2 件 dummy vec を返す。"""
    def fn(texts):
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]][: len(texts)]
    return fn


def _make_cosine_fn(value: float):
    """cosine_fn mock: 固定値を返す。gap は 1 - value で算出される。"""
    def fn(a, b):
        return value
    return fn


def _patch_vector_ready_true():
    """is_vector_ready が True を返すよう module 内 import を patch する。

    check_raw_subjective_gap 内で `from core.embedding import is_vector_ready`
    を関数内 import してるため、core.embedding.is_vector_ready を patch する。
    """
    return patch("core.embedding.is_vector_ready", return_value=True)


# ============================================================
# 早期 return パターン (None 返却、record しない)
# ============================================================

def test_no_id_returns_none():
    print("== entry に id 不在 → None (skip) ==")
    state = _make_state()
    with _patch_vector_ready_true():
        res = check_raw_subjective_gap(state, {})  # entry に id なし
    return all([
        _assert(res is None, "戻り値 None"),
        _assert(state.get("prediction_error_history_ec", []) == [],
                "EC history 増えない"),
    ])


def test_raw_part_missing_returns_none():
    print("== 同 id の raw_part 不在 → None ==")
    state = {"raw_events": [],
             "subjective_entries": [{"id": "e1", "intent": "x", "expect": "y"}]}
    with _patch_vector_ready_true():
        res = check_raw_subjective_gap(state, {"id": "e1"})
    return _assert(res is None, "raw_part 不在で None")


def test_subj_part_missing_returns_none():
    print("== 同 id の subj_part 不在 → None ==")
    state = {"raw_events": [{"id": "e1", "result": "x", "args": "y"}],
             "subjective_entries": []}
    with _patch_vector_ready_true():
        res = check_raw_subjective_gap(state, {"id": "e1"})
    return _assert(res is None, "subj_part 不在で None")


def test_empty_raw_text_returns_none():
    print("== raw_text (result+args) 空 → None ==")
    state = _make_state(raw_fields={"result": "", "args": ""})
    with _patch_vector_ready_true():
        res = check_raw_subjective_gap(state, {"id": "e1"})
    return _assert(res is None, "raw_text 空で None")


def test_empty_subj_text_returns_none():
    print("== subj_text (intent+expect) 空 → None ==")
    state = _make_state(subj_fields={"intent": "", "expect": ""})
    with _patch_vector_ready_true():
        res = check_raw_subjective_gap(state, {"id": "e1"})
    return _assert(res is None, "subj_text 空で None")


def test_vector_not_ready_returns_none():
    print("== is_vector_ready False → None (graceful skip) ==")
    state = _make_state()
    with patch("core.embedding.is_vector_ready", return_value=False):
        res = check_raw_subjective_gap(state, {"id": "e1"})
    return _assert(res is None, "is_vector_ready False で skip")


def test_embed_fn_exception_returns_none():
    print("== embed_fn 例外 → None (graceful skip) ==")
    def failing_embed(texts):
        raise RuntimeError("intentional failure")
    state = _make_state()
    with _patch_vector_ready_true():
        res = check_raw_subjective_gap(
            state, {"id": "e1"},
            embed_fn=failing_embed,
            cosine_fn=_make_cosine_fn(0.5),
        )
    return _assert(res is None, "embed 例外で graceful skip")


# ============================================================
# 閾値判定 (boundary 含む)
# ============================================================

def test_gap_below_ambiguous_skip():
    """gap < 0.15 (同義領域) → skip (誤実装で record してたら検出)。"""
    print("== gap=0.05 (cosine=0.95、同義領域) → skip ==")
    state = _make_state()
    with _patch_vector_ready_true():
        res = check_raw_subjective_gap(
            state, {"id": "e1"},
            embed_fn=_make_embed_fn(),
            cosine_fn=_make_cosine_fn(0.95),  # gap=0.05、< 0.15
        )
    return all([
        _assert(res is None, "gap=0.05 で skip"),
        _assert(state.get("prediction_error_history_ec", []) == [],
                "EC history 増えない"),
    ])


def test_gap_in_ambiguous_zone_skip():
    """0.15 <= gap < 0.30 (Tier 3 ambiguous) → Phase 0.4 lean skip。

    識別力 (CLAUDE.md §5): もし誤実装で 「gap >= ambiguous で record」 だと
    この test は fail する (= ambiguous でも record してしまうから)。
    """
    print("== gap=0.22 (Tier 3 ambiguous) → skip (Phase 0.4 lean) ==")
    state = _make_state()
    with _patch_vector_ready_true():
        res = check_raw_subjective_gap(
            state, {"id": "e1"},
            embed_fn=_make_embed_fn(),
            cosine_fn=_make_cosine_fn(0.78),  # gap=0.22、ambiguous
        )
    return all([
        _assert(res is None, "gap=0.22 ambiguous で skip"),
        _assert(state.get("prediction_error_history_ec", []) == [],
                "EC history 増えない"),
    ])


def test_gap_above_threshold_records():
    """gap >= 0.30 → record + verdict 返却。"""
    print("== gap=0.60 (cosine=0.40、確定 gap) → record + verdict ==")
    state = _make_state()
    with _patch_vector_ready_true():
        res = check_raw_subjective_gap(
            state, {"id": "e1"},
            embed_fn=_make_embed_fn(),
            cosine_fn=_make_cosine_fn(0.40),  # gap=0.60、>= 0.30
        )
    return all([
        _assert(res is not None, "verdict 返却"),
        _assert(res["id"] == "e1", f"verdict.id=e1 (got {res['id']})"),
        _assert(abs(res["gap"] - 0.60) < 1e-6,
                f"verdict.gap=0.60 (got {res['gap']})"),
        _assert(len(state.get("prediction_error_history_ec", [])) == 1,
                "EC history 1 件追加"),
        _assert(abs(state["prediction_error_history_ec"][0] - 0.60) < 1e-6,
                "EC magnitude=gap"),
    ])


def test_gap_at_record_boundary_records():
    """gap == record_threshold (0.30) ジャスト → record (>= の境界)。"""
    print("== gap=0.30 boundary (cosine=0.70) → record ==")
    state = _make_state()
    record_threshold = 1.0 - EMBEDDING_DIFFERENT_THRESHOLD
    with _patch_vector_ready_true():
        res = check_raw_subjective_gap(
            state, {"id": "e1"},
            embed_fn=_make_embed_fn(),
            cosine_fn=_make_cosine_fn(EMBEDDING_DIFFERENT_THRESHOLD),
        )
    return all([
        _assert(res is not None, "boundary で record"),
        _assert(abs(res["gap"] - record_threshold) < 1e-6,
                f"gap=record_threshold={record_threshold}"),
    ])


def test_gap_at_ambiguous_boundary_skips():
    """gap == ambiguous_threshold (0.15) ジャスト → skip (Tier 3 lean ゾーン)。"""
    print("== gap=0.15 boundary (cosine=0.85) → skip ==")
    state = _make_state()
    with _patch_vector_ready_true():
        res = check_raw_subjective_gap(
            state, {"id": "e1"},
            embed_fn=_make_embed_fn(),
            cosine_fn=_make_cosine_fn(EMBEDDING_SAME_THRESHOLD),
        )
    return _assert(res is None, "boundary 0.15 で skip (record_threshold 未満)")


# ============================================================
# 記録の content 検証 (CLAUDE.md §5 排他性 assert)
# ============================================================

def test_record_source_label():
    """source='raw_subj_gap' で by_source history に記録される。"""
    print("== record 時の source label = 'raw_subj_gap' ==")
    state = _make_state()
    with _patch_vector_ready_true():
        check_raw_subjective_gap(
            state, {"id": "e1"},
            embed_fn=_make_embed_fn(),
            cosine_fn=_make_cosine_fn(0.40),  # gap=0.60、record
        )
    by_source = state.get("prediction_error_history_by_source", {})
    return all([
        _assert("raw_subj_gap" in by_source, "source='raw_subj_gap' key 存在"),
        _assert(len(by_source["raw_subj_gap"]) == 1, "raw_subj_gap history 1 件"),
        _assert("reconciliation" not in by_source,
                "誤って source='reconciliation' で記録されてない (sibling 区別)"),
    ])


def test_verdict_excerpts_match_text():
    """verdict.raw_excerpt / subj_excerpt が text の prefix を含む。"""
    print("== verdict.raw_excerpt / subj_excerpt 内容検証 ==")
    state = _make_state(
        raw_fields={"result": "返答完了", "args": {"channel": "device"}},
        subj_fields={"intent": "iku に話しかける", "expect": "iku が返事する"},
    )
    with _patch_vector_ready_true():
        res = check_raw_subjective_gap(
            state, {"id": "e1"},
            embed_fn=_make_embed_fn(),
            cosine_fn=_make_cosine_fn(0.40),
        )
    return all([
        _assert("返答完了" in res["raw_excerpt"], "raw_excerpt に result 含む"),
        _assert("iku に話しかける" in res["subj_excerpt"],
                "subj_excerpt に intent 含む"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    print("test_raw_subjective_gap.py (段階13 Phase 0.4)")
    print("=" * 60)

    groups = [
        ("entry id 不在 → None", test_no_id_returns_none),
        ("raw_part 不在 → None", test_raw_part_missing_returns_none),
        ("subj_part 不在 → None", test_subj_part_missing_returns_none),
        ("raw_text 空 → None", test_empty_raw_text_returns_none),
        ("subj_text 空 → None", test_empty_subj_text_returns_none),
        ("is_vector_ready False → None", test_vector_not_ready_returns_none),
        ("embed_fn 例外 → None", test_embed_fn_exception_returns_none),
        ("gap=0.05 同義 → skip", test_gap_below_ambiguous_skip),
        ("gap=0.22 ambiguous → skip", test_gap_in_ambiguous_zone_skip),
        ("gap=0.60 確定 → record", test_gap_above_threshold_records),
        ("gap=0.30 boundary → record", test_gap_at_record_boundary_records),
        ("gap=0.15 boundary → skip", test_gap_at_ambiguous_boundary_skips),
        ("source label = raw_subj_gap", test_record_source_label),
        ("verdict excerpts 内容検証", test_verdict_excerpts_match_text),
    ]

    results = []
    for label, fn in groups:
        print()
        ok = fn()
        results.append((label, ok))

    print()
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"  {passed}/{total} groups passed")

    sys.exit(0 if passed == total else 1)
