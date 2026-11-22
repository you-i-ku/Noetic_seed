"""F-001 + F-002 production wiring テスト (v0.5 Phase 5 Slice 4).

Phase 4 (commit 14ed392) で signature を拡張した update_link_strength_used /
should_explore_new_links を、Slice 4 で production path
(core.memory.get_relevant_memories) に配線した部分の verify。

設計確定事項 (Codex Slice 4 review 2026-05-10):
  F-001 timing = **前 cycle modulation** (cycle 構造上 retrieval は cycle 序盤、
  prediction_error history 書込は cycle 末 = main.py:1080 のため、retrieval 時の
  state['prediction_error_history_ec'][-1] は前 cycle 末尾値)。RL/Active Inference
  の TD error / FEP 学習則 (= 過去誤差で現在 policy 更新) と整合的、cycle 1
  bootstrap も history=[] → None graceful で自然。
  F-002 self-inclusion 回避 = should_explore_new_links に渡す state は
  history tail (= current_pe) を除外した shallow view (memory.py の caller 側で
  整形済、percentile threshold 計算で current_pe を含めない)。

成功条件 (F-001 wiring):
  - history=[0.3] → prediction_error=0.3 (= 前 cycle の error) が渡る
  - history=[0.95] → prediction_error=0.95 が渡る
  - history=[] → prediction_error=None (Phase 4 既存 graceful 経路 fallback)
  - history=[..., tail] → prediction_error=tail (前 cycle 末尾値が source)

成功条件 (F-002 wiring):
  - current_pe=None (history 空) → trigger 評価 skip、co_activation 生成 0
  - should_explore_new_links=False → co_activation 生成 0 (誤差小 cycle で爆発抑制)
  - should_explore_new_links=True → co_activation 生成 >= 1

識別力 (CLAUDE.md §5):
  - F-001 = prediction_error 引数を渡さない誤実装で test_f001_low / _high が fail
  - F-002 = caller 配線漏れの誤実装で test_f002_invokes_when_high が fail
  - 注意: 「同 cycle vs 前 cycle」の write/read timing bug は本 unit test では
    検出不能 (mock で history 直接 seed)。production の同 cycle 化 regression は
    integration smoke (BP-2) で検出する設計。

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_memory_link_pc_wiring.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.memory as memory_mod
import core.memory_links as ml_mod
import core.tag_registry as tr


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


# ----- module attribute snapshot for restore -----

_ORIG = {}


def _snapshot_originals():
    if _ORIG:
        return
    _ORIG["externals"] = memory_mod._recent_externals_from_archive
    _ORIG["network_search"] = memory_mod.memory_network_search
    _ORIG["follow_links"] = ml_mod.follow_links
    _ORIG["update_strength"] = ml_mod.update_link_strength_used
    _ORIG["explore"] = ml_mod.should_explore_new_links
    _ORIG["gen_coact"] = ml_mod.generate_co_activation_links


def _patch_for_wiring(captured: dict, *, explore_returns: bool = False):
    """F-001 spy + F-002 hook 制御 patch。memory.py:get_relevant_memories
    の link path / co_activation hook を deterministic に走らせる setup。"""
    _snapshot_originals()
    memory_mod._recent_externals_from_archive = lambda limit=3, days_back=7: []
    # network_mems 2 件 (F-002 hook の len >= 2 条件 + F-001 link path の入口)
    memory_mod.memory_network_search = (
        lambda query, networks=None, limit=8: [
            {"id": "m1", "content": "x", "network": "default"},
            {"id": "m2", "content": "y", "network": "default"},
        ]
    )
    ml_mod.follow_links = lambda node_id, **kwargs: [{
        "memory_entry": {"id": "m3", "content": "z", "network": "default"},
        "via_link": {"id": "link_x"},
        "depth": 1,
        "strength_hint": 0.5,
    }]

    def spy_update(link_id, current_cycle=None, prediction_error=None):
        captured["update_call"] = {
            "link_id": link_id,
            "current_cycle": current_cycle,
            "prediction_error": prediction_error,
        }
        return None
    ml_mod.update_link_strength_used = spy_update

    def spy_gen_coact(memories, current_cycle=None, **kwargs):
        captured["coact_call"] = {
            "memory_count": len(memories or []),
            "current_cycle": current_cycle,
        }
        return []
    ml_mod.generate_co_activation_links = spy_gen_coact

    ml_mod.should_explore_new_links = (
        lambda state, current_error: bool(explore_returns)
    )


def _restore():
    if not _ORIG:
        return
    memory_mod._recent_externals_from_archive = _ORIG["externals"]
    memory_mod.memory_network_search = _ORIG["network_search"]
    ml_mod.follow_links = _ORIG["follow_links"]
    ml_mod.update_link_strength_used = _ORIG["update_strength"]
    ml_mod.should_explore_new_links = _ORIG["explore"]
    ml_mod.generate_co_activation_links = _ORIG["gen_coact"]


def _make_state(history: list) -> dict:
    return {
        "cycle_id": 50,
        "prediction_error_history_ec": list(history),
        "subjective_entries": [{"intent": "test query"}],
    }


# ============================================================
# Section A: F-001 production wiring (prediction_error propagation)
# ============================================================

def test_f001_low_error(tmp_path: Path):
    print("== F-001 history=[0.3] (前 cycle 末尾) → prediction_error=0.3 渡る ==")
    _setup(tmp_path)
    captured = {}
    _patch_for_wiring(captured, explore_returns=False)
    try:
        memory_mod.get_relevant_memories(_make_state([0.3]), use_links=True)
    finally:
        _restore()
    pe = captured.get("update_call", {}).get("prediction_error")
    return _assert(pe == 0.3,
                   f"F-001 prediction_error=0.3 (got: {pe})")


def test_f001_high_error(tmp_path: Path):
    print("== F-001 history=[0.95] (前 cycle 末尾 high) → prediction_error=0.95 渡る ==")
    _setup(tmp_path)
    captured = {}
    _patch_for_wiring(captured, explore_returns=False)
    try:
        memory_mod.get_relevant_memories(_make_state([0.95]), use_links=True)
    finally:
        _restore()
    pe = captured.get("update_call", {}).get("prediction_error")
    return _assert(pe == 0.95,
                   f"F-001 prediction_error=0.95 (got: {pe})")


def test_f001_history_empty_none_fallback(tmp_path: Path):
    print("== F-001 history=[] → prediction_error=None (modulator なし fallback) ==")
    _setup(tmp_path)
    captured = {}
    _patch_for_wiring(captured, explore_returns=False)
    try:
        memory_mod.get_relevant_memories(_make_state([]), use_links=True)
    finally:
        _restore()
    pe = captured.get("update_call", {}).get("prediction_error")
    return _assert(pe is None,
                   f"F-001 prediction_error=None (got: {pe!r})")


def test_f001_takes_previous_cycle_tail(tmp_path: Path):
    """前 cycle modulation: history=[..., tail] → prediction_error=tail。
    cycle N の retrieval は cycle N-1 末尾値 (= 前 cycle の error) で modulate。
    RL/Active Inference の TD error / FEP 学習則 (過去誤差 → 現在 policy 更新)
    の標準パターンと整合的 (Slice 4 設計確定、Codex P1-1 review 整合)。"""
    print("== F-001 history=[0.1, 0.2, 0.7] → 前 cycle 末尾 0.7 が modulator ==")
    _setup(tmp_path)
    captured = {}
    _patch_for_wiring(captured, explore_returns=False)
    try:
        memory_mod.get_relevant_memories(_make_state([0.1, 0.2, 0.7]),
                                         use_links=True)
    finally:
        _restore()
    pe = captured.get("update_call", {}).get("prediction_error")
    return _assert(pe == 0.7,
                   f"F-001 前 cycle 末尾値 0.7 (got: {pe})")


# ============================================================
# Section B: F-002 caller wiring (should_explore_new_links → generate)
# ============================================================

def test_f002_skip_when_pe_none(tmp_path: Path):
    print("== F-002 history=[] (pe=None) → co_activation 生成呼ばれない ==")
    _setup(tmp_path)
    captured = {}
    _patch_for_wiring(captured, explore_returns=True)   # 仮に True でも
    try:
        memory_mod.get_relevant_memories(_make_state([]), use_links=True)
    finally:
        _restore()
    return _assert("coact_call" not in captured,
                   f"co_activation 呼び出しなし (got: {captured.get('coact_call')})")


def test_f002_skip_when_explore_false(tmp_path: Path):
    print("== F-002 explore=False (誤差小 cycle) → 生成呼ばれない ==")
    _setup(tmp_path)
    captured = {}
    _patch_for_wiring(captured, explore_returns=False)
    try:
        memory_mod.get_relevant_memories(_make_state([0.3]), use_links=True)
    finally:
        _restore()
    return _assert("coact_call" not in captured,
                   "explore=False で skip")


def test_f002_invokes_when_explore_true(tmp_path: Path):
    print("== F-002 explore=True (high EC error) → 生成呼ばれる ==")
    _setup(tmp_path)
    captured = {}
    _patch_for_wiring(captured, explore_returns=True)
    try:
        memory_mod.get_relevant_memories(_make_state([0.95]), use_links=True)
    finally:
        _restore()
    call = captured.get("coact_call")
    return all([
        _assert(call is not None, "co_activation 呼ばれる"),
        _assert(call and call.get("memory_count") == 2,
                f"network_mems 2 件渡る (got: {call})"),
        _assert(call and call.get("current_cycle") == 50,
                f"current_cycle=50 渡る (got: {call})"),
    ])


def test_f002_excludes_tail_from_explore_history(tmp_path: Path):
    """P1-2 fix: should_explore_new_links に渡す state['prediction_error_history_ec']
    は tail (= current_pe) を除外した shallow view (Codex Slice 4 review P1-2)。
    tail を含めて渡すと percentile threshold 計算で self-include されて high-error
    trigger が抑制される副作用を回避するための caller 側整形。"""
    print("== F-002 should_explore_new_links は tail 除外 history で評価 ==")
    _setup(tmp_path)
    captured = {}
    _patch_for_wiring(captured, explore_returns=True)

    def spy_explore(state, current_error):
        captured["explore_call"] = {
            "history": list(state.get("prediction_error_history_ec", [])),
            "current_error": current_error,
        }
        return True
    ml_mod.should_explore_new_links = spy_explore

    full_history = [0.1, 0.2, 0.3, 0.4, 0.95]   # tail = 0.95
    try:
        memory_mod.get_relevant_memories(_make_state(full_history),
                                         use_links=True)
    finally:
        _restore()
    call = captured.get("explore_call")
    if not call:
        return _assert(False, "should_explore_new_links 呼ばれてない")
    return all([
        _assert(call["current_error"] == 0.95,
                f"current_error = full history tail 0.95 (got: {call['current_error']})"),
        _assert(call["history"] == [0.1, 0.2, 0.3, 0.4],
                f"渡された history は tail 除外済 (got: {call['history']})"),
        _assert(0.95 not in call["history"],
                f"history に tail 0.95 含まれない (got: {call['history']})"),
    ])


def test_f002_use_links_false_skips_hook(tmp_path: Path):
    print("== F-002 use_links=False → link path 全 skip (hook も呼ばれない) ==")
    _setup(tmp_path)
    captured = {}
    _patch_for_wiring(captured, explore_returns=True)
    try:
        memory_mod.get_relevant_memories(_make_state([0.95]), use_links=False)
    finally:
        _restore()
    return all([
        _assert("update_call" not in captured, "F-001 update も呼ばれない"),
        _assert("coact_call" not in captured, "F-002 hook も呼ばれない"),
    ])


def run_all():
    print("=" * 60)
    print("test_memory_link_pc_wiring.py (v0.5 Phase 5 Slice 4 F-001 + F-002)")
    print("=" * 60)
    results = []
    with tempfile.TemporaryDirectory() as td:
        for fn in [
            test_f001_low_error,
            test_f001_high_error,
            test_f001_history_empty_none_fallback,
            test_f001_takes_previous_cycle_tail,
            test_f002_skip_when_pe_none,
            test_f002_skip_when_explore_false,
            test_f002_invokes_when_explore_true,
            test_f002_excludes_tail_from_explore_history,
            test_f002_use_links_false_skips_hook,
        ]:
            sub = Path(td) / fn.__name__
            sub.mkdir(exist_ok=True)
            results.append(fn(sub))
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"結果: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
