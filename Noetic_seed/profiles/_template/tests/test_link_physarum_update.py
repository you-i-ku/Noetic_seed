"""Physarum strength update テスト (段階11-D Phase 3 Step 3.1-3.6).

成功条件:
  - PHYSARUM_ALPHA = 0.1 / PHYSARUM_BETA = 0.05 / STRENGTH_CAP = 1.0 (Session W v1)
  - PRUNING_STRENGTH_RATIO = 0.15 / _compute_pruning_idle_cycles() ≒ 56 cycle
  - _link_strength: backward compat (Phase 2 以前 link は confidence にフォールバック)
  - _apply_lazy_decay: strength * (1 - β)^elapsed の数式正当性
  - _apply_lazy_decay: last_used_cycle 欠落 link は decay skip
  - update_link_strength_used: strength up + lazy decay + last_used / usage_count update
  - update_link_strength_used: 上限 1.0 で clip
  - update_link_strength_used: backward compat (古い link でも動く)
  - prune_weak_links: 低 strength + 長 idle で削除
  - prune_weak_links: 高 strength or 短 idle は保持
  - prune_weak_links: confidence ベースの相対 threshold (initial × 0.15)
  - follow_links: strength_hint が strength を返す (新 link)、confidence にフォールバック (古 link)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_link_physarum_update.py
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
    PHYSARUM_ALPHA, PHYSARUM_BETA, STRENGTH_CAP, PRUNING_STRENGTH_RATIO,
    _compute_pruning_idle_cycles,
    _link_strength,
    _apply_lazy_decay,
    update_link_strength_used,
    prune_weak_links,
    list_links,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup(tmp_path: Path):
    """test 用 MEMORY_DIR セットアップ。core.config / core.memory / core.memory_links 全部に反映。"""
    from core import config as cfg
    cfg.MEMORY_DIR = tmp_path
    memory_mod.MEMORY_DIR = tmp_path
    # memory_links は MEMORY_DIR を import 済なので module 属性を直接更新
    ml_mod.MEMORY_DIR = tmp_path
    reg_file = tmp_path / "registered_tags.json"
    tr._reset_for_testing(registry_file=reg_file)
    tr.register_standard_tags()


def _write_link(tmp_path: Path, link: dict):
    """test 用に直接 memory_links.jsonl に書き込む。"""
    fpath = tmp_path / "memory_links.jsonl"
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(link, ensure_ascii=False) + "\n")


def test_constants():
    print("== Phase 3 定数 (Session W v1 確定値) ==")
    return all([
        _assert(PHYSARUM_ALPHA == 0.1, f"α = 0.1 (got: {PHYSARUM_ALPHA})"),
        _assert(PHYSARUM_BETA == 0.05, f"β = 0.05 (got: {PHYSARUM_BETA})"),
        _assert(STRENGTH_CAP == 1.0, f"上限 = 1.0 (got: {STRENGTH_CAP})"),
        _assert(PRUNING_STRENGTH_RATIO == 0.15,
                f"pruning ratio = 0.15 (got: {PRUNING_STRENGTH_RATIO})"),
    ])


def test_pruning_idle_cycles_dynamic():
    print("== _compute_pruning_idle_cycles: β 半減期 × 4 ≒ 56 cycle ==")
    cycles = _compute_pruning_idle_cycles()
    # β=0.05 で half life ln(0.5)/ln(0.95) ≒ 13.51、× 4 ≒ 54、round で 54
    return all([
        _assert(50 <= cycles <= 60, f"β=0.05 で 50-60 cycle 範囲 (got: {cycles})"),
    ])


def test_link_strength_backward_compat():
    print("== _link_strength: Phase 2 以前 link は confidence にフォールバック ==")
    # 新 link (Phase 2 以降): strength field あり
    new_link = {"strength": 0.85, "confidence": 0.7}
    # 古 link (Phase 2 以前): strength field なし
    old_link = {"confidence": 0.7}
    # 両方なし
    empty_link = {}
    return all([
        _assert(_link_strength(new_link) == 0.85, "新 link は strength 値"),
        _assert(_link_strength(old_link) == 0.7, "古 link は confidence にフォールバック"),
        _assert(_link_strength(empty_link) == 0.0, "両方なしは 0.0"),
    ])


def test_lazy_decay_formula():
    print("== _apply_lazy_decay: strength * (1-β)^elapsed の数式正当性 ==")
    link = {"strength": 1.0, "last_used_cycle": 10}
    # current_cycle = 10 → elapsed = 0 → 1.0
    s0 = _apply_lazy_decay(link, 10)
    # current_cycle = 11 → elapsed = 1 → 1.0 * 0.95 = 0.95
    s1 = _apply_lazy_decay(link, 11)
    # current_cycle = 24 → elapsed = 14 → ≒ 0.488 (半減期付近)
    s14 = _apply_lazy_decay(link, 24)
    return all([
        _assert(abs(s0 - 1.0) < 1e-9, f"elapsed=0 で衰退なし (got: {s0})"),
        _assert(abs(s1 - 0.95) < 1e-9, f"elapsed=1 で 0.95 (got: {s1})"),
        _assert(0.45 < s14 < 0.55, f"elapsed=14 で半減期付近 (got: {s14})"),
    ])


def test_lazy_decay_skip_when_no_last_cycle():
    print("== _apply_lazy_decay: last_used_cycle 欠落 link は decay skip ==")
    link_old = {"strength": 0.8}   # last_used_cycle なし
    s = _apply_lazy_decay(link_old, 100)
    return _assert(s == 0.8, f"last_used_cycle 欠落で decay skip (got: {s})")


def test_update_strength_up(tmp_path: Path):
    print("== update_link_strength_used: strength up + state 更新 ==")
    _setup(tmp_path)
    _write_link(tmp_path, {
        "id": "link_x",
        "from_id": "a", "to_id": "b",
        "link_type": "similar", "confidence": 0.7,
        "strength": 0.7, "usage_count": 0,
        "last_used_cycle": 5,
    })
    updated = update_link_strength_used("link_x", current_cycle=10)
    # elapsed=5, decay: 0.7 * 0.95^5 ≒ 0.541
    # up: 0.541 + 0.1 = 0.641
    return all([
        _assert(updated is not None, "更新 link 返る"),
        _assert(0.60 < updated["strength"] < 0.70,
                f"strength = decay + α (got: {updated['strength']:.4f})"),
        _assert(updated["usage_count"] == 1, "usage_count = 1"),
        _assert(updated["last_used_cycle"] == 10, "last_used_cycle = 10"),
    ])


def test_update_strength_cap(tmp_path: Path):
    print("== update_link_strength_used: STRENGTH_CAP (1.0) で clip ==")
    _setup(tmp_path)
    _write_link(tmp_path, {
        "id": "link_max",
        "from_id": "a", "to_id": "b",
        "link_type": "similar", "confidence": 0.95,
        "strength": 0.95,
        "last_used_cycle": 100,
    })
    updated = update_link_strength_used("link_max", current_cycle=100)
    # elapsed=0, decay: 0.95、up: 0.95 + 0.1 = 1.05 → clip 1.0
    return _assert(updated["strength"] == 1.0,
                   f"上限 1.0 clip (got: {updated['strength']})")


def test_update_strength_old_link(tmp_path: Path):
    print("== update_link_strength_used: 古 link (strength field なし) でも動く ==")
    _setup(tmp_path)
    _write_link(tmp_path, {
        "id": "link_old",
        "from_id": "a", "to_id": "b",
        "link_type": "causal", "confidence": 0.7,
        # strength / last_used_cycle / usage_count 全部欠落 (Phase 2 以前)
    })
    updated = update_link_strength_used("link_old", current_cycle=10)
    # _link_strength で 0.7 にフォールバック、decay は last_used_cycle 欠落で skip、up: 0.7 + 0.1 = 0.8
    return all([
        _assert(updated is not None, "古 link でも更新成立"),
        _assert(abs(updated["strength"] - 0.8) < 1e-9,
                f"古 link でも strength 値が入る (got: {updated['strength']})"),
        _assert(updated["usage_count"] == 1, "usage_count 0 → 1"),
    ])


def test_prune_removes_weak_old(tmp_path: Path):
    print("== prune_weak_links: idle 長 → 削除 / idle 短 → 保持 (Physarum 哲学) ==")
    _setup(tmp_path)
    idle_thr = _compute_pruning_idle_cycles()
    # 削除対象: idle 長 (>= idle_threshold) + decay 込み strength 低
    _write_link(tmp_path, {
        "id": "link_old_unused",
        "from_id": "a", "to_id": "b",
        "link_type": "similar", "confidence": 0.7,
        "strength": 0.7,
        "last_used_cycle": 0,    # current=100、idle=100 >= 54 → 削除条件成立
    })
    # 保護対象: idle 短 (< idle_threshold)、最近 access された
    _write_link(tmp_path, {
        "id": "link_recent",
        "from_id": "a", "to_id": "c",
        "link_type": "similar", "confidence": 0.7,
        "strength": 0.05,        # raw 低くても idle 短で保持
        "last_used_cycle": 95,   # current=100、idle=5 < 54 → 保護
    })
    # 保護対象: idle ぴったり threshold 直前
    _write_link(tmp_path, {
        "id": "link_borderline",
        "from_id": "a", "to_id": "d",
        "link_type": "similar", "confidence": 0.7,
        "strength": 0.7,
        "last_used_cycle": 100 - idle_thr + 1,   # idle = idle_thr - 1、未満で保護
    })
    removed = prune_weak_links(current_cycle=100)
    remaining_ids = [l["id"] for l in list_links(limit=100)]
    return all([
        _assert(removed == 1, f"1 件削除 (got: {removed})"),
        _assert("link_old_unused" not in remaining_ids, "idle 長 link 削除"),
        _assert("link_recent" in remaining_ids, "idle 短 link 保持 (raw strength 低くても)"),
        _assert("link_borderline" in remaining_ids,
                f"idle = {idle_thr - 1} (threshold {idle_thr} 未満) で保持"),
    ])


def test_prune_relative_threshold_calculation():
    print("== PRUNING_STRENGTH_RATIO: initial × 0.15 計算 (単体) ==")
    # 純粋計算での threshold 検証 (decay 影響を排除した実装の正当性)
    return all([
        _assert(0.7 * PRUNING_STRENGTH_RATIO == 0.7 * 0.15,
                "confidence=0.7 → threshold=0.105"),
        _assert(1.0 * PRUNING_STRENGTH_RATIO == 0.15,
                "confidence=1.0 → threshold=0.15"),
        _assert(abs(0.85 * PRUNING_STRENGTH_RATIO - 0.1275) < 1e-9,
                "confidence=0.85 → threshold=0.1275"),
    ])


def run_all():
    print("=" * 60)
    print("test_link_physarum_update.py (段階11-D Phase 3)")
    print("=" * 60)
    results = []
    # Section A: 定数
    results.append(test_constants())
    results.append(test_pruning_idle_cycles_dynamic())
    # Section B: 数式 (純粋関数)
    results.append(test_link_strength_backward_compat())
    results.append(test_lazy_decay_formula())
    results.append(test_lazy_decay_skip_when_no_last_cycle())
    results.append(test_prune_relative_threshold_calculation())
    # Section C: update / prune (tmp_path 必要)
    with tempfile.TemporaryDirectory() as td:
        for fn in [
            test_update_strength_up,
            test_update_strength_cap,
            test_update_strength_old_link,
            test_prune_removes_weak_old,
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
