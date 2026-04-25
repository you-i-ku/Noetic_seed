"""Predictive coding × link strength テスト (段階11-D Phase 4 Step 4.1-4.4).

成功条件:
  - update_link_strength_used に prediction_error 引数 (modulator)
  - prediction_error=0.0 (成功) → strength_delta = α
  - prediction_error=1.0 (完全失敗) → strength_delta = 0、strength up ゼロ
  - prediction_error=0.5 (中間) → strength_delta = α * 0.5
  - prediction_error の clamp (0-1 外も安全に処理)
  - prediction_error=None (Phase 3 互換) → modulator なし、α そのまま
  - should_explore_new_links: 動的 90% percentile threshold
  - should_explore_new_links: サンプル不足で initial fallback 0.7
  - should_explore_new_links: 段階10 経路 (state["prediction_error_history_ec"]) 接続
  - should_explore_new_links: state なし / current_error 不正で graceful False

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_predictive_coding_link.py
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
    PHYSARUM_ALPHA,
    update_link_strength_used,
    should_explore_new_links,
    NEW_LINK_EXPLORATION_FALLBACK,
    NEW_LINK_EXPLORATION_MIN_SAMPLES,
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


def _write_link(tmp_path: Path, link: dict):
    fpath = tmp_path / "memory_links.jsonl"
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(link, ensure_ascii=False) + "\n")


def _setup_link_with_strength(tmp_path: Path, initial_strength: float = 0.5,
                               last_used_cycle: int = 100) -> str:
    """test 用 link を作成、id を返す。"""
    _setup(tmp_path)
    link_id = "link_test"
    _write_link(tmp_path, {
        "id": link_id,
        "from_id": "a", "to_id": "b",
        "link_type": "similar", "confidence": 0.7,
        "strength": initial_strength,
        "last_used_cycle": last_used_cycle,
        "usage_count": 0,
    })
    return link_id


# ============================================================
# Section A: Phase 4 modulator (prediction_error modulator)
# ============================================================

def test_modulator_no_error_alpha_full(tmp_path: Path):
    print("== prediction_error=0.0 (成功) → strength_delta = α (満額) ==")
    link_id = _setup_link_with_strength(tmp_path, initial_strength=0.5,
                                          last_used_cycle=100)
    updated = update_link_strength_used(link_id, current_cycle=100,
                                         prediction_error=0.0)
    expected = 0.5 + PHYSARUM_ALPHA   # decay なし (idle=0) + α 満額
    return _assert(abs(updated["strength"] - expected) < 1e-9,
                   f"strength = {expected:.4f} (got: {updated['strength']:.4f})")


def test_modulator_max_error_zero_delta(tmp_path: Path):
    print("== prediction_error=1.0 (完全失敗) → strength_delta = 0 ==")
    link_id = _setup_link_with_strength(tmp_path, initial_strength=0.5,
                                          last_used_cycle=100)
    updated = update_link_strength_used(link_id, current_cycle=100,
                                         prediction_error=1.0)
    expected = 0.5   # decay なし + α × 0 = up なし
    return _assert(abs(updated["strength"] - expected) < 1e-9,
                   f"strength up ゼロ (got: {updated['strength']:.4f})")


def test_modulator_mid_error_half(tmp_path: Path):
    print("== prediction_error=0.5 (中間) → strength_delta = α × 0.5 ==")
    link_id = _setup_link_with_strength(tmp_path, initial_strength=0.5,
                                          last_used_cycle=100)
    updated = update_link_strength_used(link_id, current_cycle=100,
                                         prediction_error=0.5)
    expected = 0.5 + PHYSARUM_ALPHA * 0.5   # 0.5 + 0.05
    return _assert(abs(updated["strength"] - expected) < 1e-9,
                   f"strength = {expected:.4f} (got: {updated['strength']:.4f})")


def test_modulator_clamp_negative(tmp_path: Path):
    print("== prediction_error=-0.5 (不正値) → 0.0 にクランプして満額 up ==")
    link_id = _setup_link_with_strength(tmp_path, initial_strength=0.5,
                                          last_used_cycle=100)
    updated = update_link_strength_used(link_id, current_cycle=100,
                                         prediction_error=-0.5)
    expected = 0.5 + PHYSARUM_ALPHA   # 0 にクランプされて α 満額
    return _assert(abs(updated["strength"] - expected) < 1e-9,
                   f"clamp 後 α 満額 (got: {updated['strength']:.4f})")


def test_modulator_clamp_overshoot(tmp_path: Path):
    print("== prediction_error=2.0 (上限超え) → 1.0 にクランプして up ゼロ ==")
    link_id = _setup_link_with_strength(tmp_path, initial_strength=0.5,
                                          last_used_cycle=100)
    updated = update_link_strength_used(link_id, current_cycle=100,
                                         prediction_error=2.0)
    expected = 0.5   # 1.0 にクランプ → α × 0 = up なし
    return _assert(abs(updated["strength"] - expected) < 1e-9,
                   f"clamp 後 up ゼロ (got: {updated['strength']:.4f})")


def test_phase3_compat_no_prediction_error(tmp_path: Path):
    print("== prediction_error=None (Phase 3 互換) → α 満額、Phase 3 挙動と一致 ==")
    link_id = _setup_link_with_strength(tmp_path, initial_strength=0.5,
                                          last_used_cycle=100)
    updated = update_link_strength_used(link_id, current_cycle=100)   # error 引数省略
    expected = 0.5 + PHYSARUM_ALPHA
    return _assert(abs(updated["strength"] - expected) < 1e-9,
                   f"Phase 3 挙動 = α 満額 (got: {updated['strength']:.4f})")


# ============================================================
# Section B: should_explore_new_links (動的 percentile threshold)
# ============================================================

def test_explore_fallback_insufficient_samples():
    print("== should_explore_new_links: サンプル不足 (< 5) で fallback 0.7 ==")
    state = {"prediction_error_history_ec": [0.1, 0.2]}   # 2 サンプルのみ
    return all([
        _assert(should_explore_new_links(state, 0.8) is True,
                "current 0.8 > fallback 0.7 → True"),
        _assert(should_explore_new_links(state, 0.5) is False,
                "current 0.5 < fallback 0.7 → False"),
    ])


def test_explore_dynamic_percentile():
    print("== should_explore_new_links: 過去 20 cycle 90% percentile 動的 threshold ==")
    # 10 サンプル: 0.0, 0.1, ..., 0.9 (90% percentile = idx 9 = 0.9)
    state = {"prediction_error_history_ec": [i * 0.1 for i in range(10)]}
    return all([
        _assert(should_explore_new_links(state, 0.95) is True,
                "current 0.95 > 0.9 (90% percentile) → True"),
        # 0.9 ぴったりは > 比較で False
        _assert(should_explore_new_links(state, 0.9) is False,
                "current 0.9 = threshold → False (>比較)"),
        _assert(should_explore_new_links(state, 0.5) is False,
                "current 0.5 < threshold → False"),
    ])


def test_explore_recent_window_only():
    print("== should_explore_new_links: 過去 N=20 cycle のみ参照 (古いサンプル無視) ==")
    # 古い 30 サンプル (高 error)、最新 20 サンプル (低 error)
    history = [0.95] * 30 + [0.1] * 20
    state = {"prediction_error_history_ec": history}
    # 最新 20 のみ参照 = 全部 0.1、90% percentile = 0.1
    return _assert(should_explore_new_links(state, 0.5) is True,
                   "古いサンプル無視、最新 20 の 90% percentile が低い")


def test_explore_graceful_no_state():
    print("== should_explore_new_links: state なし / 不正値で graceful ==")
    return all([
        _assert(should_explore_new_links({}, 0.8) is True,
                "空 state → fallback 0.7、0.8 > 0.7 → True"),
        _assert(should_explore_new_links({}, 0.3) is False,
                "空 state、0.3 < 0.7 → False"),
        # current_error が float でない時
        _assert(should_explore_new_links({}, "invalid") is False,
                "current 不正値 → False"),
        # state が dict でない時
        _assert(should_explore_new_links(None, 0.8) is True,
                "state=None でも graceful (fallback path)"),
    ])


def test_explore_min_samples_constant():
    print("== NEW_LINK_EXPLORATION_MIN_SAMPLES = 5 (Session W v1 確定) ==")
    return all([
        _assert(NEW_LINK_EXPLORATION_MIN_SAMPLES == 5,
                f"min samples = 5 (got: {NEW_LINK_EXPLORATION_MIN_SAMPLES})"),
        _assert(NEW_LINK_EXPLORATION_FALLBACK == 0.7,
                f"fallback = 0.7 (got: {NEW_LINK_EXPLORATION_FALLBACK})"),
    ])


def run_all():
    print("=" * 60)
    print("test_predictive_coding_link.py (段階11-D Phase 4)")
    print("=" * 60)
    results = []
    # Section A: modulator (tmp_path 必要)
    with tempfile.TemporaryDirectory() as td:
        for fn in [
            test_modulator_no_error_alpha_full,
            test_modulator_max_error_zero_delta,
            test_modulator_mid_error_half,
            test_modulator_clamp_negative,
            test_modulator_clamp_overshoot,
            test_phase3_compat_no_prediction_error,
        ]:
            sub = Path(td) / fn.__name__
            sub.mkdir(exist_ok=True)
            results.append(fn(sub))
    # Section B: should_explore_new_links (純粋関数)
    results.append(test_explore_fallback_insufficient_samples())
    results.append(test_explore_dynamic_percentile())
    results.append(test_explore_recent_window_only())
    results.append(test_explore_graceful_no_state())
    results.append(test_explore_min_samples_constant())
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"結果: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
