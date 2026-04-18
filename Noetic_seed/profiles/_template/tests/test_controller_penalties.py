"""controller.py の段階5 ペナルティ 2 関数の unit test。

成功条件 (STAGE5_IMPLEMENTATION_PLAN.md §7-2):
  - channel_mismatch: 不一致で score × multiplier、penalties に記録
  - channel_mismatch: internal tool (channel=None) は影響なし (1.0)
  - channel_mismatch: pending に channel がなければ 1.0
  - predicted_outcome: error / no_response で減点、penalties 記録
  - predicted_outcome: other / positive_reply で減点なし
  - WORLD_MODEL_CFG の値が乗算 multiplier として反映される
  - controller_select 実行時に candidate に predicted_outcome が付く (E2E)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_controller_penalties.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.controller import (
    _channel_mismatch_multiplier,
    _predicted_outcome_multiplier,
    controller_select,
)
from core.config import WORLD_MODEL_CFG
from core.world_model import init_world_model, ensure_channel
from core.channel_registry import channel_from_device_input


def _wm_with_test_channels():
    """(v3) テスト用 WM: 起動直後 channels={} なので、device + x + claude を ensure。
    bootstrap 撤去により、段階5 penalty テストは明示登録で channel_mismatch を発火させる。
    """
    wm = init_world_model()
    ensure_channel(wm, **channel_from_device_input())
    # x channel は channel_registry にない (将来 skills/computer use 吸収予定)。
    # テスト目的で手動 spec 登録。
    ensure_channel(wm, id="x", type="social",
                   tools_in=["x_timeline", "x_search", "x_get_notifications"],
                   tools_out=["x_post", "x_reply", "x_quote", "x_like"])
    return wm


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# channel_mismatch
# ============================================================

def test_channel_mismatch_penalty_reduces_score():
    print("== channel_mismatch: 不一致で multiplier 適用 + penalties 記録 ==")
    wm = _wm_with_test_channels()  # (v3) device + x channels を ensure 済
    state = {
        "world_model": wm,
        "pending": [{
            "type": "pending",
            "observed_channel": "device",
            "observed_content": None,
        }],
    }
    cand = {"tool": "x_post"}  # x channel
    mult = _channel_mismatch_multiplier(cand, state, WORLD_MODEL_CFG)
    return all([
        _assert(abs(mult - 0.5) < 1e-9, f"multiplier=0.5 (actual: {mult})"),
        _assert("penalties" in cand, "penalties field 追加"),
        _assert(any("channel_mismatch" in p for p in cand.get("penalties", [])),
                "channel_mismatch 理由記録"),
    ])


def test_channel_mismatch_match_no_penalty():
    print("== channel_mismatch: 一致で 1.0 (penalty なし) ==")
    wm = _wm_with_test_channels()  # (v3) device channel を ensure 済
    state = {
        "world_model": wm,
        "pending": [{
            "type": "pending",
            "observed_channel": "device",
            "observed_content": None,
        }],
    }
    cand = {"tool": "output_display"}  # device channel
    mult = _channel_mismatch_multiplier(cand, state, WORLD_MODEL_CFG)
    return all([
        _assert(mult == 1.0, f"multiplier=1.0 (actual: {mult})"),
        _assert("penalties" not in cand or not cand["penalties"],
                "penalties 記録なし"),
    ])


def test_channel_mismatch_skipped_for_internal_tool():
    print("== channel_mismatch: internal tool (channel=None) は影響なし ==")
    wm = init_world_model()
    state = {
        "world_model": wm,
        "pending": [{
            "type": "pending",
            "observed_channel": "device",
            "observed_content": None,
        }],
    }
    # read_file は channel 未登録 (internal)
    cand = {"tool": "read_file"}
    mult = _channel_mismatch_multiplier(cand, state, WORLD_MODEL_CFG)
    return _assert(mult == 1.0, f"multiplier=1.0 (actual: {mult})")


def test_channel_mismatch_no_pending_channels():
    print("== channel_mismatch: pending に channel なしで 1.0 ==")
    wm = init_world_model()
    state = {"world_model": wm, "pending": []}
    cand = {"tool": "x_post"}
    mult = _channel_mismatch_multiplier(cand, state, WORLD_MODEL_CFG)
    return _assert(mult == 1.0, f"multiplier=1.0 (actual: {mult})")


def test_channel_mismatch_observed_skipped():
    print("== channel_mismatch: observed 済 pending は無視 ==")
    wm = init_world_model()
    state = {
        "world_model": wm,
        "pending": [{
            "type": "pending",
            "observed_channel": "device",
            "observed_content": "already observed",  # 消化済
        }],
    }
    cand = {"tool": "x_post"}
    mult = _channel_mismatch_multiplier(cand, state, WORLD_MODEL_CFG)
    return _assert(mult == 1.0, f"消化済 pending → 1.0 (actual: {mult})")


# ============================================================
# predicted_outcome
# ============================================================

def test_predicted_outcome_error_penalizes():
    print("== predicted_outcome: category=error で減点 + penalties 記録 ==")
    cand = {"tool": "x_post"}
    pred = {"category": "error", "confidence": 0.6, "detail": "light"}
    mult = _predicted_outcome_multiplier(pred, cand, WORLD_MODEL_CFG)
    return all([
        _assert(abs(mult - 0.5) < 1e-9, f"multiplier=0.5 (actual: {mult})"),
        _assert("predicted_error" in cand.get("penalties", []),
                "predicted_error 記録"),
    ])


def test_predicted_outcome_no_response_penalizes():
    print("== predicted_outcome: category=no_response で減点 ==")
    cand = {"tool": "x_post"}
    pred = {"category": "no_response", "confidence": 0.5}
    mult = _predicted_outcome_multiplier(pred, cand, WORLD_MODEL_CFG)
    return all([
        _assert(abs(mult - 0.5) < 1e-9, f"multiplier=0.5 (actual: {mult})"),
        _assert("predicted_no_response" in cand.get("penalties", []),
                "predicted_no_response 記録"),
    ])


def test_predicted_outcome_other_no_penalty():
    print("== predicted_outcome: category=other / positive で減点なし ==")
    cand1 = {"tool": "x_post"}
    cand2 = {"tool": "x_post"}
    m1 = _predicted_outcome_multiplier(
        {"category": "other"}, cand1, WORLD_MODEL_CFG)
    m2 = _predicted_outcome_multiplier(
        {"category": "positive_reply"}, cand2, WORLD_MODEL_CFG)
    return all([
        _assert(m1 == 1.0, f"other → 1.0 (actual: {m1})"),
        _assert(m2 == 1.0, f"positive_reply → 1.0 (actual: {m2})"),
        _assert("penalties" not in cand1 or not cand1["penalties"],
                "other で penalty 記録なし"),
    ])


# ============================================================
# 設定値の反映
# ============================================================

def test_multipliers_read_from_cfg():
    print("== 設定値: cfg override で multiplier 変動 ==")
    wm = _wm_with_test_channels()  # (v3) device + x channels を ensure 済
    state = {
        "world_model": wm,
        "pending": [{"type": "pending", "observed_channel": "device",
                     "observed_content": None}],
    }
    cand = {"tool": "x_post"}
    custom_cfg = {"channel_mismatch_multiplier": 0.2,
                  "predicted_error_multiplier": 0.1}
    m1 = _channel_mismatch_multiplier(cand, state, custom_cfg)

    cand2 = {"tool": "x"}
    m2 = _predicted_outcome_multiplier(
        {"category": "error"}, cand2, custom_cfg)
    return all([
        _assert(abs(m1 - 0.2) < 1e-9, f"channel_mismatch=0.2 (actual: {m1})"),
        _assert(abs(m2 - 0.1) < 1e-9, f"predicted_error=0.1 (actual: {m2})"),
    ])


# ============================================================
# controller_select E2E: candidate に predicted_outcome が付くこと
# ============================================================

def test_controller_select_attaches_predicted_outcome():
    print("== controller_select E2E: candidate に predicted_outcome 付与 ==")
    wm = init_world_model()
    state = {
        "energy": 50,
        "entropy": 0.65,
        "log": [],
        "pending": [],
        "world_model": wm,
    }
    ctrl = {"tool_rank": {"output_display": 60, "wait": 40}}
    candidates = [
        {"tool": "output_display", "reason": "応答する",
         "expected": "応答が返る"},
        {"tool": "wait", "reason": "待機", "expected": "変化なし"},
    ]
    chosen = controller_select(candidates, ctrl, state)
    return all([
        _assert(chosen in candidates, "候補からひとつ選ばれる"),
        _assert("predicted_outcome" in candidates[0],
                "output_display に predicted_outcome 付与"),
        _assert("predicted_outcome" in candidates[1],
                "wait に predicted_outcome 付与"),
        _assert(candidates[0]["predicted_outcome"].get("category")
                == "positive_reply",
                f"output_display/応答 → positive_reply "
                f"(actual: {candidates[0]['predicted_outcome'].get('category')})"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("channel_mismatch: 不一致で減点", test_channel_mismatch_penalty_reduces_score),
        ("channel_mismatch: 一致で 1.0", test_channel_mismatch_match_no_penalty),
        ("channel_mismatch: internal tool skip", test_channel_mismatch_skipped_for_internal_tool),
        ("channel_mismatch: pending なし 1.0", test_channel_mismatch_no_pending_channels),
        ("channel_mismatch: observed 済 skip", test_channel_mismatch_observed_skipped),
        ("predicted_outcome: error 減点", test_predicted_outcome_error_penalizes),
        ("predicted_outcome: no_response 減点", test_predicted_outcome_no_response_penalizes),
        ("predicted_outcome: other 減点なし", test_predicted_outcome_other_no_penalty),
        ("cfg override 反映", test_multipliers_read_from_cfg),
        ("E2E: controller_select predicted_outcome 付与", test_controller_select_attaches_predicted_outcome),
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
