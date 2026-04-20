"""controller.py の段階5→9 ペナルティ関数の unit test。

成功条件 (STAGE5_IMPLEMENTATION_PLAN.md §7-2 + STAGE9 §4-3):
  - channel_mismatch: 不一致で score × multiplier、penalties に記録 (段階5 仕様維持)
  - channel_mismatch: internal tool (channel=None) は影響なし (1.0)
  - channel_mismatch: pending に channel がなければ 1.0
  - predicted_outcome (段階9 置換): predicted_e2 ベース連続値 multiplier
    · predicted_e2=20 → 0.2 (error 相当) / 30 → 0.3 (no_response 相当)
    · predicted_e2=50 → 0.5 (other 相当) / 70 → 0.7 (positive_reply 相当)
    · ratio < 0.4 で penalties に "low_predicted_e2=XX" 記録
  - WORLD_MODEL_CFG の値が乗算 multiplier として反映される (floor)
  - controller_select 実行時に candidate に predicted_outcome + _predicted_e2 付与

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

def test_predicted_outcome_low_e2_penalizes():
    print("== predicted_outcome (段階9): 低 predicted_e2 で減点 + penalties 記録 ==")
    cand = {"tool": "x_post"}
    pred = {"category": "error", "confidence": 0.6, "detail": "light",
            "predicted_e2": 20}
    # 段階10 柱 C: signature に state 追加、penalty ラベル "low_outcome" に変更
    mult = _predicted_outcome_multiplier(pred, cand, {}, WORLD_MODEL_CFG)
    return all([
        _assert(abs(mult - 0.2) < 1e-9, f"multiplier=0.2 (actual: {mult})"),
        _assert(any("low_outcome" in p for p in cand.get("penalties", [])),
                "low_outcome 記録 (段階10 柱 C で low_predicted_e2 から改名)"),
    ])


def test_predicted_outcome_mid_e2_moderate_penalty():
    print("== predicted_outcome (段階9): 中 predicted_e2=30 → 0.3 + penalty ==")
    cand = {"tool": "x_post"}
    pred = {"category": "no_response", "confidence": 0.5, "predicted_e2": 30}
    mult = _predicted_outcome_multiplier(pred, cand, {}, WORLD_MODEL_CFG)
    return all([
        _assert(abs(mult - 0.3) < 1e-9, f"multiplier=0.3 (actual: {mult})"),
        _assert(any("low_outcome" in p for p in cand.get("penalties", [])),
                "low_outcome 記録 (< 0.4)"),
    ])


def test_predicted_outcome_continuous_scaling():
    print("== predicted_outcome (段階9): pe2=50 → 0.5 / pe2=70 → 0.7、連続値 ==")
    cand1 = {"tool": "x_post"}
    cand2 = {"tool": "x_post"}
    cand3 = {"tool": "x_post"}
    m1 = _predicted_outcome_multiplier(
        {"category": "other", "predicted_e2": 50}, cand1, {}, WORLD_MODEL_CFG)
    m2 = _predicted_outcome_multiplier(
        {"category": "positive_reply", "predicted_e2": 70}, cand2, {}, WORLD_MODEL_CFG)
    m3 = _predicted_outcome_multiplier(
        {"category": "positive_reply", "predicted_e2": 100}, cand3, {}, WORLD_MODEL_CFG)
    return all([
        _assert(abs(m1 - 0.5) < 1e-9, f"pe2=50 → 0.5 (actual: {m1})"),
        _assert(abs(m2 - 0.7) < 1e-9, f"pe2=70 → 0.7 (actual: {m2})"),
        _assert(abs(m3 - 1.0) < 1e-9, f"pe2=100 → 1.0 (actual: {m3})"),
        _assert("penalties" not in cand2 or not cand2["penalties"],
                "pe2=70 (>=0.4) で penalty 記録なし"),
    ])


def test_predicted_outcome_zero_floor():
    print("== predicted_outcome (段階9): pe2=0 → floor 0.05 ==")
    cand = {"tool": "x_post"}
    pred = {"predicted_e2": 0}
    mult = _predicted_outcome_multiplier(pred, cand, {}, WORLD_MODEL_CFG)
    return all([
        _assert(abs(mult - 0.05) < 1e-9, f"pe2=0 → floor 0.05 (actual: {mult})"),
        _assert(any("low_outcome" in p for p in cand.get("penalties", [])),
                "low_outcome 記録 (pe2=0 相当)"),
    ])


def test_predicted_outcome_missing_pe2_defaults_to_50():
    print("== predicted_outcome (段階9): predicted_e2 欠損 → default 50 = 0.5 ==")
    cand = {"tool": "x_post"}
    pred = {"category": "error"}  # 段階5 互換、predicted_e2 なし
    mult = _predicted_outcome_multiplier(pred, cand, {}, WORLD_MODEL_CFG)
    return _assert(abs(mult - 0.5) < 1e-9,
                   f"欠損時 default 50 → 0.5 (actual: {mult})")


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
    # 段階9: predicted_error_multiplier は廃止、predicted_e2_floor に置換
    custom_cfg = {"channel_mismatch_multiplier": 0.2,
                  "predicted_e2_floor": 0.1}
    m1 = _channel_mismatch_multiplier(cand, state, custom_cfg)

    cand2 = {"tool": "x"}
    # pe2=0 は通常 floor 0.05 だが、custom cfg で floor=0.1 に引き上げられる
    m2 = _predicted_outcome_multiplier(
        {"predicted_e2": 0}, cand2, {}, custom_cfg)
    return all([
        _assert(abs(m1 - 0.2) < 1e-9, f"channel_mismatch=0.2 (actual: {m1})"),
        _assert(abs(m2 - 0.1) < 1e-9, f"predicted_e2_floor=0.1 (actual: {m2})"),
    ])


# ============================================================
# controller_select E2E: candidate に predicted_outcome が付くこと
# ============================================================

def test_controller_select_attaches_predicted_outcome():
    print("== controller_select E2E: candidate に predicted_outcome + _predicted_e2 付与 ==")
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
        # 段階9: _predicted_e2 が candidate に記録される (main.py が log entry に転写)
        _assert("_predicted_e2" in candidates[0],
                "output_display に _predicted_e2 付与 (段階9)"),
        _assert(candidates[0]["_predicted_e2"] == 70,
                f"Light → positive_reply → 70 (actual: {candidates[0]['_predicted_e2']})"),
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
        ("predicted_outcome (9): pe2=20 減点", test_predicted_outcome_low_e2_penalizes),
        ("predicted_outcome (9): pe2=30 減点", test_predicted_outcome_mid_e2_moderate_penalty),
        ("predicted_outcome (9): 連続値スケール", test_predicted_outcome_continuous_scaling),
        ("predicted_outcome (9): pe2=0 → floor", test_predicted_outcome_zero_floor),
        ("predicted_outcome (9): pe2 欠損 → default 50", test_predicted_outcome_missing_pe2_defaults_to_50),
        ("cfg override 反映", test_multipliers_read_from_cfg),
        ("E2E: controller_select predicted_outcome + _predicted_e2", test_controller_select_attaches_predicted_outcome),
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
