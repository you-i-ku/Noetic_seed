"""OUTWARD step 0e: 一致境界と更新前の軸別履歴を通常の assert で検証。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.predictor import _is_match, update_predictor_confidence


@pytest.mark.parametrize("error, expected", [(0, True), (5, False), (1e-12, False)])
def test_zero_median(error, expected):
    assert _is_match(error, [0] * 5) is expected


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("error, expected", [(3, True), (10, True), (11, False)])
def test_nonzero_median(error, expected, sign):
    assert _is_match(sign * error, [20, 0, 10, 30, 5]) is expected


@pytest.mark.parametrize("length", range(5))
@pytest.mark.parametrize("error", [0, 100, -100])
def test_bootstrap(length, error):
    assert _is_match(error, [0] * length) is True


def test_five_entries_end_bootstrap():
    assert _is_match(100, [0] * 5) is False


def test_even_history_uses_upper_median_without_mutation():
    history = [30, 0, 20, 1, 10, 2]  # 上側10、下側2、平均6
    assert _is_match(10, history) is True
    assert _is_match(11, history) is False
    assert history == [30, 0, 20, 1, 10, 2]


@pytest.mark.parametrize("e2_error, ec_error", [(0, 0), (50, 0.5)])
def test_twenty_constant_errors_keep_increasing_confidence(e2_error, ec_error):
    # 0だけ特例にする代案も、非ゼロ一定誤差で落とす。
    state = {}
    previous = {"e2_conf": 0.7, "ec_conf": 0.7}
    for _ in range(20):
        update_predictor_confidence(state, "tool_A", e2_error, ec_error)
        entry = state["predictor_confidence"]["tool_A"]
        for key in previous:
            assert entry[key] > previous[key]
            assert entry[key] == pytest.approx(previous[key] + 0.05 * (1 - previous[key]))
        previous = entry.copy()
    assert state["prediction_error_history_e2"] == [e2_error] * 20
    assert state["prediction_error_history_ec"] == [ec_error] * 20


@pytest.mark.parametrize("axis, scale", [("e2", 1), ("ec", 0.01)])
@pytest.mark.parametrize("history, error, matches", [
    ([0] * 4, 11, True),  # 追加を先にすると5件となり、不一致になる
    ([0] * 5, 11, False),
    ([0, 1, 2, 10, 20], 8, False),  # 追加を先にすると中央値8で一致する
    ([0] * 5, 0, True),
    ([0] * 5, 1e-10, False),  # ECでは1e-12。許容誤差を導入しない
    ([0, 5, 10, 20, 30], -10, True),
    ([0, 5, 10, 20, 30], -11, False),
    ([0, 5, 10, 20, 30], -3, True),
    ([0, 1, 2, 10, 20, 30], 10, True),
])
def test_axis_update_uses_previous_history(axis, scale, history, error, matches):
    key = f"prediction_error_history_{axis}"
    before = [value * scale for value in history]
    state = {key: before.copy()}
    e2_error = error * scale if axis == "e2" else 0
    ec_error = error * scale if axis == "ec" else 0
    update_predictor_confidence(state, "tool_A", e2_error, ec_error)
    entry = state["predictor_confidence"]["tool_A"]
    assert entry[f"{axis}_conf"] == pytest.approx(0.715 if matches else 0.55)
    assert state[key] == before + [abs(error * scale)]
    other_axis = "ec" if axis == "e2" else "e2"
    assert entry[f"{other_axis}_conf"] == pytest.approx(0.715)
    assert state[f"prediction_error_history_{other_axis}"] == [0]


def test_history_is_shared_between_tools_but_separate_between_axes():
    state = {}
    for _ in range(5):
        update_predictor_confidence(state, "tool_A", 0, 0.5)
    first_entry = state["predictor_confidence"]["tool_A"].copy()
    # 新toolも既存履歴を使う。E2は不一致、ECは一致 (軸混同でも落ちる)。
    update_predictor_confidence(state, "tool_B", 1, 0.25)
    assert state["predictor_confidence"]["tool_B"] == pytest.approx(
        {"e2_conf": 0.55, "ec_conf": 0.715}
    )
    assert state["predictor_confidence"]["tool_A"] == first_entry
    assert state["prediction_error_history_e2"] == [0] * 5 + [1]
    assert state["prediction_error_history_ec"] == [0.5] * 5 + [0.25]


def test_history_keeps_latest_hundred_absolute_errors():
    state = {}
    for i in range(105):
        update_predictor_confidence(state, "tool_A", -i, -i / 200)
    assert state["prediction_error_history_e2"] == list(range(5, 105))
    assert state["prediction_error_history_ec"] == [i / 200 for i in range(5, 105)]
