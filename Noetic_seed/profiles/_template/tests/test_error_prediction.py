"""C-2: 先読み、KL逆向き、run限定復元、故障時の計数欠落を検出する。"""
import ast
import builtins
import copy
import json
import math
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import metrics
from test_outward_metrics import _fire_harness, _run, fixed_io  # noqa: F401
from test_tool_invocation_observation import STAGES, PROPOSAL, _build, _record


def invocation(number=0, tool="A", error=False, run="old", **changes):
    row = {"event_type": "tool_invocation", "run_id": run, "attempt_id": "fire",
           "chain_position": 0, "invocation_position": number, "tool_id": "reused",
           "tool": tool, "is_error": error, "entry_id": None, "cycle_id": None, "time": "T"}
    return {**row, **changes}


def lines(directory, kind=None):
    path = directory / metrics.METRICS_FILE_NAME
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return events if kind is None else [e for e in events if e["event_type"] == kind]


@pytest.mark.parametrize("error", [False, True])
def test_initial_and_asymmetric_calculation(error):
    first = metrics.beta_error_prediction(1, 1, error)
    assert first["p_error"] == 0.5 and first["brier"] == 0.25
    assert first["surprise_nats"] == pytest.approx(math.log(2))
    assert first["kl_nats"] == pytest.approx(math.log(2) - 0.5)
    assert (first["posterior_alpha"], first["posterior_beta"]) == ((2, 1) if error else (1, 2))
    # 整数事前の独立な手計算: psi(n)=H_(n-1)-gamma。
    row = metrics.beta_error_prediction(3, 9, error)
    observed = 3 if error else 9
    expected = math.log(12 / observed) - sum(1 / k for k in range(observed + 1, 13))
    assert row["kl_nats"] == pytest.approx(expected)
    assert row["p_error"] == 0.25
    assert row["brier"] == (0.25 - int(error)) ** 2
    assert row["surprise_nats"] == pytest.approx(-math.log(0.25 if error else 0.75))


def test_error_after_long_success_sequence():
    # Beta(1,10001): 10000回成功後のエラー。逆KL・強引な0補正を落とす。
    error = metrics.beta_error_prediction(1, 10001, True)
    expected = math.log(10002) + 1 - math.fsum(1 / k for k in range(1, 10003))
    assert error["kl_nats"] == pytest.approx(expected, abs=1e-11)
    assert error["kl_nats"] > metrics.beta_error_prediction(1, 10001, False)["kl_nats"]
    assert error["surprise_nats"] == pytest.approx(math.log(10002))


@pytest.mark.parametrize("alpha,beta,error", [(0, 1, False), (-1, 2, True),
    (math.inf, 1, False), (1, math.nan, True), (1, 1, "false"), (1, 1, 0)])
def test_invalid_calculation_input(alpha, beta, error):
    with pytest.raises(ValueError):
        metrics.beta_error_prediction(alpha, beta, error)


@pytest.mark.parametrize("values,raises", [([0, 10], True), ([math.nan, 0], True),
                                          ([0, math.log(2) + 1e-12], False)])
def test_invalid_kl_is_not_hidden(monkeypatch, values, raises):
    import scipy.special
    monkeypatch.setattr(scipy.special, "digamma", Mock(side_effect=values))
    if raises:
        with pytest.raises(ValueError):
            metrics.beta_error_prediction(1, 1, True)
    else:
        assert metrics.beta_error_prediction(1, 1, True)["kl_nats"] == 0


def test_order_tool_separation_and_global_baseline(fixed_io):
    observer = metrics.ErrorPredictionObserver("new")
    events = [invocation(0, error=True), invocation(1), invocation(2, tool="B")]
    saved = copy.deepcopy(events)
    observer.observe(events)
    rows = lines(fixed_io, "error_prediction")
    assert [(r["tool"], r["alpha"], r["beta"], r["p_error"]) for r in rows] == [
        ("A", 1, 1, 0.5), ("A", 2, 1, 2/3), ("B", 1, 1, 0.5)]
    assert [(r["baseline"]["alpha"], r["baseline"]["beta"]) for r in rows] == [(1, 1), (2, 1), (2, 2)]
    assert all(r["prior"] == r["baseline"]["prior"] == {"alpha": 1, "beta": 1} for r in rows)
    assert all(r["prediction_timing"] == "posthoc_from_preceding_invocations" for r in rows)
    assert events == saved and observer.counts == {"A": (2, 2), "B": (1, 2)}


def test_rebuild_all_runs_and_shared_validation(fixed_io):
    valid = [invocation(), invocation(run="other", error=True),
             invocation(chain_position=1), invocation(attempt_id="other")]
    bad = [invocation(5, is_error="false"), invocation(6, is_error=None),
           invocation(7, is_error=0), invocation(8, run_id=""),
           invocation(9, chain_position=True), invocation(10, invocation_position=-1),
           invocation(11, tool=""), valid[0]]
    missing = invocation(12)
    del missing["is_error"]
    bad.append(missing)
    path = fixed_io / metrics.METRICS_FILE_NAME
    path.write_bytes(b"broken\n\xff\n[]\n" + b"".join(
        (json.dumps(e) + "\n").encode() for e in [*valid, *bad, {"event_type": "cycle"}]))
    rebuilt = metrics.ErrorPredictionObserver("new")
    report = rebuilt.rebuild()
    assert report["accepted"] == 4 and report["excluded"] == 12
    assert report["exclusion_reasons"] == {
        "invalid_json": 3, "invalid_is_error": 4, "invalid_key": 3, "invalid_tool": 1, "duplicate": 1}
    assert rebuilt.counts == {"A": (2, 4)} and rebuilt.total == (2, 4)
    live = metrics.ErrorPredictionObserver("live")
    live.observe(valid + bad)
    assert live.counts == rebuilt.counts and live.seen == rebuilt.seen
    exclusions = lines_after_corruption(path, "error_prediction_excluded")
    assert [r["reason"] for r in exclusions] == [
        "invalid_is_error", "invalid_is_error", "invalid_is_error", "invalid_key",
        "invalid_key", "invalid_key", "invalid_tool", "duplicate", "invalid_is_error"]
    rebuilt.observe([invocation(run="new")])
    row = lines_after_corruption(path, "error_prediction")[-1]
    assert (row["alpha"], row["beta"]) == (2, 4)


def lines_after_corruption(path, kind):
    events = []
    for line in path.read_bytes().splitlines():
        try:
            event = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(event, dict) and event.get("event_type") == kind:
            events.append(event)
    return events


@pytest.mark.parametrize("value", [None, "false", 0])
def test_builder_preserves_invalid_observation(value):
    event = _build([_record(is_error=value)])[0]
    assert event["is_error"] is value
    assert metrics._invocation_exclusion(event, set()) == "invalid_is_error"


def test_runtime_flag_not_status_or_result(fixed_io):
    observer = metrics.ErrorPredictionObserver("r")
    observer.observe([invocation(0, status="tool_error", result="Error: business", error=False),
                      invocation(1, status="rejected", result="[REJECTED]", error=True)])
    rows = lines(fixed_io, "error_prediction")
    assert [(r["is_error"], r["alpha"], r["beta"]) for r in rows] == [
        (False, 1, 1), (True, 1, 2)]


def test_empty_history_and_startup_emit_failure(monkeypatch, fixed_io):
    observer = metrics.ErrorPredictionObserver("r")
    report = observer.rebuild()
    assert report["accepted"] == report["excluded"] == 0 and report["history_reset"] is False
    observer.observe([])
    assert lines(fixed_io, "error_prediction") == []
    metrics.emit_tool_invocations([invocation(error=True)])
    with monkeypatch.context() as patch:
        patch.setattr(metrics, "_atomic_append_jsonl", Mock(side_effect=OSError("disk")))
        report = observer.rebuild()
    assert report["accepted"] == 1 and report["history_reset"] is False
    observer.observe([invocation(run="r")])
    assert lines(fixed_io, "error_prediction")[-1]["p_error"] == 2/3


def test_all_invalid_history_and_live_batch(fixed_io):
    bad = [invocation(is_error="false"), invocation(1, run_id="")]
    metrics.emit_tool_invocations(bad)
    observer = metrics.ErrorPredictionObserver("r")
    report = observer.rebuild()
    assert report["accepted"] == 0 and report["excluded"] == 2
    observer.observe(bad)
    assert observer.counts == {} and observer.total == (1, 1)
    assert lines(fixed_io, "error_prediction") == []


def test_rebuild_failure_is_temporary_and_linked(monkeypatch, fixed_io):
    metrics.emit_tool_invocations([invocation(error=True)])
    observer = metrics.ErrorPredictionObserver("broken-start")
    real = Path.open

    class BrokenRead:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __iter__(self):
            yield (json.dumps(invocation(error=True)) + "\n").encode()
            raise OSError("read interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", lambda p, *a, **kw: BrokenRead() if a == ("rb",) else real(p, *a, **kw))
        report = observer.rebuild()
    assert report["accepted"] == 0 and report["history_reset"] is True
    current = invocation(run="broken-start")
    metrics.emit_tool_invocations([current])
    observer.observe([current])
    row = lines(fixed_io, "error_prediction")[-1]
    assert row["history_id"] == report["history_id"] == "broken-start"
    assert row["history_reset"] is True and row["history_scope"] == "since_startup"
    assert row["p_error"] == 0.5
    recovered = metrics.ErrorPredictionObserver("recovered")
    assert recovered.rebuild()["accepted"] == 2
    recovered.observe([invocation(run="recovered")])
    row = lines(fixed_io, "error_prediction")[-1]
    assert (row["alpha"], row["beta"]) == (2, 2)
    assert row["history_reset"] is False and row["history_scope"] == "profile"


@pytest.mark.parametrize("fault", ["calculation", "append", "index", "scipy"])
def test_prediction_failures_keep_all_counts(monkeypatch, fixed_io, fault):
    events = [invocation(0, error=True), invocation(1), invocation(2, error=True)]
    metrics.emit_tool_invocations(events)
    observer = metrics.ErrorPredictionObserver("new")
    with monkeypatch.context() as patch:
        if fault == "calculation":
            real = metrics.beta_error_prediction
            def calculate(*args):
                if call.call_count == 1:
                    raise ValueError("KL")
                return real(*args)
            call = Mock(side_effect=calculate)
            patch.setattr(metrics, "beta_error_prediction", call)
        elif fault == "append":
            real = metrics._atomic_append_jsonl
            def append(path, row):
                if row["event_type"] == "error_prediction" and row["invocation_position"] == 0:
                    raise OSError("prediction disk")
                return real(path, row)
            patch.setattr(metrics, "_atomic_append_jsonl", append)
        elif fault == "index":
            patch.setattr(metrics, "_update_metrics_index", Mock(side_effect=OSError("index")))
        else:
            real = builtins.__import__
            def importing(name, *a, **kw):
                if name == "scipy.special":
                    raise ImportError("scipy unavailable")
                return real(name, *a, **kw)
            patch.setattr(builtins, "__import__", importing)
        observer.observe(events)
    assert observer.counts == {"A": (3, 2)} and observer.total == (3, 2)
    predictions = lines(fixed_io, "error_prediction")
    assert [r["invocation_position"] for r in predictions] == (
        [] if fault == "scipy" else [0, 1, 2] if fault == "index" else [1, 2])
    if predictions:
        assert (predictions[-1]["alpha"], predictions[-1]["beta"]) == (2, 2)
    assert metrics.summarize_error_prediction(lines(fixed_io))["missing"] == (
        3 if fault == "scipy" else 0 if fault == "index" else 1)
    restored = metrics.ErrorPredictionObserver("restart")
    assert restored.rebuild()["accepted"] == 3 and restored.counts == observer.counts
    observer.observe([invocation(3)])
    assert lines(fixed_io, "error_prediction")[-1]["p_error"] == 3/5


@pytest.mark.parametrize("fail_at", [0, 2, None])
def test_main_saved_prefix_only(monkeypatch, fixed_io, fail_at):
    observer = metrics.ErrorPredictionObserver("r")
    real = metrics._atomic_append_jsonl
    writes = []
    def append(path, event):
        if event["event_type"] == "tool_invocation":
            if len(writes) == fail_at:
                raise OSError("invocation disk")
            writes.append(event["tool_id"])
        real(path, event)
    monkeypatch.setattr(metrics, "_atomic_append_jsonl", append)
    funcs, *_ = _fire_harness(monkeypatch, fixed_io, stages=STAGES, proposal=PROPOSAL,
                              env_overrides={"_error_observer": observer})
    _run(funcs["observed"])
    expected = ["a", "b", "c", "d", "e"][:fail_at]
    assert writes == expected
    predictions = lines(fixed_io, "error_prediction")
    assert [r["tool_id"] for r in predictions] == expected
    assert len(observer.seen) == len(expected)
    assert len(lines(fixed_io, "outward_attempt")) == 1


@pytest.mark.parametrize("fault", [False, True])
def test_observation_has_no_behavior_effect(monkeypatch, fixed_io, fault):
    plain = _fire_harness(monkeypatch, fixed_io, stages=STAGES, proposal=PROPOSAL)
    result = _run(plain[0]["plain"])
    if fault:
        monkeypatch.setattr(metrics, "beta_error_prediction", Mock(side_effect=ValueError("KL")))
    observed = _fire_harness(monkeypatch, fixed_io, stages=STAGES, proposal=PROPOSAL)
    assert _run(observed[0]["observed"]) == result
    assert observed[1:3] == plain[1:3]  # state と prompt
    assert observed[4:6] == plain[4:6]  # 候補・G・選択
    assert [r.system_prompt for r in observed[3]] == [r.system_prompt for r in plain[3]]
    assert "error_prediction" not in json.dumps(observed[1])
    assert len(lines(fixed_io, "error_prediction_missing" if fault else "error_prediction")) == 5


def test_main_startup_restores_before_fire(fixed_io):
    metrics.emit_tool_invocations([invocation(error=True)])
    source = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    setup = next(n for n in main.body if isinstance(n, ast.Try) and any(
        isinstance(c, ast.Attribute) and c.attr == "rebuild" for c in ast.walk(n)))
    fire = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "_run_observed_fire")
    assert setup.lineno < fire.lineno
    state = {"run_id": "new"}
    env = {"state": state}
    exec(compile(ast.Module(body=[setup], type_ignores=[]), "startup", "exec"), env)
    assert env["_error_observer"].counts == {"A": (2, 1)}
    assert state == {"run_id": "new"}


def test_summary_scope_and_baseline_discrimination(fixed_io):
    observer = metrics.ErrorPredictionObserver("old")
    history = [invocation(i, tool="A" if i % 2 else "B", error=bool(i % 2)) for i in range(40)]
    metrics.emit_tool_invocations(history)
    observer.observe(history)
    restored = metrics.ErrorPredictionObserver("smoke")
    restored.rebuild()
    current = [invocation(i, tool="A" if i % 2 else "B", error=bool(i % 2), run="smoke") for i in range(10)]
    metrics.emit_tool_invocations(current)
    restored.observe(current)
    events = lines(fixed_io)
    summary = metrics.summarize_error_prediction(events, run_id="smoke")
    assert summary["count"] == summary["invocations"] == 10 and summary["missing"] == 0
    assert summary["log_loss_delta"] < -0.5 and summary["brier_delta"] < -0.2
    assert summary["log_loss"] == summary["mean_surprise_nats"]
    assert {k: v["count"] for k, v in summary["per_tool"].items()} == {"A": 5, "B": 5}
    assert metrics.summarize_error_prediction(events)["count"] == 50
    assert metrics.summarize_error_prediction(events + events, run_id="smoke") == summary
    for selected in ([], [invocation()], [{"event_type": "cycle"}], events):
        empty = metrics.summarize_error_prediction(selected, run_id="absent")
        assert empty["count"] == 0 and empty["log_loss"] is None and empty["per_tool"] == {}
