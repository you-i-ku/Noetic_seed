"""0d-1: 全実行の理由・引数・結果の観察 (OUTWARD_STEP0_PLAN §3-2)。

観察ファイルに 1 実行 1 行で残ること、iku の認知・state・fire の戻り値を
変えないこと、途中終了・書き出し失敗でも本体と 0b の観察が続くことを確かめる。
fire 本体は test_outward_metrics の AST ハーネスで実コードを動かす。
"""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import metrics
from core.config import TOOL_RESULT_MAX_CHARS
from core.providers.base import ToolUseBlock
from core.runtime.hooks import HookRunner, HookRunResult
from test_outward_metrics import _fire_harness, _outward_lines, _run, fixed_io  # noqa: F401

# 段ごとに理由・引数・結果をすべて別の値にする (末尾だけ・chain 0 だけを採る実装を落とす)。
# 1 段目の最後と 2 段目の最後は同じ reflect + path で、2 段目は重複スキップされる。
STAGES = [
    [ToolUseBlock(id="a", name="reflect", input={
        "tool_intent": "理由A", "tool_expected_outcome": "予想A", "note": "伝A",
        "payload": "引数A", "echo": "結果A"}),
     ToolUseBlock(id="b", name="x_like", input={
         "tool_intent": "理由B", "tool_expected_outcome": "予想B", "note": "伝B",
         "payload": "引数B", "echo": "[REJECTED] 拒否B"}),
     ToolUseBlock(id="c", name="reflect", input={
         "tool_intent": "理由C", "tool_expected_outcome": "予想C", "note": "伝C",
         "path": "p", "echo": "結果C"})],
    [ToolUseBlock(id="d", name="elyth_post", input={"payload": "引数D", "echo": "エラー: 失敗D"}),
     ToolUseBlock(id="e", name="reflect", input={
         "tool_intent": "理由E", "tool_expected_outcome": "予想E", "note": "伝E",
         "path": "p", "echo": "結果E"})],
]
PROPOSAL = ("1. [考える] → reflect+x_like (pe2=40, pec=0.2)\n"
            "2. [応答] → output_display (pe2=70, pec=0.5)")


def _invocation_lines(directory):
    path = directory / metrics.METRICS_FILE_NAME
    if not path.exists():
        return []
    return [e for e in (json.loads(s) for s in path.read_text(encoding="utf-8").splitlines())
            if e["event_type"] == "tool_invocation"]


def _harness(monkeypatch, directory, **kw):
    return _fire_harness(monkeypatch, directory, stages=STAGES, proposal=PROPOSAL, **kw)


# === 組み立て関数 (純粋) ===

def _record(**kw):
    base = {"chain_position": 0, "invocation_position": 0, "tool_id": "t", "tool": "reflect",
            "mode": "forced", "tool_input": {}, "output": "ok", "is_error": False,
            "in_cycle_result": False}
    base.update(kw)
    return base


def _build(records, entry_id="s_0004", cycle_id=4):
    return metrics.build_tool_invocation_events(
        records, run_id="r", attempt_id="r_1", entry_id=entry_id, cycle_id=cycle_id, time="T")


def test_build_empty():
    assert _build([]) == []


def test_build_approval_fields_split_and_missing():
    """承認 3 層が args に混ざる / 欠けた実行を落とす誤実装を検出する。"""
    events = _build([
        _record(tool_input={"tool_intent": "I", "tool_expected_outcome": "X", "note": "M",
                            "path": "a.txt"}),
        _record(tool_id="u", tool_input={"path": "b.txt"}),
    ])
    assert [(e["intent"], e["expect"], e["note"], e["args"]) for e in events] == [
        ("I", "X", "M", {"path": "a.txt"}), ("", "", "", {"path": "b.txt"})]


def test_business_message_and_note_are_separate():
    event = _build([_record(tool_input={"note": "自由欄", "message": "送信本文"})])[0]
    assert event["note"] == "自由欄" and event["args"] == {"message": "送信本文"}
    assert "message" not in event
    event = _build([_record(tool_input={"message": "送信本文"})])[0]
    assert event["note"] == "" and event["args"] == {"message": "送信本文"}


@pytest.mark.parametrize("kind", ["old", "new", "mixed"])
def test_old_new_observation_records_are_read_without_migration(monkeypatch, fixed_io, kind):
    """旧行の message と args.note を保存し、note 未記録と空文字を区別する。"""
    old = {"event_type": "tool_invocation", "tool": "reflect", "message": "旧承認欄",
           "args": {"note": "昔の業務引数"}}
    path = fixed_io / metrics.METRICS_FILE_NAME
    prefix = (json.dumps(old, ensure_ascii=False) + "\n").encode("utf-8")
    if kind in ("old", "mixed"):
        path.write_bytes(prefix)
    if kind in ("new", "mixed"):
        metrics.emit_tool_invocations(_build([_record(tool_input={
            "note": "新自由欄", "message": "新送信本文"})]))
    loaded = _invocation_lines(fixed_io)
    if kind in ("old", "mixed"):
        assert loaded[0] == old and "note" not in loaded[0]
        assert loaded[0].get("note", "") == ""  # 未記録であり、当時の空欄を示さない
        assert path.read_bytes().startswith(prefix)
    if kind in ("new", "mixed"):
        assert loaded[-1]["note"] == "新自由欄"
        assert loaded[-1]["args"] == {"message": "新送信本文"}
    assert len(loaded) == (2 if kind == "mixed" else 1)


@pytest.mark.parametrize("output,is_error,status", [
    ("[REJECTED] no", False, "rejected"), ("boom", True, "runtime_error"),
    ("エラー: x", False, "tool_error"), ("Error: x", False, "tool_error"), ("fine", False, "ok")])
def test_build_status_matches_0b(output, is_error, status):
    """0b と違う status 判定 (例: is_error を tool_error と呼ぶ) を検出する。"""
    event = _build([_record(output=output, is_error=is_error)])[0]
    observed = metrics.build_outward_execution("reflect", {}, output, is_error, "")
    assert event["status"] == observed["status"] == status
    assert event["is_error"] is is_error


def test_build_result_cap_boundary():
    """個別上限の境界: ちょうどは marker なし、1 字超で marker あり。"""
    exact, over = "a" * TOOL_RESULT_MAX_CHARS, "b" * (TOOL_RESULT_MAX_CHARS + 1)
    events = _build([_record(output=exact), _record(tool_id="u", output=over)])
    assert events[0]["result"] == exact
    assert events[1]["result"].startswith("b" * TOOL_RESULT_MAX_CHARS)
    assert f"{TOOL_RESULT_MAX_CHARS}/{TOOL_RESULT_MAX_CHARS + 1}字" in events[1]["result"]


def test_build_null_entry():
    event = _build([_record()], entry_id=None, cycle_id=None)[0]
    assert event["entry_id"] is None and event["cycle_id"] is None


# === fire 本体 (実コード) ===

def test_all_invocations_recorded_with_own_reason(monkeypatch, fixed_io):
    """chain 0 だけ・段の末尾だけ・同名の取り違え・重複スキップの誤判定を検出する。"""
    funcs, state, *_ = _harness(monkeypatch, fixed_io)
    _run(funcs["observed"])
    events = _invocation_lines(fixed_io)
    assert [(e["chain_position"], e["invocation_position"], e["tool_id"], e["tool"], e["mode"])
            for e in events] == [
        (0, 0, "a", "reflect", "forced"), (0, 1, "b", "x_like", "forced"),
        (0, 2, "c", "reflect", "forced"),
        (1, 0, "d", "elyth_post", "free"), (1, 1, "e", "reflect", "free")]
    assert [e["intent"] for e in events] == ["理由A", "理由B", "理由C", "", "理由E"]
    assert [e["expect"] for e in events] == ["予想A", "予想B", "予想C", "", "予想E"]
    assert [e["result"] for e in events] == ["結果A", "[REJECTED] 拒否B", "結果C", "エラー: 失敗D", "結果E"]
    assert [e["args"] for e in events] == [
        {"payload": "引数A", "echo": "結果A"}, {"payload": "引数B", "echo": "[REJECTED] 拒否B"},
        {"path": "p", "echo": "結果C"}, {"payload": "引数D", "echo": "エラー: 失敗D"},
        {"path": "p", "echo": "結果E"}]
    assert [e["status"] for e in events] == ["ok", "rejected", "ok", "tool_error", "ok"]
    # 結果リストに入ったのは 1 段目の末尾だけ (2 段目の末尾は重複スキップ)
    assert [e["in_cycle_result"] for e in events] == [False, False, True, False, False]
    entry = state["raw_events"][-1]
    assert {e["entry_id"] for e in events} == {entry["id"]}
    assert {e["cycle_id"] for e in events} == {4}
    assert {e["attempt_id"] for e in events} == {_outward_lines(fixed_io)[0]["attempt_id"]}
    # 既存の per_tool と同じ数・同じ並び
    assert [(p["tool_id"], p["chain_position"], p["invocation_position"])
            for p in entry["per_tool"]] == [
        (e["tool_id"], e["chain_position"], e["invocation_position"]) for e in events]


def test_observation_does_not_change_fire(monkeypatch, fixed_io):
    """観察の追加で entry・state・戻り値・prompt が変わる誤実装を検出する。"""
    plain = _harness(monkeypatch, fixed_io)
    result_plain = _run(plain[0]["plain"])
    observed = _harness(monkeypatch, fixed_io)
    result_observed = _run(observed[0]["observed"])
    assert result_plain == result_observed
    assert plain[1] == observed[1]
    assert plain[2] == observed[2]
    assert [r.system_prompt for r in plain[3]] == [r.system_prompt for r in observed[3]]
    assert "tool_invocation" not in json.dumps(observed[1], ensure_ascii=False)


def test_entry_missing_gives_null_keys(monkeypatch, fixed_io):
    """entry 作成前の例外で実行記録を捨てる / 偽の entry_id を入れる誤実装を検出する。"""
    funcs, state, *_ = _harness(monkeypatch, fixed_io, env_overrides={
        "calc_state_change_bonus": Mock(side_effect=RuntimeError("before entry"))})
    with pytest.raises(RuntimeError):
        _run(funcs["observed"])
    events = _invocation_lines(fixed_io)
    assert [e["tool_id"] for e in events] == ["a", "b", "c", "d", "e"]
    assert {e["entry_id"] for e in events} == {None} and {e["cycle_id"] for e in events} == {None}
    assert _outward_lines(fixed_io)[0]["end"] == "error"


def test_break_after_first_stage_keeps_records(monkeypatch, fixed_io):
    """2 段目が未呼出しで break しても、1 段目の記録と entry_id が残ること。"""
    funcs, state, *_ = _fire_harness(monkeypatch, fixed_io, stages=STAGES[:1], proposal=PROPOSAL)
    _run(funcs["observed"])
    events = _invocation_lines(fixed_io)
    assert [e["tool_id"] for e in events] == ["a", "b", "c"]
    assert {e["entry_id"] for e in events} == {state["raw_events"][-1]["id"]}


def test_emit_failure_isolated(monkeypatch, fixed_io):
    """書き出しの例外で戻り値・state が変わる / 0b の観察まで止まる誤実装を検出する。"""
    plain = _harness(monkeypatch, fixed_io)
    result_plain = _run(plain[0]["plain"])
    monkeypatch.setattr(metrics, "emit_tool_invocations", Mock(side_effect=OSError("disk")))
    observed = _harness(monkeypatch, fixed_io)
    assert _run(observed[0]["observed"]) == result_plain
    assert plain[1] == observed[1]
    assert _invocation_lines(fixed_io) == []
    assert len(_outward_lines(fixed_io)) == 1


def test_build_failure_isolated(monkeypatch, fixed_io):
    """組み立ての例外でも本体の戻り値と 0b の観察が保たれること。"""
    plain = _harness(monkeypatch, fixed_io)
    result_plain = _run(plain[0]["plain"])
    monkeypatch.setattr(metrics, "build_tool_invocation_events", Mock(side_effect=ValueError("x")))
    observed = _harness(monkeypatch, fixed_io)
    assert _run(observed[0]["observed"]) == result_plain
    assert len(_outward_lines(fixed_io)) == 1


def test_index_failure_does_not_duplicate(monkeypatch, fixed_io):
    """index 更新だけ失敗した時に再試行して行を重複させる誤実装を検出する。"""
    monkeypatch.setattr(metrics, "_update_metrics_index", Mock(side_effect=OSError("index")))
    written = metrics.emit_tool_invocations(_build([_record(tool_id="x"), _record(tool_id="y")]))
    assert written == 2
    assert [e["tool_id"] for e in _invocation_lines(fixed_io)] == ["x", "y"]


def test_summarize_outward_ignores_invocations(monkeypatch, fixed_io):
    funcs, *_ = _harness(monkeypatch, fixed_io)
    _run(funcs["observed"])
    lines = [json.loads(s) for s in (fixed_io / metrics.METRICS_FILE_NAME)
             .read_text(encoding="utf-8").splitlines()]
    assert metrics.summarize_outward(lines)["attempts"] == 1


def test_append_failure_stops_without_retry(monkeypatch, fixed_io):
    """append 途中失敗で残りを書き続ける / 再試行する誤実装を検出する。"""
    real = metrics._atomic_append_jsonl

    def first_only(path, event):
        if append.call_count > 1:
            raise OSError("disk")
        real(path, event)

    append = Mock(side_effect=first_only)
    monkeypatch.setattr(metrics, "_atomic_append_jsonl", append)
    with pytest.raises(OSError):
        metrics.emit_tool_invocations(_build([_record(tool_id=t) for t in ("x", "y", "z")]))
    assert append.call_count == 2
    assert [e["tool_id"] for e in _invocation_lines(fixed_io)] == ["x"]


def test_real_pre_hook_rejection_recorded(monkeypatch, fixed_io):
    """本物の拒否経路 (pre hook の deny、is_error=True) の実行も理由付きで残ること。"""
    runner = HookRunner()
    runner.register_pre(lambda name, inp: HookRunResult.deny(["no"]) if name == "x_like"
                        else HookRunResult.allow())
    stages = copy.deepcopy(STAGES)
    stages[0][1].input["echo"] = "liked"  # 拒否は tool の戻り値ではなく runtime が作る
    funcs, *_ = _fire_harness(monkeypatch, fixed_io, stages=stages, proposal=PROPOSAL,
                              hook_runner=runner)
    _run(funcs["observed"])
    event = next(e for e in _invocation_lines(fixed_io) if e["tool_id"] == "b")
    assert event["is_error"] is True and event["status"] == "rejected"
    assert event["intent"] == "理由B" and event["result"].startswith("[REJECTED] denied by pre hook")


def test_runtime_error_in_next_stage_keeps_first_stage(monkeypatch, fixed_io):
    """2 段目の runtime 例外で 1 段目の記録を失う誤実装を検出する。"""
    funcs, state, *_ = _fire_harness(monkeypatch, fixed_io, stages=[STAGES[0], RuntimeError("llm2")],
                                     proposal=PROPOSAL)
    _run(funcs["observed"])
    events = _invocation_lines(fixed_io)
    assert [e["tool_id"] for e in events] == ["a", "b", "c"]
    assert {e["entry_id"] for e in events} == {state["raw_events"][-1]["id"]}
    assert _outward_lines(fixed_io)[0]["end"] == "runtime_error"


def test_record_is_independent_copy():
    """採取後に元の引数を変えても記録が変わらないこと (浅いコピーへの退行を検出)。"""
    import ast
    source = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Dict)
             and any(isinstance(k, ast.Constant) and k.value == "tool_input" for k in n.keys)]
    assert len(calls) == 1
    value = calls[0].values[[k.value for k in calls[0].keys].index("tool_input")]
    assert ast.unparse(value) == "copy.deepcopy(_r.tool_input or {})"
