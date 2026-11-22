"""0d-2: 実行別の主観/raw・結果参照・検索。外部 I/O と embedding は固定する。"""
import copy
import json
import re
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import memory, state as state_module, subjective_view
from core.config import cap_tool_result
from core.providers.base import ToolUseBlock
from core.runtime.hooks import HookRunner, HookRunResult
from tools import memory_tool
from test_outward_metrics import _fire_harness, _run, fixed_io  # noqa: F401
from test_tool_invocation_observation import STAGES


def _fire(monkeypatch, directory, stages, **kwargs):
    proposal = "1. [考える] → " + "+".join(["reflect"] * len(stages)) + " (pe2=40, pec=0.2)"
    harness = _fire_harness(monkeypatch, directory, stages=stages, proposal=proposal,
                            env_overrides={"cap_tool_result": cap_tool_result}, **kwargs)
    _run(harness[0]["plain"])
    return harness[1]["raw_events"][-1]


def _round_trip(monkeypatch, directory, entry):
    monkeypatch.setattr(memory, "MEMORY_DIR", directory)
    monkeypatch.setattr(state_module, "MEMORY_DIR", directory)
    memory._archive_entries([entry])
    loaded = {}
    state_module._rebuild_views_from_jsonl(loaded)
    raw, subj = loaded["raw_events"][-1], loaded["subjective_entries"][-1]
    assert raw["invocations"] == entry["invocations"]
    assert subj["per_tool"] == entry["per_tool"]
    assert "invocations" not in subj and "per_tool" not in raw
    assert "_unclassified_fields" not in raw
    return {**raw, **subj}


def test_business_message_survives_fire_and_memory(monkeypatch, fixed_io):
    call = ToolUseBlock(id="both", name="reflect", input={
        "tool_intent": "理由", "tool_expected_outcome": "予想",
        "note": "自由欄", "message": "業務本文", "echo": "結果"})
    entry = _round_trip(monkeypatch, fixed_io, _fire(monkeypatch, fixed_io, [[call]]))
    assert entry["args"] == {"message": "業務本文", "echo": "結果"}
    assert entry["invocations"][0]["args"] == {"message": "業務本文", "echo": "結果"}


@pytest.mark.parametrize("kind", ["old", "new", "mixed"])
def test_record_read_preserves_old_fields(monkeypatch, fixed_io, kind):
    old = {"tool": "reflect", "message": "旧欄", "args": {"note": "昔の業務引数"},
           "result": "旧結果"}
    new = {"tool": "reflect", "note": "新欄", "args": {"message": "業務本文"},
           "result": "新結果"}
    invs = [old] if kind == "old" else [new] if kind == "new" else [old, new]
    entry = {"id": "s_0001", "time": "T", "tool": "reflect", "result": "",
             "invocations": invs, "per_tool": []}
    loaded = _round_trip(monkeypatch, fixed_io, entry)
    assert loaded["invocations"] == invs
    if kind != "new":
        inv = loaded["invocations"][0]
        assert "note" not in inv and inv.get("note", "") == ""
        assert inv["args"]["note"] == "昔の業務引数"


def test_record_every_execution_and_round_trip(monkeypatch, fixed_io):
    """末尾だけ保存・同名や重複IDで結合・重複スキップの結果消失・承認欄混入を検出。"""
    stages = copy.deepcopy(STAGES)
    for stage in stages:
        for call in stage:
            call.id = "duplicate"
    stages[1][0].id = ""
    entry = _fire(monkeypatch, fixed_io, stages)
    loaded = _round_trip(monkeypatch, fixed_io, entry)
    invs, pts = loaded["invocations"], loaded["per_tool"]
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    assert [(p["chain_position"], p["invocation_position"]) for p in pts] == positions
    assert [(p["chain_position"], p["invocation_position"]) for p in invs] == positions
    assert [p["tool_id"] for p in invs] == ["duplicate"] * 3 + ["", "duplicate"]
    assert [p["intent"] for p in pts] == ["理由A", "理由B", "理由C", "", "理由E"]
    assert [p["expect"] for p in pts] == ["予想A", "予想B", "予想C", "", "予想E"]
    assert [p["status"] for p in invs] == ["ok", "ok", "ok", "ok", "ok"]
    assert [p["in_cycle_result"] for p in invs] == [False, False, True, False, False]
    expected = ["結果A", "[REJECTED] 拒否B", "結果C", "エラー: 失敗D", "結果E"]
    assert [memory.invocation_result(loaded, p) for p in invs] == expected
    for inv, call in zip(invs, [c for stage in stages for c in stage]):
        assert inv["args"] == {k: v for k, v in call.input.items()
                               if k not in ("tool_intent", "tool_expected_outcome", "note")}
        assert ("result_ref" in inv) == inv["in_cycle_result"]
        assert ("result" in inv) != ("result_ref" in inv)


@pytest.mark.parametrize("tail,error", [
    ([], "no_tool: completed"),
    (RuntimeError("後段の例外"), "runtime error: 後段の例外"),
])
def test_interrupted_chain_preserves_previous_executions(monkeypatch, fixed_io, tail, error):
    """後段のbreakで前段の理由や結果を捨てる誤実装を検出。個別保存と参照の両方を確認。"""
    first = [
        ToolUseBlock(id="first", name="reflect", input={
            "tool_intent": "前段の最初の理由", "tool_expected_outcome": "最初の予想",
            "echo": "個別に保存する結果"}),
        ToolUseBlock(id="last", name="reflect", input={
            "tool_intent": "前段の最後の理由", "tool_expected_outcome": "最後の予想",
            "echo": "参照で保存する結果"}),
    ]
    entry = _fire(monkeypatch, fixed_io, [first, tail])
    assert entry["id"] == "s_0004" and entry["parse_error"] == error
    entry = _round_trip(monkeypatch, fixed_io, entry)
    assert [(pt["tool_id"], pt["chain_position"], pt["invocation_position"],
             pt["intent"], pt["expect"]) for pt in entry["per_tool"]] == [
        ("first", 0, 0, "前段の最初の理由", "最初の予想"),
        ("last", 0, 1, "前段の最後の理由", "最後の予想"),
    ]
    invs = entry["invocations"]
    assert [(inv["tool_id"], inv["chain_position"], inv["invocation_position"],
             inv["status"], inv["in_cycle_result"]) for inv in invs] == [
        ("first", 0, 0, "ok", False), ("last", 0, 1, "ok", True),
    ]
    assert invs[0]["result"] == "個別に保存する結果" and "result_ref" not in invs[0]
    assert invs[1]["result_ref"] == {"start": 10, "length": len("参照で保存する結果")}
    assert "result" not in invs[1]
    assert entry["result"] == "[reflect]\n参照で保存する結果"
    assert [memory.invocation_result(entry, inv) for inv in invs] == [
        "個別に保存する結果", "参照で保存する結果"]


@pytest.mark.parametrize("failure,status", [("reject", "rejected"), ("raise", "runtime_error")])
def test_real_failure_keeps_own_reason(monkeypatch, fixed_io, failure, status):
    """本物のpre-hook拒否・tool例外でも理由と結果を欠落させる誤実装を検出。"""
    runner = HookRunner()
    if failure == "reject":
        runner.register_pre(lambda *a: HookRunResult.deny(["拒否の証拠"]))
    harness = _fire_harness(monkeypatch, fixed_io, stages=[[ToolUseBlock(
        id="bad", name="reflect", input={"tool_intent": "失敗の理由"})]], hook_runner=runner)
    if failure == "raise":
        runtime = harness[0]["plain"].__globals__["_runtime"]
        # 既存 registry の実 tool handler で例外を起こす。
        runtime.tool_registry.get("reflect").handler = Mock(side_effect=RuntimeError("例外の証拠"))
    _run(harness[0]["plain"])
    entry = harness[1]["raw_events"][-1]
    assert entry["per_tool"][0]["intent"] == "失敗の理由"
    assert entry["invocations"][0]["status"] == status
    assert ("拒否の証拠" if failure == "reject" else "例外の証拠") in memory.invocation_result(
        entry, entry["invocations"][0])


@pytest.mark.parametrize("extra", [0, 1, 10002])
def test_final_result_boundary_and_loss(monkeypatch, fixed_io, extra):
    """3段で50000字ちょうど/1字超、4段で部分/全欠落。常に参照・常に全文の両方を落とす。"""
    # 見出し10字×3 + 区切り5字×2 = 40。各結果は20000字以下。
    outputs = ["あ" * 20000, "い" * 20000, "う" * (9960 + min(extra, 10000))]
    if extra == 10002:
        outputs.append("後段は全欠落")
    stages = [[ToolUseBlock(id=str(i), name="reflect", input={"echo": result})]
              for i, result in enumerate(outputs)]
    entry = _round_trip(monkeypatch, fixed_io, _fire(monkeypatch, fixed_io, stages))
    assert len(entry["result"]) == 50000
    assert [memory.invocation_result(entry, inv) for inv in entry["invocations"]] == outputs
    for i, inv in enumerate(entry["invocations"]):
        assert inv["in_cycle_result"] is True
        assert ("result_ref" in inv) == (i < 2 or extra == 0)
        assert ("result" in inv) != ("result_ref" in inv)


def test_unicode_identical_empty_and_individual_cap(monkeypatch, fixed_io):
    """findで同一結果を取り違える・バイト位置・空文字を欠損扱い・cap省略を検出。"""
    outputs = ["日本語\n同じ結果🙂", "日本語\n同じ結果🙂", "", "界" * 20001]
    stages = [[ToolUseBlock(id=str(i), name="reflect", input={"echo": result})]
              for i, result in enumerate(outputs)]
    entry = _round_trip(monkeypatch, fixed_io, _fire(monkeypatch, fixed_io, stages))
    cursor = 0
    for inv, output in zip(entry["invocations"], outputs):
        result = cap_tool_result(output)
        assert inv["result_ref"] == {"start": cursor + len("[reflect]\n"), "length": len(result)}
        assert memory.invocation_result(entry, inv) == result
        cursor += len("[reflect]\n") + len(result) + len("\n---\n")
    assert memory.invocation_result(entry, {"result": "", "result_ref": {"start": 0, "length": 9}}) == ""
    assert memory.invocation_result(entry, {}) == ""


def _search_entry():
    first = "前段 " * 200
    second = "outcome_token 後段の結果"
    return {"id": "target", "time": "T", "tool": "reflect+reflect", "intent": "最初の理由",
            "result": first + second,
            "per_tool": [
                {"chain_position": 0, "invocation_position": 0, "tool_id": "same",
                 "intent": "前段理由 " * 200, "expect": "前段予想"},
                {"chain_position": 1, "invocation_position": 0, "tool_id": "same",
                 "intent": "reason_token 後段の理由", "expect": "expect_token 後段の予想"}],
            "invocations": [
                {"chain_position": 0, "invocation_position": 0, "tool_id": "same", "tool": "reflect",
                 "status": "ok", "args": {}, "result_ref": {"start": 0, "length": len(first)}},
                {"chain_position": 1, "invocation_position": 0, "tool_id": "same", "tool": "reflect",
                 "status": "rejected", "args": {"payload": "後段引数"},
                 "result_ref": {"start": len(first), "length": len(second)}}]}


def _search_setup(monkeypatch, directory, entries, route):
    monkeypatch.setattr(memory_tool, "MEMORY_DIR", directory)
    monkeypatch.setattr(memory, "MEMORY_DIR", directory)
    (directory / "archive_test.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries), encoding="utf-8")
    monkeypatch.setattr(memory_tool, "is_vector_ready", lambda: route != "unavailable")

    def embed(texts):
        if route == "failed":
            raise RuntimeError("embedding unavailable")
        query = set(re.findall(r"\w+", texts[0].lower()))
        # 決定的なベクトルで、完全一致・部分一致・不一致の順位を区別する。
        return [[len(query & set(re.findall(r"\w+", t.lower()))) / len(query), 1.0]
                for t in texts]

    spy = Mock(side_effect=embed)
    monkeypatch.setattr(memory_tool, "_embed_sync", spy)
    return spy


@pytest.mark.parametrize("route", ["vector", "unavailable", "failed"])
@pytest.mark.parametrize("query", ["reason_token", "expect_token", "outcome_token"])
@pytest.mark.parametrize("individual", [False, True])
def test_search_each_execution_and_id_detail(monkeypatch, fixed_io, route, query, individual):
    """entry先頭のみ/全文連結後400字/参照結果を検索しない/ID結合の誤実装を落とす。"""
    entry = _search_entry()
    if individual:
        inv = entry["invocations"][1]
        inv["result"] = memory.invocation_result(entry, inv)
        del inv["result_ref"]
    # rawの並びに依存せず、位置の組で同じ実行を結合する。
    entry["invocations"].reverse()
    spy = _search_setup(monkeypatch, fixed_io, [{"id": "control", "intent": "無関係"}, entry], route)
    result = memory_tool._search_memory({"query": query, "max_results": 1})
    assert "id=target " in result and "id=control" not in result
    assert "1-0 reflect [rejected] intent=reason_token 後段の理由" in result
    assert "後段引数" not in result and "outcome_token" not in result
    detail = memory_tool._search_memory({"id": "arget"})
    assert "expect=expect_token 後段の予想" in detail
    assert 'args={"payload": "後段引数"}' in detail
    assert "result=outcome_token 後段の結果" in detail
    if route == "vector":
        assert all(len(t) <= 400 for t in spy.call_args.args[0][1:])
        assert len(spy.call_args.args[0]) == 5  # query + 旧entry + 新entryの3照合文
    elif route == "unavailable":
        spy.assert_not_called()
    else:
        spy.assert_called_once()


@pytest.mark.parametrize("route", ["vector", "unavailable", "failed"])
def test_scores_use_max_not_sum_or_join(monkeypatch, fixed_io, route):
    """異なる実行の一致を足す誤実装を落とす。完全/部分/不一致を別IDで比較。"""
    partial = _search_entry()
    partial.update(id="partial", intent="alpha", result="")
    partial["per_tool"][0].update(intent="alpha", expect="")
    partial["per_tool"][1].update(intent="beta", expect="")
    for inv in partial["invocations"]:
        inv.pop("result_ref")
        inv["result"] = ""
    full = {"id": "full", "intent": "alpha beta"}
    _search_setup(monkeypatch, fixed_io, [partial, {"id": "none", "intent": "gamma"}, full], route)
    result = memory_tool._search_memory({"query": "alpha beta", "max_results": 1})
    assert "id=full " in result and "id=partial" not in result


@pytest.mark.parametrize("route", ["unavailable", "failed"])
def test_legacy_keyword_full_text_and_display(monkeypatch, fixed_io, route):
    """旧entryも400字に切る退行、旧per_toolに新表示を足す退行を検出。"""
    old = {"id": "legacy", "time": "T", "tool": "reflect", "intent": "古い理由",
           "result": "x " * 250 + " legacy_tail", "per_tool": [{"tool": "reflect"}]}
    _search_setup(monkeypatch, fixed_io, [_search_entry(), old], route)
    assert memory_tool._search_memory({"query": "legacy_tail"}) == (
        "[100%] id=legacy time=T tool=reflect intent=古い理由")
    assert memory_tool._search_memory({"id": "legacy"}) == (
        "id=legacy time=T tool=reflect intent=古い理由 result=" + old["result"][:200])
    assert "一致するエントリなし" in memory_tool._search_memory({"query": "nomatch"})


def test_legacy_vector_ranking_and_display(monkeypatch, fixed_io):
    """旧ベクトル経路の照合・順位・同点順・表示を維持。キーワードへの退行も検出。"""
    entries = [
        {"id": "partial", "time": "T1", "tool": "reflect", "intent": "alpha", "result": ""},
        {"id": "tie_z", "time": "T2", "tool": "reflect", "intent": "alpha beta " + "長" * 210,
         "result": "結果" * 150, "per_tool": [{"tool": "reflect", "e2": "50%"}]},
        {"id": "none", "time": "T3", "tool": "reflect", "intent": "gamma",
         "result": "x " * 250 + " alpha beta"},
        {"id": "tie_a", "time": "T4", "tool": "reflect", "intent": "alpha beta", "result": ""},
    ]
    spy = _search_setup(monkeypatch, fixed_io, entries, "vector")
    result = memory_tool._search_memory({"query": "alpha beta", "max_results": 4})
    # 旧実装のcosine順位: 完全一致2件(同点はarchive順)、部分一致、先頭400字で不一致。
    assert result == "\n".join([
        "[100%] id=tie_z time=T2 tool=reflect intent=alpha beta " + "長" * 89,
        "[100%] id=tie_a time=T4 tool=reflect intent=alpha beta",
        "[95%] id=partial time=T1 tool=reflect intent=alpha",
        "[71%] id=none time=T3 tool=reflect intent=gamma",
    ])
    spy.assert_called_once_with(["alpha beta"] + [
        f"{entry['intent']} {entry['result']}"[:400] for entry in entries])
    assert memory_tool._search_memory({"id": "tie_z"}) == (
        "id=tie_z time=T2 tool=reflect intent=alpha beta " + "長" * 189
        + " result=" + "結果" * 100)


@pytest.mark.parametrize("missing", ["invocations", "reason_fields"])
def test_partial_old_schema_stays_legacy(missing):
    """invocations欠損/理由欄欠損の旧形式で追加照合・追加表示する誤実装を検出。"""
    entry = _search_entry()
    if missing == "invocations":
        entry.pop("invocations")
    else:
        for pt in entry["per_tool"]:
            pt.pop("intent")
            pt.pop("expect")
    assert memory_tool._memory_execution_lines(entry, detail=True) == ""
    assert memory_tool._memory_search_texts(entry) == [f"{entry['intent']} {entry['result']}"]


def test_display_limits_and_empty_search(monkeypatch, fixed_io):
    """一覧100字/詳細200字を越える表示と、空archiveを例外にする誤実装を検出。"""
    entry = _search_entry()
    entry["per_tool"][1].update(intent="あ" * 201, expect="い" * 201)
    entry["invocations"][1]["result"] = "う" * 201
    entry["invocations"][1]["args"] = {"note": "え" * 201}
    listing = memory_tool._memory_execution_lines(entry)
    detail = memory_tool._memory_execution_lines(entry, detail=True)
    assert "あ" * 100 in listing and "あ" * 101 not in listing
    for char in "あいう":
        assert char * 200 in detail and char * 201 not in detail
    assert json.dumps(entry["invocations"][1]["args"], ensure_ascii=False)[:200] in detail
    _search_setup(monkeypatch, fixed_io, [], "unavailable")
    assert memory_tool._search_memory({"query": "anything"}) == "記憶ファイルが空です"


def test_prompt_and_saved_embedding_unchanged(monkeypatch, fixed_io):
    """per_toolの理由を自動prompt/保存embeddingへ混ぜる誤実装を検出。"""
    entry = _search_entry()
    entry["expect"] = "最初の予想"
    old = {k: v for k, v in entry.items() if k not in ("per_tool", "invocations")}
    old["per_tool"] = [{k: v for k, v in pt.items() if k not in ("intent", "expect")}
                       for pt in entry["per_tool"]]
    assert memory.format_memories_for_prompt([{**entry, "kind": "subjective"}]) == (
        memory.format_memories_for_prompt([{**old, "kind": "subjective"}]))
    embed = Mock(return_value=[[1.0, 0.0]])
    monkeypatch.setattr(memory, "is_vector_ready", lambda: True)
    monkeypatch.setattr(memory, "_embed_sync", embed)
    monkeypatch.setattr(memory, "_generate_memory_metadata", lambda *a: {})
    memory._enrich_subjective_inline(entry)
    embed.assert_called_once_with(["intent: 最初の理由\nexpect: 最初の予想"])
    outputs = []
    for item in (old, entry):
        harness = _fire_harness(monkeypatch, fixed_io)
        raw, subj, _ = memory._split_entry_fields(item)
        harness[1]["raw_events"] = [raw]
        harness[1]["subjective_entries"] = [subj]
        monkeypatch.setattr(memory, "get_relevant_memories",
                            lambda *a, row=item, **kw: [{**row, "kind": "subjective"}])
        subjective = subjective_view.build_subjective_state(harness[1])
        _run(harness[0]["plain"])
        outputs.append((harness[2], subjective))
    assert outputs[0] == outputs[1] and outputs[0][0]
