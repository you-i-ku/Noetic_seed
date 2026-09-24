"""step 0a-2: main の実コードを AST から取り出し、runtime と接続して検証。

常駐ループ・LLM・ディスク保存は起動しない。候補選択後から fire の return
までと、外側の E 更新・pressure reset を実行する（処理の写しは作らない）。
実行: python -B tests/test_failed_tool_learning.py
"""
import ast
import copy
import json
import re
import sys
import traceback
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.entropy import apply_negentropy
from core.eval import _update_energy
from core.providers.base import AssistantMessage, BaseProvider, ToolUseBlock
from core.runtime.conversation import ConversationRuntime
from core.runtime.hooks import HookRunner, HookRunResult
from core.runtime.permissions import PermissionEnforcer, PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec

MAIN_PATH = Path(__file__).resolve().parent.parent / "main.py"
MAIN_SOURCE = globals().get("MAIN_SOURCE", MAIN_PATH.read_text(encoding="utf-8"))


def _compile_main_parts(env):
    tree = ast.parse(MAIN_SOURCE)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    hook = next(n for n in main.body if isinstance(n, ast.FunctionDef)
                and n.name == "_post_hook_with_sync")
    registration = next(n for n in main.body if isinstance(n, ast.Expr)
                        and isinstance(n.value, ast.Call)
                        and any(isinstance(a, ast.Name) and a.id == hook.name
                                for a in n.value.args))
    fire = copy.deepcopy(next(n for n in main.body if isinstance(n, ast.FunctionDef)
                              and n.name == "_run_one_fire"))
    start = next(i for i, n in enumerate(fire.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "chain_tools" for t in n.targets))
    fire.body = fire.body[start:]
    outer = copy.deepcopy(next(n for n in ast.walk(main) if isinstance(n, ast.If)
                              and ast.unparse(n.test) ==
                              "_last_fire_result and _last_fire_result.get('executed')"))
    end = next(i for i, n in enumerate(outer.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                       and t.slice.value == "reflection_cycle" for t in n.targets))
    outer.body = outer.body[:end]
    module = ast.Module(body=[hook, registration, fire], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(MAIN_PATH), "exec"), env)
    return compile(ast.fix_missing_locations(ast.Module(body=[outer], type_ignores=[])),
                   str(MAIN_PATH), "exec")


class FakeProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, stages, completed=False):
        super().__init__(model="fake", api_key="")
        self.stages = iter(stages)
        self.completed = completed
        if completed:
            self.name = "claude_code"

    def supports_tool_use(self):
        return True

    def stream(self, request):
        calls = next(self.stages)
        if self.completed:
            records = []
            for call in calls:
                output, is_error = request.tool_executor(call.id, call.name, call.input)
                records.append(dict(tool_id=call.id, tool_name=call.name,
                                    tool_input=call.input, output=output, is_error=is_error))
            return AssistantMessage(tool_invocations=records)
        return AssistantMessage(tool_uses=calls)


def _call(tool, tool_id, score=80, **kwargs):
    return ToolUseBlock(id=tool_id, name=tool, input={"score": score, **kwargs})


def _exercise(stages, planned=None, completed=False, old_e=True):
    planned = planned or [calls[0].name for calls in stages]
    state = {
        "cycle_id": 10, "session_id": "test", "energy": 50.0, "entropy": 0.8,
        "pressure": 20.0, "last_e1": 0.11, "last_e2": 0.22,
        "last_e3": 0.33, "last_e4": 0.44, "last_prediction_error": 17,
        "unresponded_external_count": 2, "unresolved_external": 0.7,
        "raw_events": [], "pending": [],
        "predictor_confidence": {"A": {"e2_conf": 0.6, "ec_conf": 0.6}},
    }
    if old_e:
        state["e_values"] = {**{k: "99%" for k in ("e1", "e2", "e2_raw", "e3", "e4")},
                             "eff": 0.99}
    before = copy.deepcopy(state)
    hooks = HookRunner()
    hooks.register_pre(lambda n, a: HookRunResult.deny() if a.get("deny")
                       else HookRunResult.allow())
    registry = ToolRegistry()
    names = {c.name for calls in stages for c in calls} | set(planned)

    def handler(args):
        if args.get("fail"):
            raise RuntimeError("tool failed")
        return f"result {args['score']}"

    for name in names:
        registry.register(ToolSpec(name, "", {"type": "object"}, PermissionMode.READ_ONLY, handler))

    def evaluation(name, args, output):
        if args.get("eval_fail") == "before":
            raise RuntimeError("evaluation failed before mutation")
        score = args["score"]
        state["e_values"] = {**{k: f"{score}%" for k in ("e1", "e2", "e2_raw", "e3", "e4")},
                             "eff": score / 100}
        if args.get("eval_fail") == "after":
            raise RuntimeError("evaluation failed after mutation")
        return HookRunResult.allow()

    runtime = ConversationRuntime(FakeProvider(stages, completed), registry, hooks,
                                  PermissionEnforcer(PermissionMode.ALLOW))
    env = dict(
        copy=copy, json=json, re=re, datetime=datetime, state=state,
        _hook_runner=hooks, _runtime=runtime, _rt_registry=registry,
        _hook_ctx={"state_before": copy.deepcopy(state), "evaluations": []},
        _refresh_state=Mock(), _base_post_hook=evaluation, save_state=Mock(),
        selected={"tool": planned[0], "tools": planned, "reason": "test",
                  "chain": [{"tool": n, "predicted_e2": 90 - i * 20,
                             "predicted_ec": 0.9 - i * 0.2} for i, n in enumerate(planned)],
                  "_predicted_e2": 90, "_predicted_ec": 0.9},
        ctrl={"allowed_tools": names}, TOOLS={n: {} for n in names},
        assemble_system_prompt=Mock(return_value="test"), append_debug_log=Mock(),
        _extract_action_key=lambda n, a: n, cap_tool_result=lambda s: s,
        broadcast_log=Mock(), calc_state_change_bonus=Mock(return_value=0.0),
        _update_energy=Mock(wraps=_update_energy), propose_resp="", lv_msg="",
        _get_channel=lambda n: "self", make_perspective=lambda **kw: kw,
        event_emitter=SimpleNamespace(fire_event=lambda s, e: s["raw_events"].append(e)),
        maybe_compress_log=Mock(), pending_prune=Mock(), llm_cfg={}, load_pref=lambda: {},
        pressure=20.0, pp={"post_fire_reset": 0.3}, main=lambda: None,
        apply_negentropy=Mock(wraps=apply_negentropy), broadcast_e_values=Mock(),
        broadcast_state=Mock(),
    )
    post_code = _compile_main_parts(env)
    with ExitStack() as stack:
        for path in ("core.world_model.update_basin_state",
                     "core.transfer_entropy.maybe_te_maintenance",
                     "core.memory_links.prune_weak_links"):
            stack.enter_context(patch(path))
        metrics = stack.enter_context(patch("core.metrics.emit_cycle_metrics"))
        result = env["_run_one_fire"]("test", False, env["pp"], 10, datetime.now())
        env["_last_fire_result"] = result
        exec(post_code, env)
    return state, before, env, metrics


def _entries(state):
    return state["raw_events"][-1]["per_tool"]


def test_two_successes_in_one_turn():
    """1回の run_turn の A→B に最終 B の E を共用する誤実装を検出する。"""
    state, _, _, _ = _exercise([[_call("A", "a0", 20)],
                                [_call("A", "a1", 40), _call("B", "b1", 80)]], ["A", "A"])
    assert [(p.get("tool_id"), p["e2"]) for p in _entries(state)] == [
        ("a0", "20%"), ("a1", "40%"), ("b1", "80%")]
    assert state["raw_events"][-1]["e2"] == "80%"


def test_stage_prediction_mapping():
    """追加 B の予測対応・第1段 B の欠落・集約ログを全実行に広げる誤実装を検出する。"""
    state, _, _, _ = _exercise([[_call("A", "a", 20), _call("B", "extra", 40)],
                                [_call("B", "b", 80)]], ["A", "B"])
    a, extra, b = _entries(state)
    assert [(p["tool_id"], p["e2"]) for p in (a, extra, b)] == [
        ("a", "20%"), ("extra", "40%"), ("b", "80%")]
    assert a["prediction_error"] == 70
    assert "prediction_error" not in extra and "predicted_e2" not in extra
    assert b["prediction_error"] == 10 and b["predicted_e2"] == 70
    assert state["prediction_error_history_e2"] == [70, 10]
    assert state["raw_events"][-1]["prediction_error"] == 10
    assert state["raw_events"][-1]["tool"] == "B+B"
    assert state["raw_events"][-1]["result"] == "[B]\nresult 40\n---\n[B]\nresult 80"


def test_success_then_failure():
    """例外の実行に直前の E を付け、cycle の代表値を失敗で上書きする誤実装を検出する。"""
    state, _, env, _ = _exercise([[_call("A", "ok", 80)], [_call("B", "fail", fail=True)]])
    ok, failed = _entries(state)
    assert ok["e2"] == "80%" and failed["e2"] is None
    assert failed["is_error"] and "prediction_error" not in failed
    assert state["raw_events"][-1]["e2"] == "80%"
    assert env["_update_energy"].call_args.args[1:] == ("80%", "80%", "80%")


def test_repeated_names_and_rejections():
    """同名の複数実行と拒否の混在を名前や成功順だけで紐付ける誤実装を検出する。"""
    state, _, _, _ = _exercise([[_call("A", "denied", deny=True),
                                _call("A", "a", 20), _call("A", "b", 80)]])
    denied, a, b = _entries(state)
    assert [(p["tool_id"], p["e2"]) for p in (denied, a, b)] == [
        ("denied", None), ("a", "20%"), ("b", "80%")]
    assert all("prediction_error" not in p for p in (denied, a, b))
    assert state["last_prediction_error"] == 17


def test_evaluation_failure_is_missing():
    """採点前/途中の例外で古い E や途中の E を使う、成功toolを失敗扱いする誤実装を検出する。"""
    for when in ("before", "after"):
        state, before, env, _ = _exercise([[_call("output_display", "reply", eval_fail=when)]])
        entry, = _entries(state)
        assert entry["e2"] is None and not entry["is_error"]
        assert "prediction_error" not in entry
        assert state["unresponded_external_count"] == 1
        for key in ("energy", "entropy", "last_e1", "last_e2", "last_e3", "last_e4",
                    "last_prediction_error", "predictor_confidence"):
            assert state[key] == before[key], key
        assert not env["_update_energy"].called and not env["apply_negentropy"].called


def test_failed_head_is_not_replaced():
    """先頭失敗後の成功を繰り上げたり、後段の E と chain[0] の予測を比較する誤実装を検出する。"""
    state, _, _, _ = _exercise([[_call("A", "fail", fail=True), _call("A", "extra", 40)],
                                [_call("B", "b", 80)]], ["A", "B"])
    failed, extra, b = _entries(state)
    assert failed["e2"] is None and extra["e2"] == "40%"
    assert "prediction_error" not in failed and "prediction_error" not in extra
    assert b["prediction_error"] == 10
    assert state["last_prediction_error"] == 10
    assert state["prediction_error_history_e2"] == [10]


def test_unplanned_tool_has_no_prediction():
    """先頭別名の tool やその後の予定名 tool に予測を付ける誤実装を検出する。"""
    state, _, _, _ = _exercise([[_call("B", "other", 80), _call("A", "extra", 60)]], ["A"])
    assert [p["e2"] for p in _entries(state)] == ["80%", "60%"]
    assert all("prediction_error" not in p for p in _entries(state))
    assert state["last_prediction_error"] == 17
    assert "prediction_error" not in state["raw_events"][-1]


def test_all_failed_keeps_learning_but_advances_cycle():
    """全失敗時の古い E/空文字→0.5 更新と、欠測による cycle/pressure/記録停止を検出する。"""
    for old_e in (True, False):
        state, before, env, metrics = _exercise(
            [[_call("output_display", "fail", fail=True)], [_call("B", "deny", deny=True)]],
            old_e=old_e)
        for key in ("energy", "entropy", "last_e1", "last_e2", "last_e3", "last_e4",
                    "last_prediction_error", "predictor_confidence", "unresponded_external_count",
                    "unresolved_external"):
            assert state[key] == before[key], key
        assert all(p["e2"] is None for p in _entries(state))
        assert "e2" not in state["raw_events"][-1]
        assert not env["_update_energy"].called and not env["apply_negentropy"].called
        assert state["cycle_id"] == 11 and state["pressure"] == 6.0
        assert state["raw_events"][-1]["id"] == "test_0011"
        assert state["raw_events"][-1]["tool"] == "output_display+B"
        assert metrics.call_count == 1 and env["pending_prune"].call_count == 1


def test_successful_output_reduces_unresponded():
    """output_display が段の末尾でないと応答成功を数えない誤実装を検出する。"""
    state, _, env, _ = _exercise([[_call("output_display", "reply", 80),
                                  _call("B", "fail", fail=True)]])
    assert state["unresponded_external_count"] == 1
    assert env["apply_negentropy"].call_count == 1
    assert state["last_e2"] == 0.8 and state["energy"] > 50 and state["entropy"] < 0.8


def test_last_success_without_score_does_not_reuse_previous():
    """最後の成功が採点未完了なのに、前の成功の E を cycle 代表として使う誤実装を検出する。"""
    state, before, env, _ = _exercise([[_call("A", "scored", 80),
                                       _call("B", "unscored", eval_fail="before")]])
    scored, unscored = _entries(state)
    assert scored["e2"] == "80%" and unscored["e2"] is None
    assert not unscored["is_error"] and "e2" not in state["raw_events"][-1]
    assert state["energy"] == before["energy"] and not env["apply_negentropy"].called


def test_ambiguous_or_missing_ids_are_missing():
    """重複/欠落IDを任意の採点に対応させる誤実装を検出する。"""
    for calls in ([_call("A", "duplicate", 20), _call("A", "duplicate", 80)],
                  [_call("A", "duplicate", 20), _call("B", "duplicate", 80)],
                  [_call("A", "", 80)]):
        state, _, env, _ = _exercise([calls])
        assert all(p["e2"] is None for p in _entries(state))
        assert not env["_update_energy"].called


def test_unscored_head_is_not_replaced():
    """先頭の採点失敗時に後続の同名成功を予測対象へ繰り上げる誤実装を検出する。"""
    state, _, _, _ = _exercise([[_call("A", "head", eval_fail="after"),
                                _call("A", "extra", 60)]])
    head, extra = _entries(state)
    assert head["e2"] is None and not head["is_error"]
    assert extra["e2"] == "60%"
    assert all("prediction_error" not in p for p in (head, extra))
    assert state["last_prediction_error"] == 17


def test_post_hook_id_opt_in_compatibility():
    """既存3引数handlerを壊す、opt-inにIDを渡さない、ID省略で前回値を使う誤実装を検出する。"""
    hooks = HookRunner()
    legacy = Mock(return_value=HookRunResult.allow())
    identified = Mock(return_value=HookRunResult.allow())
    hooks.register_post(legacy)
    hooks.register_post(identified, with_tool_id=True)
    args = {"content": "test"}
    hooks.run_post_tool_use("A", args, "result", tool_id="a")
    legacy.assert_called_once_with("A", args, "result")
    identified.assert_called_once_with("A", args, "result", "a")
    hooks.run_post_tool_use("B", args, "result")
    identified.assert_called_with("B", args, "result", None)


def test_provider_completed_path_and_stage_id_reuse():
    """provider完結経路でIDを失う、前段の同じIDの採点を流用する誤実装を検出する。"""
    state, _, _, _ = _exercise([[_call("A", "same", 20)], [_call("A", "same", 80)]],
                                completed=True)
    assert [(p["chain_position"], p["e2"]) for p in _entries(state)] == [(0, "20%"), (1, "80%")]
    assert state["prediction_error_history_e2"] == [70, 10]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"[OK] {test.__name__}")
        except Exception:
            print(f"[FAIL] {test.__name__}")
            traceback.print_exc()
    print(f"{passed}/{len(tests)} groups passed")
    sys.exit(0 if passed == len(tests) else 1)
