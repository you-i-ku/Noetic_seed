"""C-1: 欠測の伝播と E 非依存処理の継続。python -B tests/test_missing_evaluation.py

外部 LLM・embedding・ログの保存先のみ隔離し、採点パーサ・hook・main は実コード。
識別する誤実装: 0.5補完、部分採点、0の棄却、古いEの再利用、早期return、
最後の採点成功への遡及、scored の truthiness 判定。
"""
import copy
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import eval as evaluation
from core.pending_unified import pending_add
from core.runtime.hooks import make_post_tool_use_evaluation
from test_failed_tool_learning import _call, _entries, _exercise


GOOD = "E1=70\nE2=20\nE3=80\nE4=90"
ARGS = {"tool_intent": "設定を確認する", "tool_expected_outcome": "確認できる"}


class MissingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        tmp = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch("core.config.RESOLUTION_LOG", Path(tmp) / "ledger.log"))
        self.stack.enter_context(patch("core.state.append_debug_log"))
        self.stack.enter_context(patch("core.eval.is_vector_ready", return_value=False))
        self.state = {"cycle_id": 10, "pending": [], "action_ledger": [],
                      "self": {}, "files_read": [], "files_written": []}
        self.before = copy.deepcopy(self.state)
        # 実際の state 差分で eff=0.5。eff=0 だと減衰を止めても見分けられない。
        self.state["files_read"] = ["settings.json"]

    def hook(self, response):
        return make_post_tool_use_evaluation(
            self.state, lambda: self.before, Mock(side_effect=[response]),
            lambda: 10, lambda: [],
        )

    def assert_missing(self, result):
        self.assertFalse(result.failed)
        self.assertFalse(result.denied)
        self.assertEqual(self.state["e_values"], {"scored": False, "eff": 0.5})
        self.assertIn("採点欠測", result.messages[0])

    def test_parser_contract(self):
        for response, expected in [
            (GOOD, {"e1": .7, "e2": .2, "e3": .8, "e4": .9}),
            ("E1=70\nE2=20\nE3=80", {"e1": .7, "e2": .2, "e3": .8}),
            ("70\n20\n80", {"e1": .7, "e2": .2, "e3": .8}),
            ("E1=70\nE2=20", None), ("bad response", None),
            (RuntimeError("offline"), None),
        ]:
            with self.subTest(response=response):
                self.assertEqual(evaluation.eval_with_llm(
                    "intent", "expect", "result", [], Mock(side_effect=[response])), expected)

    def test_each_missing_key_and_invalid_number(self):
        good = dict(e1=.7, e2=.2, e3=.8, e4=.9)
        cases = [None, {}]
        for key in good:
            cases.append({k: v for k, v in good.items() if k != key})
            for value in (-.01, 1.01, float("nan"), float("inf"),
                          -float("inf"), None, "0.7", True, False):
                cases.append({**good, key: value})
        for scores in cases:
            with self.subTest(scores=scores), patch(
                    "core.eval.eval_with_llm", return_value=scores):
                self.state["e_values"] = {"scored": True, "e1": "77%", "e3": "77%"}
                self.assert_missing(self.hook(GOOD)("bash", ARGS, "result"))

    def test_zero_and_one_are_valid(self):
        for value in (0, 100):
            with self.subTest(value=value):
                response = "\n".join(f"E{i}={value}" for i in range(1, 5))
                result = self.hook(response)("bash", ARGS, "result")
                ev = self.state["e_values"]
                self.assertFalse(result.failed)
                self.assertIs(ev["scored"], True)
                self.assertEqual([ev[k] for k in ("e1", "e2_raw", "e3", "e4")],
                                 [f"{value}%"] * 4)
                self.assertEqual(ev["e2"], "0%" if value == 0 else "65%")

    def test_previous_scored_is_replaced(self):
        self.hook(GOOD)("bash", ARGS, "first")
        self.assertIs(self.state["e_values"]["scored"], True)
        self.assert_missing(self.hook("bad response")("bash", ARGS, "second"))
        self.assertEqual([p["result_snippet"] for p in self.state["action_ledger"]],
                         ["first", "second"])

    def test_missing_skips_e2_and_e3_but_keeps_ledger(self):
        for response in ("bad response", "E1=70\nE2=20\nE4=90",
                         "E1=101\nE2=20\nE3=80\nE4=90", RuntimeError("offline")):
            with self.subTest(response=response), patch(
                    "core.eval.apply_effective_change_to_e2", wraps=evaluation.apply_effective_change_to_e2
            ) as cap, patch("core.eval.update_unresolved_intents",
                            wraps=evaluation.update_unresolved_intents) as unresolved:
                count = len(self.state["action_ledger"])
                self.assert_missing(self.hook(response)("bash", ARGS, "confirmed"))
                cap.assert_not_called()
                unresolved.assert_not_called()
                self.assertEqual(self.state["pending"], [])
                self.assertEqual(len(self.state["action_ledger"]), count + 1)
                last = self.state["action_ledger"][-1]
                self.assertEqual((last["tool"], last["intent"], last["ec"], last["result_snippet"]),
                                 ("bash", ARGS["tool_intent"], .5, "confirmed"))

    def test_missing_still_decays_relevant_pending(self):
        relevant = pending_add(self.state, "old", "result", "cycles", "related",
                               cycle_id=1, initial_gap=.8, semantic_merge=True)
        unrelated = pending_add(self.state, "other", "other", "cycles", "unrelated",
                                 cycle_id=1, initial_gap=.6, semantic_merge=True)
        with patch("core.eval.is_vector_ready", return_value=True), patch(
                "core.eval._embed_sync", return_value=[[1, 0], [1, 0], [0, 1]]):
            self.assert_missing(self.hook("bad response")("bash", ARGS, "result"))
        self.assertAlmostEqual(relevant["gap"], .6)
        self.assertEqual(unrelated["gap"], .6)
        self.assertEqual(len(self.state["pending"]), 2)

    def test_missing_still_observes_matching_pending(self):
        matching = pending_add(self.state, "old", "result", "cycles", "wait",
                               cycle_id=1, match_pattern={"source_action": "bash"})
        other = pending_add(self.state, "other", "result", "cycles", "wait",
                            cycle_id=1, match_pattern={"source_action": "read_file"})
        self.assert_missing(self.hook("bad response")("bash", ARGS, "arrived"))
        self.assertEqual(matching["observed_content"], "arrived")
        self.assertIsNone(other["observed_content"])

    def test_real_parser_hook_main_missing(self):
        for response in ("bad response", "E1=70\nE2=20\nE4=90", RuntimeError("offline")):
            with self.subTest(response=response):
                state, before, env, _ = _exercise(
                    [[_call("A", "a", **ARGS)]], llm_responses=[response])
                entry, = _entries(state)
                self.assertFalse(entry["is_error"])
                self.assertIsNone(entry["e2"])
                self.assertIsNone(entry["eff"])
                self.assertNotIn("prediction_error", entry)
                self.assertEqual(env["_hook_ctx"]["evaluations"], [])
                self.assertEqual(state["e_values"], {"scored": False, "eff": 0.0})
                for key in ("energy", "last_e1", "last_e2", "last_e3", "last_e4",
                            "last_prediction_error", "predictor_confidence", "pending"):
                    self.assertEqual(state[key], before[key], key)
                env["_update_energy"].assert_not_called()
                env["apply_negentropy"].assert_not_called()
                env["broadcast_e_values"].assert_not_called()
                self.assertEqual(len(state["action_ledger"]), 1)
                self.assertEqual(state["action_ledger"][0]["tool"], "A")
                env["observe"].assert_called_once()
                self.assertEqual(state["cycle_id"], 11)
                self.assertEqual(state["pressure"], 6.0)

    def test_mixed_scores_use_last_execution_success(self):
        for missing_first, failed_last in ((False, False), (True, False), (False, True)):
            with self.subTest(missing_first=missing_first, failed_last=failed_last):
                responses = ["bad response", GOOD] if missing_first else [GOOD, "bad response"]
                state, before, env, _ = _exercise(
                    [[_call("A", "a", **ARGS)], [_call("B", "b", fail=failed_last, **ARGS)]],
                    llm_responses=responses)
                a, b = _entries(state)
                self.assertEqual((a["e2"], b["e2"]),
                                 (None, "6%") if missing_first else ("6%", None))
                self.assertEqual(b["is_error"], failed_last)
                learned = b if missing_first else a
                # 実hookでは eff=0 → E2=20% * 0.3 = 6%。予測はA=90/B=70。
                self.assertEqual(learned["prediction_error"], 64 if missing_first else 84)
                self.assertEqual(state["prediction_error_history_e2"],
                                 [64] if missing_first else [84])
                self.assertNotEqual(state["predictor_confidence"], before["predictor_confidence"])
                if not missing_first and not failed_last:
                    self.assertNotIn("e2", state["raw_events"][-1])
                    self.assertEqual(state["energy"], before["energy"])
                    env["_update_energy"].assert_not_called()
                    env["apply_negentropy"].assert_not_called()
                else:
                    self.assertEqual(state["raw_events"][-1]["e2"], "6%")
                    env["_update_energy"].assert_called_once()
                    env["apply_negentropy"].assert_called_once()
                    self.assertNotEqual(state["energy"], before["energy"])
                    self.assertEqual(state["last_e2"], .06)

    def test_main_requires_literal_true(self):
        # 数値1や文字列Trueもtruthy。is True条件を消す変異を検出する。
        for scored in (False, None, 1, "True"):
            with self.subTest(scored=scored):
                state, _, env, _ = _exercise([[_call("A", "a", scored=scored)]])
                self.assertIsNone(_entries(state)[0]["e2"])
                self.assertEqual(env["_hook_ctx"]["evaluations"], [])
                env["_update_energy"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
