"""step 0a-1: 送信受付と到達確認を区別し、未接続は失敗経路に流す。"""
import json
import queue
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import ws_server
from core.pending_unified import pending_add_response_intent
from core.providers.base import AssistantMessage, BaseProvider, ToolUseBlock
from core.runtime.conversation import ConversationRuntime
from core.runtime.hooks import HookRunner, HookRunResult, make_post_tool_use_evaluation
from core.runtime.permissions import PermissionEnforcer, PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec
from tools.ui_tools import _output_display


def _assert(cond, label):
    print(f"  [{'OK ' if cond else 'FAIL'}] {label}")
    return cond


def test_no_clients_raises():
    """接続 0 でも送信完了やエラー文字列を正常返却する誤実装を検出する。"""
    with patch.object(ws_server, "_ws_clients", set()), \
            patch.object(ws_server, "_send_queue", queue.Queue()) as outgoing:
        try:
            _output_display({"channel": "device", "content": "こんにちは"})
        except RuntimeError as exc:
            return all([
                _assert("0 件" in str(exc) and "channel=device" in str(exc),
                        "未接続の理由と channel が例外に含まれる"),
                _assert(outgoing.empty(), "未接続ではキューに積まない"),
            ])
    return _assert(False, "接続 0 は RuntimeError")


def test_two_clients_queued():
    """接続数欠落・到達済み扱い・content 切り詰め・channel 欠落を検出する。"""
    content = "こんにちは\n" * 30 + "末尾"
    with patch.object(ws_server, "_ws_clients", {object(), object()}), \
            patch.object(ws_server, "_send_queue", queue.Queue()) as outgoing, \
            patch("tools.ui_tools.broadcast_log"):
        result = _output_display({"channel": " device ", "content": content})
        messages = list(outgoing.queue)
    return all([
        _assert(result == ("送信キューに登録 (channel=device、受付時の接続 2 件、"
                           f"実際に届いたかは未確認): {content}"), "受付数・未確認・全文"),
        _assert([json.loads(m) for m in messages] == [
            {"type": "reply", "content": content, "channel": "device"},
        ], "reply の content と channel をそのまま登録"),
    ])


def test_broadcast_counts():
    """None/固定値を返す、0 件で積む、複数接続分を重複登録する誤実装を検出する。"""
    results = []
    for count in (0, 1, 3):
        msg = {"type": "reply", "content": f"接続 {count}", "channel": "device"}
        with patch.object(ws_server, "_ws_clients", {object() for _ in range(count)}), \
                patch.object(ws_server, "_send_queue", queue.Queue()) as outgoing:
            result = ws_server.broadcast(msg)
            messages = [json.loads(m) for m in list(outgoing.queue)]
        results.extend([
            _assert(type(result) is int and result == count, f"接続 {count} の整数値"),
            _assert(messages == ([msg] if count else []), f"接続 {count} のキュー内容"),
        ])
    return all(results)


def test_empty_arguments():
    """空 content/channel の既存エラー文変更や、検証前の送信を検出する。"""
    cases = [
        ({"channel": "device"}, "エラー: contentを指定してください"),
        ({"channel": "device", "content": ""}, "エラー: contentを指定してください"),
        ({"content": "hi"}, "エラー: channel を指定してください "
         "(WM.channels を観察して利用可能な channel を確認)"),
        ({"content": "hi", "channel": "  "}, "エラー: channel を指定してください "
         "(WM.channels を観察して利用可能な channel を確認)"),
    ]
    with patch("tools.ui_tools.broadcast") as send, \
            patch("tools.ui_tools.broadcast_log") as log:
        results = [_assert(_output_display(args) == expected, repr(args))
                   for args, expected in cases]
        results.append(_assert(not send.called and not log.called, "入力エラー時は送信なし"))
    return all(results)


class MockProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, args):
        super().__init__(model="mock", api_key="")
        self.args = args

    def supports_tool_use(self):
        return True

    def stream(self, req):
        return AssistantMessage(tool_uses=[
            ToolUseBlock(id="reply-1", name="output_display", input=self.args),
        ])


def test_runtime_failure_preserves_pending():
    """未接続を成功扱いして応答意図を消化する誤実装を、接続ありの対照付きで検出する。"""
    results = []
    for count in (0, 1):
        state = {"pending": [], "cycle_id": 10, "action_ledger": []}
        pending = pending_add_response_intent(state, "device", "返事をください", 10)
        other = pending_add_response_intent(state, "claude", "別の応答待ち", 10)
        before = deepcopy(state)
        args = {"channel": "device", "content": "こんにちは"}
        hooks = HookRunner()
        evaluation = Mock(wraps=make_post_tool_use_evaluation(
            state, lambda: before, lambda *a, **k: "", lambda: 10, lambda: [],
        ))
        failure = Mock(return_value=HookRunResult.allow())
        hooks.register_post(evaluation)
        hooks.register_failure(failure)
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="output_display", description="", input_schema={"type": "object"},
            required_permission=PermissionMode.READ_ONLY, handler=_output_display,
        ))
        runtime = ConversationRuntime(
            provider=MockProvider(args), tool_registry=registry, hook_runner=hooks,
            permission_enforcer=PermissionEnforcer(PermissionMode.ALLOW),
        )
        with patch.object(ws_server, "_ws_clients", {object() for _ in range(count)}), \
                patch.object(ws_server, "_send_queue", queue.Queue()), \
                patch("tools.ui_tools.broadcast_log"), \
                patch("core.eval.eval_with_llm", return_value={"e3": 1.0}), \
                patch("core.eval.calc_effective_change", return_value=0.5), \
                patch("core.eval.update_unresolved_intents"), \
                patch("core.eval.update_gaps_by_relevance"):
            summary = runtime.run_turn("応答してください")
        rec, = summary.tool_invocations
        results.append(_assert(
            (rec.tool_id, rec.tool_name, rec.tool_input) == ("reply-1", "output_display", args),
            f"接続 {count}: 対象の実行 id・名前・引数"))
        if count == 0:
            results.extend([
                _assert(rec.is_error and "0 件" in rec.output and "channel=device" in rec.output,
                        "未接続は理由付き is_error=True"),
                _assert(state["pending"] == before["pending"], "pending の id と全値が不変"),
                _assert(evaluation.call_count == 0 and failure.call_count == 1,
                        "成功 hook を通らず失敗 hook のみ呼ぶ"),
            ])
        else:
            target = next(p for p in state["pending"] if p["id"] == pending["id"])
            untouched = next(p for p in state["pending"] if p["id"] == other["id"])
            results.extend([
                _assert(not rec.is_error and evaluation.call_count == 1 and not failure.called,
                        "対照: 接続ありは成功 hook を呼ぶ"),
                _assert(target["observed_content"] == rec.output, "対照: 対象 id に出力を観測"),
                _assert(untouched == before["pending"][1], "別 channel の id と全値は不変"),
            ])
    return all(results)


if __name__ == "__main__":
    tests = [test_no_clients_raises, test_two_clients_queued, test_broadcast_counts,
             test_empty_arguments, test_runtime_failure_preserves_pending]
    results = []
    for test in tests:
        print(f"\n== {test.__name__} ==")
        results.append(test())
    print(f"\n{sum(results)}/{len(results)} groups passed")
    sys.exit(0 if all(results) else 1)
