"""tools/mcp.py と core/runtime/mcp/bridge.py の統合テスト。

tools/mcp.py の MCP / ListMcpResources / ReadMcpResource は
attach_real_bridge(McpToolBridge) されれば実 MCP server と通信する。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.mcp.bridge import McpToolBridge
from core.runtime.mcp.stdio_transport import StdioTransport
from core.runtime.registry import ToolRegistry
from core.runtime.tools import mcp as mcp_tool


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


MOCK_SERVER_PATH = Path(__file__).resolve().parent / "_mock_mcp_server.py"
VENV_PY = "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe"


def test_tool_pending_without_bridge():
    print("== MCP tool: bridge 未接続で pending ==")
    mcp_tool.detach_real_bridge()
    # 内部 _bridge dict を直接リセット (callable も real も無し)
    for k in mcp_tool._bridge:
        mcp_tool._bridge[k] = None
    reg = ToolRegistry()
    mcp_tool.register(reg)
    out = reg.execute("MCP", {"server": "x", "tool": "y", "arguments": {}})
    return _assert("pending" in out, "pending")


def test_tool_with_real_bridge():
    print("== MCP tool: attach_real_bridge で実 server 呼出 ==")
    reg = ToolRegistry()
    mcp_tool.register(reg)

    bridge = McpToolBridge(reg)
    transport = StdioTransport(VENV_PY, [str(MOCK_SERVER_PATH)])
    try:
        bridge.add_server("mock", transport,
                           start=True, auto_discover=True)
        mcp_tool.attach_real_bridge(bridge)

        # (A) MCP tool で echo を呼ぶ
        out_call = reg.execute("MCP", {
            "server": "mock", "tool": "echo",
            "arguments": {"text": "real"},
        })
        # (B) ListMcpResources
        out_list = reg.execute("ListMcpResources", {"server": "mock"})
        # (C) ReadMcpResource
        out_read = reg.execute("ReadMcpResource", {
            "server": "mock", "uri": "mock://hello",
        })
        return all([
            _assert("echo: real" in out_call, "実 server の結果"),
            _assert("mock://hello" in out_list, "resource list"),
            _assert("Hello from mock" in out_read, "resource read"),
        ])
    finally:
        mcp_tool.detach_real_bridge()
        bridge.stop_all()


def test_tool_uses_prefixed_registry_entry():
    print("== add_server で registry に prefix 付き tool が追加 → 直接呼出し可 ==")
    reg = ToolRegistry()
    mcp_tool.register(reg)  # MCP generic tool 群
    bridge = McpToolBridge(reg)
    transport = StdioTransport(VENV_PY, [str(MOCK_SERVER_PATH)])
    try:
        bridge.add_server("mock2", transport)
        mcp_tool.attach_real_bridge(bridge)
        # prefix 付き直接呼出
        out = reg.execute("mcp__mock2__add", {"a": 5, "b": 7})
        return _assert("5 + 7 = 12" in out, "直接呼出し")
    finally:
        mcp_tool.detach_real_bridge()
        bridge.stop_all()


def test_real_bridge_precedence():
    print("== real_bridge > callable_bridge の優先順 ==")
    reg = ToolRegistry()
    mcp_tool.register(reg)
    # callable bridge を先に注入
    mcp_tool.set_mcp_bridge(
        call_tool=lambda s, t, a: f"[callable] {s}:{t}",
    )
    bridge = McpToolBridge(reg)
    transport = StdioTransport(VENV_PY, [str(MOCK_SERVER_PATH)])
    try:
        bridge.add_server("mock3", transport)
        mcp_tool.attach_real_bridge(bridge)
        out = reg.execute("MCP", {"server": "mock3", "tool": "echo",
                                   "arguments": {"text": "hi"}})
        # real が優先されるので "[callable]" は出ない
        return all([
            _assert("[callable]" not in out, "callable を使わない"),
            _assert("echo: hi" in out, "real server 結果"),
        ])
    finally:
        mcp_tool.detach_real_bridge()
        mcp_tool.set_mcp_bridge(call_tool=lambda *a, **kw: None)
        bridge.stop_all()


def main():
    tests = [
        test_tool_pending_without_bridge,
        test_tool_with_real_bridge,
        test_tool_uses_prefixed_registry_entry,
        test_real_bridge_precedence,
    ]
    print(f"Running {len(tests)} test groups...\n")
    passed = 0
    for t in tests:
        if t():
            passed += 1
        print()
    print(f"=== Result: {passed}/{len(tests)} test groups passed ===")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
