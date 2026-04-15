"""MCP protocol + manager + bridge 統合テスト。

mock MCP server (tests/_mock_mcp_server.py) を subprocess で起動して
end-to-end の JSON-RPC 往復を検証。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.mcp.naming import (
    normalize_name_for_mcp, mcp_tool_prefix, mcp_tool_name,
    parse_mcp_tool_name,
)
from core.runtime.mcp.protocol import (
    JsonRpcRequest, JsonRpcResponse, JsonRpcError,
    encode_message, parse_message, parse_all_messages,
    PROTOCOL_VERSION, JSONRPC_VERSION,
)
from core.runtime.mcp.client import (
    TransportType, BaseTransport, RemoteTransportConfig,
    SseTransport, HttpTransport, WebSocketTransport,
)
from core.runtime.mcp.stdio_transport import StdioTransport
from core.runtime.mcp.manager import (
    McpServerManager, ConnectionStatus,
)
from core.runtime.mcp.bridge import McpToolBridge
from core.runtime.registry import ToolRegistry


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


MOCK_SERVER_PATH = Path(__file__).resolve().parent / "_mock_mcp_server.py"
VENV_PY = "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe"


# ============================================================
# Naming
# ============================================================

def test_normalize_name():
    print("== naming: normalize ==")
    return all([
        _assert(normalize_name_for_mcp("abc") == "abc", "英数字"),
        _assert(normalize_name_for_mcp("slack-server") == "slack_server",
                "ハイフン → _"),
        _assert(normalize_name_for_mcp("a.b.c") == "a_b_c", "ドット"),
        _assert(normalize_name_for_mcp("x_y") == "x_y", "_ は保持"),
    ])


def test_mcp_tool_name():
    print("== naming: mcp_tool_name ==")
    return all([
        _assert(mcp_tool_name("slack", "post") == "mcp__slack__post",
                "シンプル"),
        _assert(mcp_tool_name("slack-bot", "post-msg") == "mcp__slack_bot__post_msg",
                "ハイフン"),
        _assert(mcp_tool_prefix("github") == "mcp__github__",
                "prefix"),
    ])


def test_parse_mcp_tool_name():
    print("== naming: parse_mcp_tool_name ==")
    s, t = parse_mcp_tool_name("mcp__slack__post_msg")
    return all([
        _assert(s == "slack" and t == "post_msg", "分解"),
        _assert(parse_mcp_tool_name("read_file") == (None, "read_file"),
                "非 MCP は None"),
    ])


# ============================================================
# Protocol framing
# ============================================================

def test_encode_decode_roundtrip():
    print("== protocol: encode + parse roundtrip ==")
    req = JsonRpcRequest(method="tools/list",
                         params={"x": 1}, id=42)
    raw = encode_message(req)
    msg, rest = parse_message(raw)
    return all([
        _assert(rest == b"", "remainder 空"),
        _assert(msg["jsonrpc"] == "2.0", "jsonrpc 2.0"),
        _assert(msg["method"] == "tools/list", "method"),
        _assert(msg["id"] == 42, "id"),
        _assert(msg["params"] == {"x": 1}, "params"),
    ])


def test_encode_response():
    print("== protocol: response encode ==")
    resp = JsonRpcResponse(id=1, result={"ok": True})
    raw = encode_message(resp)
    msg, _ = parse_message(raw)
    return all([
        _assert(msg["result"] == {"ok": True}, "result"),
        _assert(msg["id"] == 1, "id"),
    ])


def test_encode_error_response():
    print("== protocol: error response ==")
    resp = JsonRpcResponse(id=1, error=JsonRpcError(code=-32601,
                                                     message="not found"))
    raw = encode_message(resp)
    msg, _ = parse_message(raw)
    return all([
        _assert("error" in msg and msg["error"]["code"] == -32601,
                "error code"),
        _assert(msg["error"]["message"] == "not found", "error message"),
    ])


def test_parse_partial():
    print("== protocol: 部分受信で None ==")
    msg, rest = parse_message(b"Content-Length: 100\r\n\r\nincomplete")
    return all([
        _assert(msg is None, "None"),
        _assert(len(rest) > 0, "buffer 保持"),
    ])


def test_parse_multiple():
    print("== protocol: 2 messages 連結 ==")
    r1 = encode_message(JsonRpcRequest(method="a", id=1))
    r2 = encode_message(JsonRpcRequest(method="b", id=2))
    msgs, rest = parse_all_messages(r1 + r2)
    return all([
        _assert(len(msgs) == 2, "2 件"),
        _assert(msgs[0]["method"] == "a" and msgs[1]["method"] == "b",
                "順序"),
        _assert(rest == b"", "余り無し"),
    ])


# ============================================================
# Transport (stub ones)
# ============================================================

def test_remote_transport_stubs_raise():
    print("== remote transports: start() で NotImplementedError ==")
    cfg = RemoteTransportConfig(url="https://x/")
    transports = [
        SseTransport(cfg),
        HttpTransport(cfg),
        WebSocketTransport(cfg),
    ]
    ok = True
    for t in transports:
        try:
            t.start()
            ok = False
        except NotImplementedError:
            pass
    return _assert(ok, "stub は NotImplementedError")


def test_base_transport_types():
    print("== transport types enum ==")
    return all([
        _assert(TransportType.STDIO.value == "stdio", "stdio"),
        _assert(TransportType.SSE.value == "sse", "sse"),
        _assert(TransportType.HTTP.value == "http", "http"),
        _assert(TransportType.WEBSOCKET.value == "websocket", "websocket"),
        _assert(TransportType.SDK.value == "sdk", "sdk"),
        _assert(TransportType.MANAGED_PROXY.value == "managed_proxy",
                "managed_proxy"),
    ])


# ============================================================
# Stdio transport + Manager (end-to-end)
# ============================================================

def test_stdio_initialize():
    print("== stdio + manager: initialize handshake ==")
    transport = StdioTransport(
        command=VENV_PY,
        args=[str(MOCK_SERVER_PATH)],
    )
    mgr = McpServerManager("mock", transport)
    try:
        ok = mgr.start(initialize_timeout=5.0)
        return all([
            _assert(ok, f"start 成功 (last_error={mgr.last_error})"),
            _assert(mgr.status == ConnectionStatus.CONNECTED, "connected"),
            _assert(mgr.server_info is not None
                    and mgr.server_info.get("name") == "mock-mcp",
                    "serverInfo"),
        ])
    finally:
        mgr.stop()


def test_stdio_discover_tools():
    print("== stdio + manager: tools/list ==")
    transport = StdioTransport(VENV_PY, [str(MOCK_SERVER_PATH)])
    mgr = McpServerManager("mock", transport)
    try:
        assert mgr.start(initialize_timeout=5.0)
        tools = mgr.discover_tools(timeout=5.0)
        names = {t.name for t in tools}
        return all([
            _assert(len(tools) == 2, f"2 tool (got {len(tools)})"),
            _assert("echo" in names, "echo"),
            _assert("add" in names, "add"),
            _assert(tools[0].input_schema is not None, "input_schema 含む"),
        ])
    finally:
        mgr.stop()


def test_stdio_call_tool():
    print("== stdio + manager: tools/call ==")
    transport = StdioTransport(VENV_PY, [str(MOCK_SERVER_PATH)])
    mgr = McpServerManager("mock", transport)
    try:
        assert mgr.start(initialize_timeout=5.0)
        r1 = mgr.call_tool("echo", {"text": "hello"}, timeout=5.0)
        r2 = mgr.call_tool("add", {"a": 2, "b": 3}, timeout=5.0)
        r3 = mgr.call_tool("nonexistent", {}, timeout=5.0)
        return all([
            _assert("result" in r1, "echo result"),
            _assert("echo: hello" in str(r1), "echo content"),
            _assert("2 + 3 = 5" in str(r2), "add content"),
            _assert("error" in r3, "unknown tool エラー"),
        ])
    finally:
        mgr.stop()


def test_stdio_read_resource():
    print("== stdio + manager: resources/list + read ==")
    transport = StdioTransport(VENV_PY, [str(MOCK_SERVER_PATH)])
    mgr = McpServerManager("mock", transport)
    try:
        assert mgr.start(initialize_timeout=5.0)
        resources = mgr.discover_resources(timeout=5.0)
        r = mgr.read_resource("mock://hello", timeout=5.0)
        return all([
            _assert(len(resources) == 1, "1 resource"),
            _assert(resources[0].uri == "mock://hello", "uri"),
            _assert("result" in r, "read result"),
            _assert("Hello from mock" in str(r), "content"),
        ])
    finally:
        mgr.stop()


# ============================================================
# Bridge (registry 連携)
# ============================================================

def test_bridge_add_and_call():
    print("== bridge: add_server + registry 登録 + call ==")
    reg = ToolRegistry()
    bridge = McpToolBridge(reg)
    transport = StdioTransport(VENV_PY, [str(MOCK_SERVER_PATH)])
    try:
        mgr = bridge.add_server("mock", transport,
                                 start=True, auto_discover=True)
        # registry に prefix 付きで登録される
        echo_full = "mcp__mock__echo"
        add_full = "mcp__mock__add"
        in_registry = reg.has(echo_full) and reg.has(add_full)
        # call via registry
        out_echo = reg.execute(echo_full, {"text": "hi"})
        out_add = reg.execute(add_full, {"a": 10, "b": 20})
        return all([
            _assert(mgr.status == ConnectionStatus.CONNECTED, "connected"),
            _assert(in_registry, "registry に登録"),
            _assert("echo: hi" in out_echo, "echo 実行結果"),
            _assert("10 + 20 = 30" in out_add, "add 実行結果"),
        ])
    finally:
        bridge.stop_all()


def test_bridge_call_by_full_name():
    print("== bridge: call_by_full_name ==")
    reg = ToolRegistry()
    bridge = McpToolBridge(reg)
    transport = StdioTransport(VENV_PY, [str(MOCK_SERVER_PATH)])
    try:
        bridge.add_server("mock", transport)
        r = bridge.call_by_full_name("mcp__mock__echo", {"text": "x"})
        r_bad = bridge.call_by_full_name("not_mcp", {})
        return all([
            _assert("result" in r, "正常 call"),
            _assert("error" in r_bad, "非 MCP 名 エラー"),
        ])
    finally:
        bridge.stop_all()


def test_bridge_health_snapshot():
    print("== bridge: health_snapshot ==")
    reg = ToolRegistry()
    bridge = McpToolBridge(reg)
    transport = StdioTransport(VENV_PY, [str(MOCK_SERVER_PATH)])
    try:
        bridge.add_server("mock", transport)
        snaps = bridge.health_snapshot()
        return all([
            _assert(len(snaps) == 1, "1 server"),
            _assert(snaps[0].name == "mock", "name"),
            _assert(snaps[0].status == "connected", "status"),
            _assert(snaps[0].tool_count == 2, "2 tool"),
        ])
    finally:
        bridge.stop_all()


def test_bridge_remove_server():
    print("== bridge: remove_server ==")
    reg = ToolRegistry()
    bridge = McpToolBridge(reg)
    transport = StdioTransport(VENV_PY, [str(MOCK_SERVER_PATH)])
    bridge.add_server("mock", transport)
    had = reg.has("mcp__mock__echo")
    ok = bridge.remove_server("mock")
    gone = not reg.has("mcp__mock__echo")
    bridge.stop_all()
    return all([
        _assert(had, "登録されていた"),
        _assert(ok, "remove 成功"),
        _assert(gone, "tool が消えた"),
    ])


def test_bridge_duplicate_name():
    print("== bridge: 同名 server 重複 で ValueError ==")
    reg = ToolRegistry()
    bridge = McpToolBridge(reg)
    t1 = StdioTransport(VENV_PY, [str(MOCK_SERVER_PATH)])
    bridge.add_server("mock", t1)
    try:
        t2 = StdioTransport(VENV_PY, [str(MOCK_SERVER_PATH)])
        try:
            bridge.add_server("mock", t2)
            t2.stop()
            return _assert(False, "ValueError が投げられるべき")
        except ValueError:
            return _assert(True, "ValueError")
    finally:
        bridge.stop_all()


# ============================================================
# main
# ============================================================

def main():
    tests = [
        test_normalize_name, test_mcp_tool_name, test_parse_mcp_tool_name,
        test_encode_decode_roundtrip, test_encode_response,
        test_encode_error_response, test_parse_partial, test_parse_multiple,
        test_remote_transport_stubs_raise, test_base_transport_types,
        test_stdio_initialize, test_stdio_discover_tools,
        test_stdio_call_tool, test_stdio_read_resource,
        test_bridge_add_and_call, test_bridge_call_by_full_name,
        test_bridge_health_snapshot, test_bridge_remove_server,
        test_bridge_duplicate_name,
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
