"""テスト用の minimal MCP server (stdio + JSON-RPC 2.0)。

- initialize → 空 capabilities + serverInfo
- tools/list → 2 tool を返す
- tools/call → echo する
- resources/list → 1 resource を返す
- resources/read → text content を返す
- notifications/* は無視
"""
import json
import sys


CONTENT_LENGTH_SEP = b"\r\n\r\n"


def _read_message():
    """stdin から 1 message を読む。"""
    # header 読む
    header = b""
    while True:
        ch = sys.stdin.buffer.read(1)
        if not ch:
            return None
        header += ch
        if header.endswith(CONTENT_LENGTH_SEP):
            break
    # parse Content-Length
    content_length = 0
    for line in header.decode("ascii", errors="replace").split("\r\n"):
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":", 1)[1].strip())
    body = sys.stdin.buffer.read(content_length)
    return json.loads(body.decode("utf-8", errors="replace"))


def _send(msg):
    body = json.dumps(msg).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _response(req_id, result=None, error=None):
    r = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        r["error"] = error
    else:
        r["result"] = result or {}
    return r


def main():
    while True:
        msg = _read_message()
        if msg is None:
            break
        method = msg.get("method", "")
        req_id = msg.get("id")

        if req_id is None:
            # notification - 無視
            continue

        if method == "initialize":
            _send(_response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "mock-mcp", "version": "0.1"},
            }))
        elif method == "tools/list":
            _send(_response(req_id, {
                "tools": [
                    {"name": "echo", "description": "echo input",
                     "inputSchema": {"type": "object",
                                     "properties": {"text": {"type": "string"}}}},
                    {"name": "add", "description": "add two numbers",
                     "inputSchema": {"type": "object",
                                     "properties": {"a": {"type": "number"},
                                                    "b": {"type": "number"}}}},
                ],
            }))
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name", "")
            args = params.get("arguments") or {}
            if name == "echo":
                _send(_response(req_id, {
                    "content": [{"type": "text",
                                 "text": f"echo: {args.get('text', '')}"}],
                }))
            elif name == "add":
                a = args.get("a", 0)
                b = args.get("b", 0)
                _send(_response(req_id, {
                    "content": [{"type": "text",
                                 "text": f"{a} + {b} = {a + b}"}],
                }))
            else:
                _send(_response(req_id, error={
                    "code": -32601,
                    "message": f"tool not found: {name}",
                }))
        elif method == "resources/list":
            _send(_response(req_id, {
                "resources": [
                    {"uri": "mock://hello",
                     "name": "hello",
                     "description": "hello resource",
                     "mimeType": "text/plain"},
                ],
            }))
        elif method == "resources/read":
            params = msg.get("params") or {}
            uri = params.get("uri", "")
            if uri == "mock://hello":
                _send(_response(req_id, {
                    "contents": [{"uri": uri, "mimeType": "text/plain",
                                  "text": "Hello from mock MCP"}],
                }))
            else:
                _send(_response(req_id, error={
                    "code": -32602,
                    "message": f"unknown resource: {uri}",
                }))
        else:
            _send(_response(req_id, error={
                "code": -32601, "message": f"method not found: {method}",
            }))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.stderr.write(f"mock server error: {e}\n")
