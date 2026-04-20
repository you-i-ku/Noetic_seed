"""JSON-RPC 2.0 framing for MCP.

claw-code 参照: rust/crates/runtime/src/mcp_server.rs:1-441

MCP は JSON-RPC 2.0 を stdio 上で Content-Length ヘッダ付きで運ぶ。
LSP と同じ framing (header + \r\n\r\n + body)。
"""
import json
from dataclasses import dataclass, field
from typing import Optional, Union


PROTOCOL_VERSION = "2024-11-05"  # MCP spec version
JSONRPC_VERSION = "2.0"


# ============================================================
# Message types
# ============================================================

@dataclass
class JsonRpcRequest:
    method: str
    params: Optional[dict] = None
    id: Optional[Union[str, int]] = None  # None = notification

    def to_dict(self) -> dict:
        d = {"jsonrpc": JSONRPC_VERSION, "method": self.method}
        if self.params is not None:
            d["params"] = self.params
        if self.id is not None:
            d["id"] = self.id
        return d


@dataclass
class JsonRpcError:
    code: int
    message: str
    data: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


@dataclass
class JsonRpcResponse:
    id: Union[str, int, None]
    result: Optional[dict] = None
    error: Optional[JsonRpcError] = None

    def to_dict(self) -> dict:
        d = {"jsonrpc": JSONRPC_VERSION, "id": self.id}
        if self.error is not None:
            d["error"] = self.error.to_dict()
        else:
            d["result"] = self.result if self.result is not None else {}
        return d


# ============================================================
# Error codes (JSON-RPC 2.0 + MCP)
# ============================================================

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# ============================================================
# Framing: Content-Length header + CRLF
# ============================================================

def encode_message(msg) -> bytes:
    """JsonRpcRequest / JsonRpcResponse / dict -> framed bytes。

    Format:
        Content-Length: N\r\n
        \r\n
        <body of N bytes UTF-8 JSON>
    """
    if hasattr(msg, "to_dict"):
        body_obj = msg.to_dict()
    elif isinstance(msg, dict):
        body_obj = msg
    else:
        raise TypeError(f"unsupported msg type: {type(msg)}")
    body = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def parse_message(buf: bytes) -> tuple:
    """framed bytes から 1 message を取り出す。

    戻り値: (message_dict_or_None, remaining_bytes)。
    message が完結していなければ (None, buf) を返す。
    """
    sep = b"\r\n\r\n"
    idx = buf.find(sep)
    if idx < 0:
        return (None, buf)

    header = buf[:idx].decode("ascii", errors="replace")
    # Content-Length: 123
    content_length = None
    for line in header.split("\r\n"):
        if line.lower().startswith("content-length:"):
            try:
                content_length = int(line.split(":", 1)[1].strip())
            except ValueError:
                return (None, buf)
            break
    if content_length is None:
        # header が壊れている。ヘッダ部分だけ捨てて次の境界へ
        return (None, buf[idx + len(sep):])

    body_start = idx + len(sep)
    body_end = body_start + content_length
    if len(buf) < body_end:
        return (None, buf)

    body_bytes = buf[body_start:body_end]
    remaining = buf[body_end:]
    try:
        msg = json.loads(body_bytes.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return (None, remaining)
    return (msg, remaining)


def parse_all_messages(buf: bytes) -> tuple:
    """buf に含まれる完結した message を全て取り出す。

    戻り値: (messages: list[dict], remaining: bytes)
    """
    messages: list = []
    cur = buf
    while True:
        msg, cur = parse_message(cur)
        if msg is None:
            break
        messages.append(msg)
    return (messages, cur)
