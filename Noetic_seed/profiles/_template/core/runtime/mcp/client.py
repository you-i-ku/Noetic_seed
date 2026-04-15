"""MCP client transport 抽象化。

claw-code 参照: rust/crates/runtime/src/mcp_client.rs:73-109

5 transport:
  - stdio (実装)
  - sse / http / websocket (型のみ、future work)
  - sdk (型のみ)
  - managed_proxy (型のみ)
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TransportType(Enum):
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"
    WEBSOCKET = "websocket"
    SDK = "sdk"
    MANAGED_PROXY = "managed_proxy"


class BaseTransport:
    """全 transport の共通 interface。

    実装は sync 呼出の薄いラッパ。async 化は将来。
    """

    transport_type: TransportType = TransportType.STDIO

    def start(self) -> None:
        """transport を起動する。"""
        raise NotImplementedError

    def stop(self) -> None:
        """transport を停止する。"""
        raise NotImplementedError

    def send(self, message) -> None:
        """JsonRpcRequest / JsonRpcResponse を送信。"""
        raise NotImplementedError

    def recv(self, timeout: float = 30.0) -> Optional[dict]:
        """次の message dict を受信。timeout で None。"""
        raise NotImplementedError

    def is_running(self) -> bool:
        return False


# ============================================================
# Remote transports (stub only)
# ============================================================

@dataclass
class RemoteTransportConfig:
    url: str
    headers: dict = None


class SseTransport(BaseTransport):
    transport_type = TransportType.SSE

    def __init__(self, config: RemoteTransportConfig):
        self.config = config

    def start(self):
        raise NotImplementedError("SSE transport not yet implemented")

    def stop(self):
        pass

    def send(self, message):
        raise NotImplementedError

    def recv(self, timeout=30.0):
        raise NotImplementedError


class HttpTransport(BaseTransport):
    transport_type = TransportType.HTTP

    def __init__(self, config: RemoteTransportConfig):
        self.config = config

    def start(self):
        raise NotImplementedError("HTTP transport not yet implemented")

    def stop(self):
        pass

    def send(self, message):
        raise NotImplementedError

    def recv(self, timeout=30.0):
        raise NotImplementedError


class WebSocketTransport(BaseTransport):
    transport_type = TransportType.WEBSOCKET

    def __init__(self, config: RemoteTransportConfig):
        self.config = config

    def start(self):
        raise NotImplementedError("WebSocket transport not yet implemented")

    def stop(self):
        pass

    def send(self, message):
        raise NotImplementedError

    def recv(self, timeout=30.0):
        raise NotImplementedError


class SdkTransport(BaseTransport):
    """claw-code 同梱の MCP server (in-process) 用。"""
    transport_type = TransportType.SDK

    def __init__(self, name: str):
        self.name = name

    def start(self):
        raise NotImplementedError("SDK transport not yet implemented")

    def stop(self):
        pass

    def send(self, message):
        raise NotImplementedError

    def recv(self, timeout=30.0):
        raise NotImplementedError


class ManagedProxyTransport(BaseTransport):
    """CCR (Claude Code Router) proxy 経由。"""
    transport_type = TransportType.MANAGED_PROXY

    def __init__(self, url: str, proxy_id: str):
        self.url = url
        self.proxy_id = proxy_id

    def start(self):
        raise NotImplementedError("ManagedProxy transport not yet implemented")

    def stop(self):
        pass

    def send(self, message):
        raise NotImplementedError

    def recv(self, timeout=30.0):
        raise NotImplementedError
