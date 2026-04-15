"""BaseProvider — LLM provider 抽象化のインターフェース。

claw-code の rust/crates/api/src/client.rs ApiClient trait の Python port。
厳密 claw-code 準拠。
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolUseBlock:
    """LLM が出力した tool_use。"""
    id: str
    name: str
    input: dict


@dataclass
class ApiRequest:
    """LLM 呼出リクエスト。"""
    system_prompt: str = ""
    messages: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    max_tokens: int = 24000
    temperature: float = 0.7
    image_paths: Optional[list] = None


@dataclass
class AssistantMessage:
    """LLM 応答 1 件分。"""
    text: str = ""
    tool_uses: list = field(default_factory=list)
    usage: Optional[dict] = None
    stop_reason: Optional[str] = None
    raw: Optional[dict] = None


class BaseProvider:
    """LLM provider の共通インターフェース。"""

    name: str = "base"

    def __init__(self, model: str, api_key: str = "", base_url: str = ""):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""

    def stream(self, request: ApiRequest) -> AssistantMessage:
        """LLM を呼んで AssistantMessage を返す。実装は各サブクラス。"""
        raise NotImplementedError

    def supports_tool_use(self) -> bool:
        return False

    def supports_vision(self) -> bool:
        return False
