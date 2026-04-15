"""BaseProvider — LLM provider 抽象化のインターフェース。

claw-code の rust/crates/api/src/client.rs ApiClient trait の Python port。
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ApiRequest:
    """LLM 呼出リクエスト。"""
    system_prompt: str = ""
    messages: list = field(default_factory=list)
    tools: list = field(default_factory=list)  # tool_schema.ToolSpec から変換済
    max_tokens: int = 24000
    temperature: float = 0.7
    stream: bool = False


@dataclass
class AssistantMessage:
    """LLM 応答 1 件分。"""
    text: str = ""
    tool_uses: list = field(default_factory=list)  # [{id, name, input}, ...]
    usage: Optional[dict] = None
    stop_reason: Optional[str] = None  # "end_turn" | "tool_use" | "max_tokens"


class BaseProvider:
    """LLM provider の共通インターフェース。

    各 provider はこれを継承して stream() を実装する。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def stream(self, request: ApiRequest) -> AssistantMessage:
        """LLM を呼んで AssistantMessage を返す。

        実装は各 provider サブクラスで。
        """
        raise NotImplementedError
