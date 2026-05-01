"""BaseProvider — LLM provider 抽象化のインターフェース。

claw-code の rust/crates/api/src/client.rs ApiClient trait の Python port。
厳密 claw-code 準拠 (+ claude_code provider 対応の最小拡張)。
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union


# claude_code provider 専用の tool 実行 callback 型。
# (tool_id, tool_name, tool_input) -> (output_str, is_error)
ToolExecutor = Callable[[str, str, dict], Tuple[str, bool]]


@dataclass
class ToolUseBlock:
    """LLM が出力した tool_use。"""
    id: str
    name: str
    input: dict


@dataclass
class ToolInvocationRecord:
    """Provider 完結型 (claude_code) で SDK 内部実行された tool 1 件の記録。"""
    tool_id: str
    tool_name: str
    tool_input: dict
    output: str = ""
    is_error: bool = False


@dataclass
class ApiRequest:
    """LLM 呼出リクエスト。"""
    system_prompt: str = ""
    messages: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    max_tokens: int = 24000
    temperature: float = 0.7
    image_paths: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None
    # provider 固有の tool_choice 形式をそのまま渡す:
    #   OpenAI 互換: "required" (string) — tools 側で単一 tool に絞って強制する。
    #                object 形式 ({"type":"function",...}) は LM Studio 等
    #                一部 backend 非対応なので避ける。
    #   Anthropic:   {"type": "tool", "name": "..."} (object 形式が正式サポート)
    # None なら provider のデフォルト (OpenAI="auto" / Anthropic=未指定) を使う。
    tool_executor: Optional[ToolExecutor] = None
    # claude_code provider 専用: in-process MCP handler 内から呼ばれて
    # ConversationRuntime の hook + permission + approval を経由した tool 実行を委譲する。
    # anthropic / openai_compat provider は無視 (使わない)。


@dataclass
class AssistantMessage:
    """LLM 応答 1 件分。"""
    text: str = ""
    tool_uses: list = field(default_factory=list)
    usage: Optional[dict] = None
    stop_reason: Optional[str] = None
    raw: Optional[dict] = None
    tool_invocations: list = field(default_factory=list)
    # claude_code provider 専用: SDK 内部で実行された tool の記録 (ToolInvocationRecord list)。
    # ConversationRuntime はこれが非空なら Provider 完結型として扱い、
    # tool_uses を見ずに tool_invocations を消費する。anthropic / openai_compat は空。


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
