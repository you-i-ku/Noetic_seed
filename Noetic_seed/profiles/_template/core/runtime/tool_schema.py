"""Tool Schema — JSON Schema で tool を定義する箱。

claw-code の rust/crates/tools/src/lib.rs:385-1172 (mvp_tool_specs) の Python port。

責務:
  - tool の name / description / input_schema / required_permission を保持
  - LLM provider への tool list 変換
  - function calling native 形式で LLM に渡す

TODO: 別セッションで実装。現状は型定義のみ。
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.runtime.permissions import PermissionMode


@dataclass
class ToolSpec:
    """tool 定義の標準形。"""
    name: str
    description: str
    input_schema: dict  # JSON Schema
    required_permission: PermissionMode
    handler: Callable  # (input: dict) -> str
    channel: Optional[str] = None  # Noetic 固有: device/x/elyth/display/internal

    def to_anthropic_format(self) -> dict:
        """Anthropic Messages API 形式に変換。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai_format(self) -> dict:
        """OpenAI Chat Completions 形式に変換。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_gemini_format(self) -> dict:
        """Google Gemini 形式に変換。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }
