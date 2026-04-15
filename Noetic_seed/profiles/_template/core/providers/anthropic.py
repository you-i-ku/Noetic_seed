"""AnthropicProvider — Claude Messages API (native tool_use)。

claw-code 参照: rust/crates/api/src/providers/anthropic.rs
厳密 claw-code 準拠。Noetic 既存コードに依存しない。
"""
import uuid

import httpx

from core.providers._image import load_image_base64
from core.providers.base import (
    ApiRequest,
    AssistantMessage,
    BaseProvider,
    ToolUseBlock,
)


class AnthropicProvider(BaseProvider):
    """Anthropic Claude Messages API provider。"""

    name = "anthropic"
    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, model: str, api_key: str = "",
                 base_url: str = "https://api.anthropic.com"):
        super().__init__(model=model, api_key=api_key, base_url=base_url)

    def supports_tool_use(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return True

    def stream(self, request: ApiRequest) -> AssistantMessage:
        payload = self._build_payload(request)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        url = f"{self.base_url}/v1/messages"
        resp = httpx.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        return self._parse_response(resp.json())

    def _build_payload(self, req: ApiRequest) -> dict:
        messages = self._build_messages(req)
        payload: dict = {
            "model": self.model,
            "max_tokens": req.max_tokens,
            "messages": messages,
        }
        if req.system_prompt:
            payload["system"] = req.system_prompt
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.tools:
            payload["tools"] = req.tools
        return payload

    def _build_messages(self, req: ApiRequest) -> list:
        out = list(req.messages)
        if req.image_paths and out:
            last = out[-1]
            if last.get("role") == "user":
                last["content"] = self._inline_images(
                    last.get("content", ""), req.image_paths
                )
        return out

    def _inline_images(self, text_content, image_paths: list) -> list:
        content: list = []
        if isinstance(text_content, str) and text_content:
            content.append({"type": "text", "text": text_content})
        elif isinstance(text_content, list):
            content.extend(text_content)

        for ip in image_paths:
            img = load_image_base64(ip)
            if img is None:
                continue
            b64, media_type = img
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            })
        return content

    def _parse_response(self, data: dict) -> AssistantMessage:
        text_parts: list = []
        tool_uses: list = []

        for block in data.get("content", []) or []:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_uses.append(ToolUseBlock(
                    id=block.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    name=block.get("name", ""),
                    input=block.get("input") or {},
                ))

        return AssistantMessage(
            text="".join(text_parts),
            tool_uses=tool_uses,
            usage=data.get("usage"),
            stop_reason=data.get("stop_reason") or "end_turn",
            raw=data,
        )
