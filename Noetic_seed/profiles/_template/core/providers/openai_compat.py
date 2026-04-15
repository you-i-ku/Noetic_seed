"""OpenAIProvider — OpenAI Chat Completions + Tools 互換。

対応エンドポイント:
  - OpenAI API
  - Gemini OpenAI 互換
  - LM Studio REST
  - OpenRouter / Ollama / 任意の OpenAI 互換サーバー

claw-code 参照: rust/crates/api/src/providers/openai_compat.rs
厳密 claw-code 準拠。Noetic 既存コードに依存しない。
"""
import json
import uuid

import httpx

from core.providers._image import load_image_base64
from core.providers.base import (
    ApiRequest,
    AssistantMessage,
    BaseProvider,
    ToolUseBlock,
)


class OpenAIProvider(BaseProvider):
    """OpenAI Chat Completions 互換 provider。"""

    name = "openai_compat"

    def supports_tool_use(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return True

    def stream(self, request: ApiRequest) -> AssistantMessage:
        payload = self._build_payload(request)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        resp = httpx.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        return self._parse_response(resp.json())

    def _build_payload(self, req: ApiRequest) -> dict:
        messages = self._build_messages(req)
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if req.tools:
            payload["tools"] = req.tools
            payload["tool_choice"] = "auto"
        return payload

    def _build_messages(self, req: ApiRequest) -> list:
        out: list = []
        if req.system_prompt:
            out.append({"role": "system", "content": req.system_prompt})
        out.extend(req.messages)
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
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            })
        return content

    def _parse_response(self, data: dict) -> AssistantMessage:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError):
            return AssistantMessage(text="", raw=data, stop_reason="error")

        msg = choice.get("message", {})
        text = msg.get("content") or ""
        finish_reason = choice.get("finish_reason", "")

        tool_uses: list = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "")
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            tool_uses.append(ToolUseBlock(
                id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                name=fn.get("name", ""),
                input=args,
            ))

        return AssistantMessage(
            text=text,
            tool_uses=tool_uses,
            usage=data.get("usage"),
            stop_reason=self._normalize_stop_reason(finish_reason,
                                                    bool(tool_uses)),
            raw=data,
        )

    @staticmethod
    def _normalize_stop_reason(finish_reason: str, has_tool: bool) -> str:
        if finish_reason == "tool_calls" or has_tool:
            return "tool_use"
        if finish_reason == "length":
            return "max_tokens"
        if finish_reason == "stop":
            return "end_turn"
        return finish_reason or "end_turn"
