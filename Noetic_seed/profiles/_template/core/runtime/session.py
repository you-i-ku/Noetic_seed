"""Session — 会話履歴管理 (agent loop の messages バッファ)。

claw-code の rust/crates/runtime/src/session.rs の Python port (簡略版)。

厳密 claw-code 準拠。Anthropic content-block 形式をベースに保持し、
OpenAI 形式には serialize 時に変換。
"""
from core.providers.base import AssistantMessage


class Session:
    """会話履歴バッファ。"""

    def __init__(self):
        self.messages: list = []

    def push_user_text(self, text: str) -> None:
        if not text:
            return
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": text}],
        })

    def push_assistant_message(self, msg: AssistantMessage) -> None:
        blocks: list = []
        if msg.text:
            blocks.append({"type": "text", "text": msg.text})
        for tu in msg.tool_uses:
            blocks.append({
                "type": "tool_use",
                "id": tu.id,
                "name": tu.name,
                "input": tu.input,
            })
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        self.messages.append({"role": "assistant", "content": blocks})

    def push_tool_result(self, tool_use_id: str, content: str,
                         is_error: bool = False) -> None:
        block = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            block["is_error"] = True
        self.messages.append({"role": "user", "content": [block]})

    def clear(self) -> None:
        self.messages = []

    # ---- シリアライズ ----

    def serialize_for_anthropic(self) -> list:
        """Anthropic Messages API 形式にそのまま。"""
        return [dict(m) for m in self.messages]

    def serialize_for_openai(self) -> list:
        """OpenAI Chat Completions 形式に変換。"""
        out: list = []
        for msg in self.messages:
            role = msg["role"]
            content_blocks = msg["content"]

            if role == "user":
                tool_results = [b for b in content_blocks
                                if b.get("type") == "tool_result"]
                if tool_results:
                    for tr in tool_results:
                        out.append({
                            "role": "tool",
                            "tool_call_id": tr["tool_use_id"],
                            "content": tr.get("content", ""),
                        })
                else:
                    out.append(self._user_blocks_to_openai(content_blocks))

            elif role == "assistant":
                out.append(self._assistant_blocks_to_openai(content_blocks))

        return out

    def _user_blocks_to_openai(self, blocks: list) -> dict:
        text_blocks = [b for b in blocks if b.get("type") == "text"]
        image_blocks = [b for b in blocks if b.get("type") == "image"]

        if not image_blocks:
            text = "\n".join(b.get("text", "") for b in text_blocks)
            return {"role": "user", "content": text}

        content: list = []
        for b in text_blocks:
            content.append({"type": "text", "text": b.get("text", "")})
        for b in image_blocks:
            source = b.get("source", {})
            b64 = source.get("data", "")
            media_type = source.get("media_type", "image/jpeg")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            })
        return {"role": "user", "content": content}

    def _assistant_blocks_to_openai(self, blocks: list) -> dict:
        import json

        text_parts = [b.get("text", "") for b in blocks
                      if b.get("type") == "text"]
        tool_use_blocks = [b for b in blocks
                           if b.get("type") == "tool_use"]

        msg: dict = {"role": "assistant"}
        text = "".join(text_parts)
        msg["content"] = text if text else None
        if tool_use_blocks:
            msg["tool_calls"] = [
                {
                    "id": b.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": b.get("name", ""),
                        "arguments": json.dumps(b.get("input") or {},
                                                ensure_ascii=False),
                    },
                }
                for b in tool_use_blocks
            ]
        return msg
