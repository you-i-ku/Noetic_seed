"""ClaudeCodeProvider — Claude Code CLI subprocess wrap (claude-agent-sdk 経由)。

Claude Pro/Max subscription の Sonnet 枠を流用する。
auth は bundled CLI 経由 (claude auth login の OAuth または
CLAUDE_CODE_OAUTH_TOKEN env)。pay-per-token API key 経路ではない。

PLAN: WORLD_MODEL_DESIGN/CLAUDE_CODE_UNIFIED_PROVIDER_PLAN.md

Step 1 (本コミット): text-only skeleton。
  - LLM① ③ ④ (call_llm 経路) を新 provider 経由に統合
  - tools 非空 (LLM②) は NotImplementedError、Step 3 で実装
  - image_paths 非空 (画像対応) は警告ログ + text-only fallback、Step 2 で実装

Step 2: image block injection (user message content に Anthropic native image)
Step 3: in-process MCP + tool calling + tool_executor 委譲

Noetic 哲学:
  - 都度 spawn (1 stream() 呼びで ClaudeSDKClient を async with で開閉、
    cycle 独立性維持、memory: feedback_internal_drive と整合)
  - max_turns=1 (1 invocation = 1 turn、Noetic max_iterations=1 と一致)
"""
import asyncio
from typing import AsyncIterator

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
)
from claude_agent_sdk import AssistantMessage as SDKAssistantMessage
from claude_agent_sdk import ResultMessage as SDKResultMessage
from claude_agent_sdk import TextBlock as SDKTextBlock

from core.providers._image import load_image_base64
from core.providers.base import ApiRequest, AssistantMessage, BaseProvider


class ClaudeCodeProvider(BaseProvider):
    """Claude Code CLI subprocess wrap (claude-agent-sdk 経由)。"""

    name = "claude_code"

    def __init__(self, model: str = "sonnet", api_key: str = "",
                 base_url: str = ""):
        super().__init__(model=model or "sonnet", api_key=api_key,
                         base_url=base_url)

    def supports_tool_use(self) -> bool:
        # Step 3 で in-process MCP 経由 tool calling を実装予定
        return True

    def supports_vision(self) -> bool:
        # Step 2 で image block injection を実装予定
        return True

    def stream(self, request: ApiRequest) -> AssistantMessage:
        """同期エントリポイント。SDK の async query() を asyncio.run で wrap。

        都度 spawn パターン: 毎呼びで ClaudeSDKClient (= claude CLI subprocess)
        を 1 つ開閉する。cycle 跨ぎの session 維持はしない。
        """
        return asyncio.run(self._stream_async(request))

    async def _stream_async(self, request: ApiRequest) -> AssistantMessage:
        # Step 3 未着手: tools がある呼出は今は不可
        if request.tools:
            raise NotImplementedError(
                "claude_code provider: tool calling は Step 3 で実装予定。"
                f" (tools={len(request.tools)} 個指定)"
            )

        # Step 2: image_paths があれば _build_prompt_async_iterable 内で
        # 最後の user message content array に Anthropic native image block を注入。

        options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=request.system_prompt or None,
            max_turns=1,
        )

        text_parts: list = []
        usage = None
        stop_reason = "end_turn"
        raw_messages: list = []

        prompt_iter = self._build_prompt_async_iterable(request)

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt_iter)
            async for msg in client.receive_response():
                raw_messages.append(msg)
                if isinstance(msg, SDKAssistantMessage):
                    for block in msg.content:
                        if isinstance(block, SDKTextBlock):
                            text_parts.append(block.text)
                elif isinstance(msg, SDKResultMessage):
                    usage = getattr(msg, "usage", None)
                    sr = getattr(msg, "stop_reason", None)
                    if sr:
                        stop_reason = sr

        return AssistantMessage(
            text="".join(text_parts),
            tool_uses=[],
            usage=usage,
            stop_reason=stop_reason,
            raw={"messages_count": len(raw_messages)},
            tool_invocations=[],
        )

    @staticmethod
    async def _build_prompt_async_iterable(
        request: ApiRequest,
    ) -> AsyncIterator[dict]:
        """ApiRequest の messages を SDK の AsyncIterable[dict] 形式に変換。

        Step 2: request.image_paths が非空なら最後の user message content array
        に Anthropic native image block を注入。content が str ならまず list に
        昇格してから image block を追加する。

        SDK 受付形式 (実機確認済み 2026-04-27):
            {"type": "user", "message": {"role": "user",
              "content": [{"type":"text","text":...},
                          {"type":"image","source":{...}}]}}
        """
        image_paths = list(request.image_paths or [])
        last_idx = len(request.messages) - 1

        for i, msg in enumerate(request.messages):
            role = msg.get("role", "user")

            # 最後の user message に image block を追加 (Step 2)
            if i == last_idx and role == "user" and image_paths:
                content = msg.get("content", "")
                if isinstance(content, str):
                    blocks = [{"type": "text", "text": content}] if content else []
                elif isinstance(content, list):
                    blocks = list(content)
                else:
                    blocks = []

                for ip in image_paths:
                    img = load_image_base64(ip)
                    if img is None:
                        print(f"  [claude_code] 画像読込失敗: {ip}")
                        continue
                    b64, media_type = img
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    })

                yield {
                    "type": "user",
                    "message": {"role": role, "content": blocks},
                }
            else:
                yield {
                    "type": "user" if role == "user" else "assistant",
                    "message": msg,
                }
