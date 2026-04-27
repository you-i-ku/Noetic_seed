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
import uuid
from typing import AsyncIterator

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
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
        # Step 2: image_paths があれば _build_prompt_async_iterable 内で
        # 最後の user message content array に Anthropic native image block を注入。
        # Step 3: tools があれば in-process MCP server に射影、handler 内で
        # request.tool_executor 経由で Noetic ToolRegistry に dispatch する。

        captured_invocations: list = []

        options_kwargs: dict = {
            "model": self.model,
            "system_prompt": request.system_prompt or None,
            "max_turns": 1,
            # 外部 settings (CLAUDE.md auto-discovery / .mcp.json / 既存環境の
            # 外部 MCP server 等) の読み込みを完全無効化。Noetic は self-contained
            # な provider として SDK に渡した mcp_servers のみ使う。
            "setting_sources": [],
            # Claude Code CLI の built-in tools (Bash/Read/Edit/Write/Glob/Grep/
            # ToolSearch/WebFetch/...) を全 disable。tools 引数は whitelist 形式
            # で、空 list = built-in 全消灯。Anthropic 側で built-in 追加されても
            # 影響なし (= 動的解決、ハードコード list 不要、ゆう gut check 2026-04-27)。
            "tools": [],
        }

        # Step 3: in-process MCP 配線 (tools 非空時のみ)
        if request.tools:
            if request.tool_executor is None:
                raise RuntimeError(
                    "claude_code provider: tools 指定時は ApiRequest.tool_executor "
                    "必須 (ConversationRuntime._make_tool_executor 経由で渡される)"
                )
            handlers: list = []
            tool_names: list = []
            for tool_def in request.tools:
                handlers.append(self._build_tool_handler(
                    tool_def, request.tool_executor, captured_invocations,
                ))
                tool_names.append(tool_def["name"])

            mcp_server = create_sdk_mcp_server(
                name="noetic", version="1.0.0", tools=handlers,
            )
            options_kwargs["mcp_servers"] = {"noetic": mcp_server}
            options_kwargs["allowed_tools"] = [
                f"mcp__noetic__{n}" for n in tool_names
            ]

        options = ClaudeAgentOptions(**options_kwargs)

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
            tool_invocations=captured_invocations,
        )

    @staticmethod
    def _build_tool_handler(tool_def: dict, tool_executor, captured: list):
        """Anthropic native tool 定義を in-process MCP @tool handler に射影。

        Args:
            tool_def: {"name": ..., "description": ..., "input_schema": ...}
            tool_executor: (tool_id, name, input) -> (output_str, is_error)
                ConversationRuntime._make_tool_executor で生成された callable。
            captured: ClaudeCodeProvider._stream_async 内の list、handler 内で
                実行した invocation を append する (AssistantMessage.tool_invocations
                に詰める用)。

        Returns:
            @tool decorator が返す handler 関数。
        """
        tool_name = tool_def["name"]
        tool_desc = tool_def.get("description", "")
        tool_schema = tool_def.get("input_schema", {"type": "object"})

        @tool(tool_name, tool_desc, tool_schema)
        async def handler(args, _name=tool_name):
            tool_id = f"call_{uuid.uuid4().hex[:8]}"
            # tool_executor は同期 (Noetic ToolRegistry.execute も同期)
            # → asyncio.to_thread で event loop を blocking しないよう非同期化
            output, is_error = await asyncio.to_thread(
                tool_executor, tool_id, _name, args,
            )
            captured.append({
                "tool_id": tool_id,
                "tool_name": _name,
                "tool_input": args,
                "output": output,
                "is_error": is_error,
            })
            return {
                "content": [{"type": "text", "text": output}],
                "is_error": is_error,
            }

        return handler

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
