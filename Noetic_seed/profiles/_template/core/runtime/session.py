"""Session — 会話履歴管理 (agent loop の messages バッファ)。

claw-code の rust/crates/runtime/src/session.rs の Python port (簡略版)。

上半分 (messages / push_user_text / push_assistant_message / push_tool_result
/ serialize_*) は claw-code 準拠の純粋インフラ。
Noetic 固有の push_observation は Phase 4 Step D で追加 (UPS v2 の
observation 概念を Session に流し込む拡張)。
"""
from datetime import datetime
from typing import Optional

from core.providers.base import AssistantMessage


_VALID_OBS_FORMATS = (
    "structured_compact",  # [obs channel=X action=Y time=Z]           default
    "structured_full",     # [observation channel=X ... source_action=W]
    "natural_ja",          # [Xからの声 Z]
    "compact",             # [obs X Z]
)


class Session:
    """会話履歴バッファ。

    Attributes:
        messages: Anthropic content-block 形式の message list (push_* で追記)。
        observation_label_format: push_observation 時の label 書式
            ("structured_compact" / "structured_full" / "natural_ja" / "compact")。
            default は "structured_compact" (INTEGRATION_POINTS §2.4)。
        observations: 積まれた observation の metadata 履歴
            (format 変更時の再レンダリングや分析用。messages とは独立)。
    """

    def __init__(self, observation_label_format: str = "structured_compact"):
        if observation_label_format not in _VALID_OBS_FORMATS:
            raise ValueError(
                f"unknown observation_label_format="
                f"{observation_label_format!r}; expected one of "
                f"{_VALID_OBS_FORMATS}"
            )
        self.messages: list = []
        self.observation_label_format: str = observation_label_format
        self.observations: list[dict] = []

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

    def push_observation(
        self,
        observed_channel: str,
        content: str,
        actor: Optional[str] = None,
        source_action_hint: str = "living_presence",
        observation_time: Optional[str] = None,
    ) -> None:
        """observation (外部到着 or 内部観測) を Session に積む。

        INTEGRATION_POINTS.md §2.4 の仕様。UPS v2 の observation 概念を
        LLM コンテキストに流し込む専用メソッド。user message として
        messages に積むが、prefix に label が付く点で push_user_text と
        区別される。metadata は self.observations にも保持する
        (format 再レンダリングや分析用)。

        Args:
            observed_channel: "device" / "elyth" / "x" / "self" など。
            content: 観測本文 (ゆうの発話、SNS 通知等)。
            actor: 話者識別子 (例: "ent_yuu")。natural_ja format で使う。
            source_action_hint: 対応する AI 側 action のヒント。
                未特定時は "living_presence" (UPS v2 の spontaneous 到着)。
            observation_time: "HH:MM" 形式。None で現在時刻を採用。
        """
        if not content:
            return
        if observation_time is None:
            observation_time = datetime.now().strftime("%H:%M")

        meta = {
            "observed_channel": observed_channel,
            "content": content,
            "actor": actor,
            "source_action_hint": source_action_hint,
            "observation_time": observation_time,
        }
        self.observations.append(meta)

        label = self._render_observation_label(
            meta, self.observation_label_format,
        )
        text = f"{label} {content}" if label else content
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": text}],
        })

    @staticmethod
    def _render_observation_label(meta: dict, fmt: str) -> str:
        """observation metadata を指定 format の label 文字列に変換。

        4 format 対応 (INTEGRATION_POINTS §2.4):
          - structured_compact: [obs channel=X action=Y time=Z]       default
          - structured_full:    [observation channel=X action=Y
                                 time=Z source_action=W]
          - natural_ja:         [Xからの声 Z] (既存 Noetic 流儀継承)
          - compact:            [obs X Z]

        未知 format は structured_compact にフォールバック (defensive)。
        """
        channel = str(meta.get("observed_channel") or "?")
        action = str(meta.get("source_action_hint") or "?")
        time_str = str(meta.get("observation_time") or "")
        actor = meta.get("actor")

        if fmt == "structured_full":
            return (
                f"[observation channel={channel} action={action} "
                f"time={time_str} source_action={action}]"
            )
        if fmt == "natural_ja":
            speaker = actor or channel
            return f"[{speaker}からの声 {time_str}]"
        if fmt == "compact":
            return f"[obs {channel} {time_str}]"
        # structured_compact (default) + unknown fallback
        return f"[obs channel={channel} action={action} time={time_str}]"

    def clear(self) -> None:
        self.messages = []
        self.observations = []

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
