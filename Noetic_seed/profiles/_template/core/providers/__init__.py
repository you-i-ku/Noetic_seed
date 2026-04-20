"""LLM Provider abstraction.

claw-code の rust/crates/api/src/providers/ の Python port。

既存 core/llm.py は LM Studio / OpenAI / Gemini / Claude / claude_code を
if 分岐で処理している。これを provider 抽象化で整理する。

構成:
  base.py           — BaseProvider インターフェース
  anthropic.py      — Claude Messages API
  openai_compat.py  — OpenAI / LM Studio / Gemini (OpenAI互換)
  xai.py            — Grok
  dashscope.py      — Alibaba Qwen
  claude_code.py    — claude -p CLI

既存 core/llm.py は当面そのまま保持。新 provider 層は runtime 経由で使う。

TODO: 別セッションで実装。
"""
