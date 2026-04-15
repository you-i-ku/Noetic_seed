"""v0.5 Conversation Runtime layer.

claw-code 準拠の堅牢なツール実行基盤。
既存 main.py の ★★★★ 機構 (pressure/entropy/E値/spiral/Magic-If) は変更せず、
ツール実行・hook・permission・provider 抽象化だけをこの層で置き換える。

構成:
  conversation.py  — ConversationRuntime: agent loop 層 (LLM stream + tool use)
  hooks.py         — PreToolUse / PostToolUse / PostToolUseFailure
  permissions.py   — 5 mode + allow/ask/deny rules
  tool_schema.py   — JSON Schema で tool を定義する箱
  config.py        — 3-level config merge (user/project/local)
  registry.py      — tool name → callable のマップ

参照: CLAWCODE_CAPABILITY_INVENTORY.md, V0_5_ARCHITECTURE.md
ステータス: skeleton only (実装は別セッション)
"""
