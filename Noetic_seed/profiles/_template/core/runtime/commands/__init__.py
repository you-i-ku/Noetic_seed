"""Slash Commands (claw-code 準拠)。

ユーザーが "/xxx" 形式で入力するコマンド。Tools とは別軸で、
agent loop の外側 (UI 層) から呼ばれる想定。

claw-code 参照: rust/crates/commands/src/lib.rs + commands_snapshot.json

構成:
  dispatcher.py  — CommandDispatcher (名前 → handler の routing)
  builtin.py     — 主要 15 command の実装

登録:
  register_default_commands(dispatcher, runtime_ctx) で一括登録。
"""
from core.runtime.commands.dispatcher import (
    CommandDispatcher,
    CommandSpec,
    CommandResult,
)
