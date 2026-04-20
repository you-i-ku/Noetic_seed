"""CommandDispatcher — slash command の登録と dispatch。

claw-code 参照: rust/crates/commands/src/lib.rs

責務:
  - CommandSpec (name / description / handler) の登録
  - "/foo arg1 arg2" のパース → handler(args_str, ctx) 呼出
  - 未登録なら "/help" にフォールバック (または not found)
"""
import shlex
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class CommandResult:
    """command 実行結果。"""
    text: str = ""
    is_error: bool = False
    # UI 層が追加動作するためのフラグ (exit / clear 等)
    action: Optional[str] = None


@dataclass
class CommandSpec:
    """slash command 定義。"""
    name: str                 # "status" (頭の / は除く)
    description: str
    handler: Callable         # (args: list[str], ctx: dict) -> CommandResult
    usage: str = ""


class CommandDispatcher:
    """slash command の登録と dispatch。"""

    def __init__(self):
        self._cmds: dict = {}

    def register(self, spec: CommandSpec) -> None:
        self._cmds[spec.name] = spec

    def has(self, name: str) -> bool:
        return name in self._cmds

    def get(self, name: str) -> Optional[CommandSpec]:
        return self._cmds.get(name)

    def all_names(self) -> list:
        return sorted(self._cmds.keys())

    def dispatch(self, line: str, ctx: Optional[dict] = None) -> CommandResult:
        """'/status' や '/commit -m msg' を処理。/ が先頭になければ not-a-command。"""
        ctx = ctx or {}
        line = (line or "").strip()
        if not line:
            return CommandResult(text="", is_error=True)
        if not line.startswith("/"):
            return CommandResult(
                text=f"Not a command: {line[:40]}",
                is_error=True,
            )

        # " / status  -m  msg " → ["status", "-m", "msg"]
        try:
            parts = shlex.split(line[1:])
        except ValueError as e:
            return CommandResult(text=f"Parse error: {e}", is_error=True)
        if not parts:
            return CommandResult(text="Empty command", is_error=True)

        name = parts[0]
        args = parts[1:]

        spec = self._cmds.get(name)
        if spec is None:
            return CommandResult(
                text=(f"Unknown command: /{name}\n"
                      f"Available: {', '.join('/'+n for n in self.all_names())}"),
                is_error=True,
            )

        try:
            return spec.handler(args, ctx)
        except Exception as e:
            return CommandResult(
                text=f"Command /{name} failed: {type(e).__name__}: {e}",
                is_error=True,
            )
