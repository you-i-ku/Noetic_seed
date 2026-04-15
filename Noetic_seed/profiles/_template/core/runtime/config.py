"""Runtime Config — 3-level config merge.

claw-code の rust/crates/runtime/src/config.rs (500+行) の Python port。

責務:
  - user / project / local の 3 層 config を deep merge
  - MCP servers 設定
  - hooks 設定
  - aliases 設定
  - permissions 設定

階層:
  1. ~/.claude/settings.json        (user)
  2. <profile>/settings.json        (project / 既存)
  3. <profile>/settings.local.json  (local override)

TODO: 別セッションで実装。
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RuntimeConfig:
    """実行時設定の標準形。"""
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    permission_mode: Optional[str] = None
    mcp_servers: dict = field(default_factory=dict)
    hooks: dict = field(default_factory=dict)
    aliases: dict = field(default_factory=dict)
    permissions: dict = field(default_factory=dict)
    max_tools_per_cycle: int = 1  # 1 = Noetic 細粒度息継ぎ型 (default)


def load_runtime_config(profile_dir) -> RuntimeConfig:
    """3 層 merge して RuntimeConfig を返す。

    TODO: 実装。
    """
    raise NotImplementedError
