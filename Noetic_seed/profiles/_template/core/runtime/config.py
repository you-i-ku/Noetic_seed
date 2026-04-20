"""Runtime Config — 3-level config merge.

claw-code 参照: rust/crates/runtime/src/config.rs (500+行)

階層 (後勝ち deep merge):
  1. ~/.claw/settings.json           (user primary)
  2. ~/.claw.json                    (user legacy)
  3. <workspace>/.claw/settings.json (project primary)
  4. <workspace>/.claw.json          (project legacy)
  5. <workspace>/.claw/settings.local.json (local override)

MCP server 設定:
  mcp_servers: {
    "name": {
      "type": "stdio" | "sse" | "http" | "websocket" | "sdk" | "managed_proxy",
      ... transport 固有フィールド
    }
  }
"""
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================
# MCP server config enum
# ============================================================

@dataclass
class McpServerConfig:
    name: str
    type: str                        # stdio / sse / http / websocket / sdk / managed_proxy
    raw: dict = field(default_factory=dict)

    # stdio 固有
    command: Optional[str] = None
    args: list = field(default_factory=list)
    env: dict = field(default_factory=dict)

    # sse / http / websocket 固有
    url: Optional[str] = None
    headers: dict = field(default_factory=dict)

    # sdk 固有 (claw-code 同梱 MCP サーバー参照名)
    sdk_name: Optional[str] = None

    # managed_proxy 固有
    proxy_id: Optional[str] = None

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "McpServerConfig":
        t = (data.get("type") or "stdio").lower()
        return cls(
            name=name,
            type=t,
            raw=dict(data),
            command=data.get("command"),
            args=list(data.get("args") or []),
            env=dict(data.get("env") or {}),
            url=data.get("url"),
            headers=dict(data.get("headers") or {}),
            sdk_name=data.get("name") if t == "sdk" else None,
            proxy_id=data.get("id") if t == "managed_proxy" else None,
        )


# ============================================================
# OAuth config
# ============================================================

@dataclass
class OAuthConfig:
    client_id: Optional[str] = None
    authorize_url: Optional[str] = None
    token_url: Optional[str] = None
    callback_port: int = 0
    scopes: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["OAuthConfig"]:
        if not data:
            return None
        return cls(
            client_id=data.get("client_id"),
            authorize_url=data.get("authorize_url"),
            token_url=data.get("token_url"),
            callback_port=int(data.get("callback_port") or 0),
            scopes=list(data.get("scopes") or []),
        )


# ============================================================
# RuntimeConfig
# ============================================================

@dataclass
class RuntimeConfig:
    """deep-merged 設定。"""
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    permission_mode: Optional[str] = None
    mcp_servers: dict = field(default_factory=dict)   # name -> McpServerConfig
    hooks: dict = field(default_factory=dict)         # event -> [commands]
    aliases: dict = field(default_factory=dict)       # alias -> tool
    permissions: dict = field(default_factory=dict)   # {mode, rules: {allow/ask/deny}}
    auto_compaction_threshold: Optional[int] = None
    max_tools_per_cycle: int = 1
    oauth: Optional[OAuthConfig] = None
    raw: dict = field(default_factory=dict)           # マージ後の生 dict

    @classmethod
    def from_dict(cls, data: dict) -> "RuntimeConfig":
        return cls(
            model=data.get("model"),
            max_tokens=(int(data["max_tokens"])
                        if data.get("max_tokens") is not None else None),
            permission_mode=data.get("permission_mode"),
            mcp_servers={
                name: McpServerConfig.from_dict(name, cfg)
                for name, cfg in (data.get("mcp_servers") or {}).items()
                if isinstance(cfg, dict)
            },
            hooks=dict(data.get("hooks") or {}),
            aliases=dict(data.get("aliases") or {}),
            permissions=dict(data.get("permissions") or {}),
            auto_compaction_threshold=data.get("auto_compaction_threshold"),
            max_tools_per_cycle=int(data.get("max_tools_per_cycle") or 1),
            oauth=OAuthConfig.from_dict(data.get("oauth")),
            raw=data,
        )


# ============================================================
# Deep merge + loader
# ============================================================

def deep_merge(base: dict, override: dict) -> dict:
    """dict の深いマージ。list は上書き (extend しない)。"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if (k in out and isinstance(out[k], dict)
                and isinstance(v, dict)):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_json(p: Path) -> dict:
    if not p.exists() or not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_runtime_config(
    workspace_root: Path,
    user_home: Optional[Path] = None,
) -> RuntimeConfig:
    """3-level config を deep merge して RuntimeConfig を返す。

    後ろの階層が前の階層を上書き (ネストは deep merge、list は上書き)。
    """
    home = user_home or Path.home()
    layers = [
        home / ".claw" / "settings.json",
        home / ".claw.json",
        workspace_root / ".claw" / "settings.json",
        workspace_root / ".claw.json",
        workspace_root / ".claw" / "settings.local.json",
    ]
    merged: dict = {}
    for p in layers:
        merged = deep_merge(merged, _load_json(p))
    return RuntimeConfig.from_dict(merged)
