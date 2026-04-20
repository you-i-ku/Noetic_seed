"""Noetic_seed MCP server CLI エントリポイント (段階6-C v3)。

.mcp.json から外部プロセス (Claude Code 等) に spawn される想定。
stdio transport で JSON-RPC 2.0 をやり取りする。

使い方 (.mcp.json):
    {
      "mcpServers": {
        "noetic-seed": {
          "command": "C:/.../python.exe",
          "args": ["C:/.../profiles/_template/mcp_server_main.py"]
        }
      }
    }

注意:
- stdout は JSON-RPC 専用、print() 禁止 (stdout を壊す)
- ログは logging 経由で stderr へ
- main.py と同時起動で state.json を共有する想定だが、この server は
  state 読み取りのみ (書き込みは pending_mcp_inputs.jsonl append のみ)
"""
import logging
import sys
from pathlib import Path

# profile root (this file's directory) を sys.path に追加して
# core.* を import 可能にする
_PROFILE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROFILE_ROOT))

# stderr にログを出す (stdout は JSON-RPC 専用)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
_log = logging.getLogger("noetic_seed.mcp_server")


def main() -> None:
    from core.runtime.mcp.server.server import mcp_server
    _log.info("Noetic_seed MCP server starting (profile=%s)", _PROFILE_ROOT.name)
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    main()
