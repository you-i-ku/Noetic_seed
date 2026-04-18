"""Noetic_seed MCP server tools (段階6-C v3)。

FastMCP 経由で expose する 4 つの tool を実装。
外部公開 API 名は `noetic_seed_*` prefix (基盤名、プロファイル名 iku とは独立)。

設計指針 (STAGE6C_IMPLEMENTATION_PLAN.md v3):
- accessor-only: state.json / WM 読み取りは core/state.py / core/world_model.py 経由
- pending_mcp_inputs.jsonl への書き込み経由で main.py に伝える (直接 state 書き込み避ける)
- client_name は Context 経由で取得、channel_registry の判定関数で spec 生成
- channel は caller 自己申告でなく server (この module) が判別
"""
import json
import uuid
from datetime import datetime

from mcp.server.fastmcp import Context

from core.channel_registry import channel_from_mcp_client
from core.config import BASE_DIR, STATE_FILE
from core.state import load_state


def _client_name_from_ctx(ctx: Context) -> str:
    """Context から MCP client 名を取得。
    initialize handshake の clientInfo.name を信頼する (stdio ローカル接続のみの前提)。
    """
    try:
        params = ctx.session.client_params
        if params is not None:
            return params.clientInfo.name or "unknown"
    except AttributeError:
        pass
    return "unknown"


async def noetic_seed_get_state(
    ctx: Context,
    limit_log: int = 10,
    include_pending: bool = True,
) -> dict:
    """Noetic_seed の現在状態を返す。

    Args:
        limit_log: 返却する recent_log の件数 (default=10)
        include_pending: pending (UPS v2.1 unresolved_intent) を含めるか

    Returns:
        cycle_id / energy / entropy / pressure / recent_log / pending /
        tool_level / self を含む state スナップショット。
    """
    state = load_state()
    log = state.get("log", [])
    return {
        "cycle_id": state.get("cycle_id", 0),
        "energy": state.get("energy", 50),
        "entropy": state.get("entropy", 0.65),
        "pressure": state.get("pressure", 0.0),
        "recent_log": list(log[-limit_log:]) if limit_log > 0 else [],
        "pending": list(state.get("pending", [])) if include_pending else [],
        "tool_level": state.get("tool_level", 0),
        "self": state.get("self", {}),
    }


async def noetic_seed_send_message(ctx: Context, content: str) -> dict:
    """Noetic_seed にメッセージを届ける (非同期刺激)。

    Noetic_seed server が接続元 client_name から channel spec を決定し、
    pending_mcp_inputs.jsonl に append する。main.py が次 cycle 頭で
    consume → WM に channel を動的登録 → log / pending_observations に反映。

    Args:
        content: Noetic_seed に伝えるメッセージ本文

    Returns:
        accepted / observation_id / channel / client_name。
    """
    client_name = _client_name_from_ctx(ctx)
    spec = channel_from_mcp_client(client_name)

    record = {
        "id": f"mcp_{uuid.uuid4().hex[:12]}",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content": content,
        "channel_spec": spec,
        "source": "mcp_client",
        "client_name": client_name,
    }

    input_file = BASE_DIR / "pending_mcp_inputs.jsonl"
    # append-only: 既存 pending と共存
    with input_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "accepted": True,
        "observation_id": record["id"],
        "channel": spec["id"],
        "client_name": client_name,
    }


async def noetic_seed_get_recent_outputs(
    ctx: Context,
    channel: str = "",
    since: str = "",
    limit: int = 20,
) -> dict:
    """Noetic_seed の output_display 履歴を channel で filter して返す。

    Args:
        channel: 取得する channel id (空文字なら接続元 client から推定)
        since: ISO 時刻文字列、これより後の entry のみ (空なら制限なし)
        limit: 返却件数上限 (default=20)

    Returns:
        outputs / channel / total_returned。
    """
    # channel 省略時は接続元から推定
    if not channel:
        client_name = _client_name_from_ctx(ctx)
        channel = channel_from_mcp_client(client_name)["id"]

    state = load_state()
    log = state.get("log", [])

    matched = []
    for entry in reversed(log):
        if entry.get("tool") != "output_display":
            continue
        entry_channel = entry.get("channel", "device")
        if entry_channel != channel:
            continue
        entry_time = entry.get("time", "")
        if since and entry_time <= since:
            break
        matched.append({
            "time": entry_time,
            "content": entry.get("result") or entry.get("content", ""),
            "cycle_id": entry.get("cycle_id", 0),
        })
        if len(matched) >= limit:
            break

    return {
        "outputs": list(reversed(matched)),  # 古い順に並べ替え
        "channel": channel,
        "total_returned": len(matched),
    }


async def noetic_seed_get_wm_snapshot(ctx: Context) -> dict:
    """Noetic_seed の world_model snapshot を返す (entities / channels 全体)。

    起動直後 (v3) は channels={} 空。初の [device_input] or MCP 接続で
    channel が動的に生える。未接続 channel は含まれない。

    Returns:
        entities / channels / version / last_updated。
    """
    state = load_state()
    wm = state.get("world_model") or {}
    return {
        "entities": wm.get("entities", {}),
        "channels": wm.get("channels", {}),
        "version": wm.get("version", 1),
        "last_updated": wm.get("last_updated", ""),
    }
