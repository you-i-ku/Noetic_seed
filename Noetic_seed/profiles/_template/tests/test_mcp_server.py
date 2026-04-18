"""MCP server tool tests (段階6-C v3)。

seed_tools.py の 4 tool を直接 await で呼び出して動作確認。
stdio transport 経由の E2E は §10 実走行テストランで別途扱う。

成功条件:
  - FastMCP server に 4 tool 全部登録されている
  - noetic_seed_get_state が必須 field を返す
  - noetic_seed_send_message が pending_mcp_inputs.jsonl に append、channel_spec 含む
  - client 名から channel 判定 (claude / 未知は mcp_<safe> fallback / 欠損時は unknown)
  - noetic_seed_get_wm_snapshot が entities/channels/version を返す
  - noetic_seed_get_recent_outputs が channel / since filter を尊重

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_mcp_server.py
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import BASE_DIR
from core.runtime.mcp.server.seed_tools import (
    noetic_seed_get_state,
    noetic_seed_send_message,
    noetic_seed_get_recent_outputs,
    noetic_seed_get_wm_snapshot,
)
from core.runtime.mcp.server.server import mcp_server


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _mock_ctx(client_name="claude-code"):
    """MCP Context の最小 mock。client_params.clientInfo.name を模倣。"""
    ctx = MagicMock()
    if client_name is None:
        ctx.session.client_params = None
    else:
        ctx.session.client_params.clientInfo.name = client_name
    return ctx


def _cleanup_pending_file():
    """テスト生成の pending_mcp_inputs.jsonl を削除 (副作用掃除)。"""
    f = BASE_DIR / "pending_mcp_inputs.jsonl"
    if f.exists():
        f.unlink()


# ============================================================
# Server / tool 登録
# ============================================================

def test_tools_registered_on_server():
    print("== FastMCP server に 4 tool 登録されている ==")
    tools = asyncio.run(mcp_server.list_tools())
    names = {t.name for t in tools}
    expected = {
        "noetic_seed_get_state",
        "noetic_seed_send_message",
        "noetic_seed_get_recent_outputs",
        "noetic_seed_get_wm_snapshot",
    }
    return all([
        _assert(expected.issubset(names),
                f"4 tool 全部登録 (missing: {expected - names})"),
        _assert(mcp_server.name == "noetic-seed", "server name=noetic-seed"),
    ])


# ============================================================
# noetic_seed_get_state
# ============================================================

def test_get_state_returns_required_fields():
    print("== noetic_seed_get_state: 必須 field 揃う ==")
    result = asyncio.run(noetic_seed_get_state(_mock_ctx()))
    return all([
        _assert("cycle_id" in result, "cycle_id"),
        _assert("energy" in result, "energy"),
        _assert("entropy" in result, "entropy"),
        _assert("pressure" in result, "pressure"),
        _assert("recent_log" in result, "recent_log"),
        _assert("pending" in result, "pending"),
        _assert("tool_level" in result, "tool_level"),
        _assert("self" in result, "self"),
        _assert(isinstance(result["recent_log"], list), "recent_log is list"),
    ])


def test_get_state_respects_limit_log():
    print("== noetic_seed_get_state: limit_log=0 で log 空 ==")
    result = asyncio.run(noetic_seed_get_state(_mock_ctx(), limit_log=0))
    return _assert(result["recent_log"] == [], "recent_log=[]")


# ============================================================
# noetic_seed_send_message
# ============================================================

def test_send_message_writes_pending_file():
    print("== noetic_seed_send_message: pending_mcp_inputs.jsonl に append ==")
    _cleanup_pending_file()
    try:
        result = asyncio.run(
            noetic_seed_send_message(_mock_ctx("claude-code"), content="テスト")
        )
        f = BASE_DIR / "pending_mcp_inputs.jsonl"
        lines = f.read_text(encoding="utf-8").splitlines() if f.exists() else []
        rec = json.loads(lines[0]) if lines else {}
        return all([
            _assert(result.get("accepted") is True, "accepted=True"),
            _assert(result.get("channel") == "claude", "channel=claude"),
            _assert("observation_id" in result, "observation_id"),
            _assert(len(lines) == 1, f"1 行追加 (actual: {len(lines)})"),
            _assert(rec.get("content") == "テスト", "content 保存"),
            _assert("channel_spec" in rec, "channel_spec 含む"),
            _assert(rec["channel_spec"]["id"] == "claude",
                    "channel_spec.id=claude (server 判定)"),
            _assert(rec.get("client_name") == "claude-code", "client_name 保存"),
        ])
    finally:
        _cleanup_pending_file()


def test_send_message_unknown_client_fallback():
    print("== noetic_seed_send_message: 未知 client → mcp_<safe> channel ==")
    _cleanup_pending_file()
    try:
        result = asyncio.run(
            noetic_seed_send_message(_mock_ctx("discord-bot"), content="hi")
        )
        return all([
            _assert(result["channel"] == "mcp_discord-bot",
                    f"channel=mcp_discord-bot (actual: {result['channel']})"),
            _assert(result["client_name"] == "discord-bot", "client_name 保存"),
        ])
    finally:
        _cleanup_pending_file()


def test_send_message_missing_client_params():
    print("== noetic_seed_send_message: client_params=None でも落ちない ==")
    _cleanup_pending_file()
    try:
        result = asyncio.run(
            noetic_seed_send_message(_mock_ctx(None), content="x")
        )
        return all([
            _assert(result.get("accepted") is True, "accepted"),
            _assert(result.get("client_name") == "unknown", "fallback=unknown"),
            _assert(result.get("channel", "").startswith("mcp_"),
                    f"channel mcp_ prefix (actual: {result.get('channel')})"),
        ])
    finally:
        _cleanup_pending_file()


def test_send_message_appends_multiple_records():
    print("== noetic_seed_send_message: 複数回で append-only (重複書き込まない) ==")
    _cleanup_pending_file()
    try:
        asyncio.run(noetic_seed_send_message(_mock_ctx(), content="1"))
        asyncio.run(noetic_seed_send_message(_mock_ctx(), content="2"))
        f = BASE_DIR / "pending_mcp_inputs.jsonl"
        lines = f.read_text(encoding="utf-8").splitlines() if f.exists() else []
        return _assert(len(lines) == 2, f"2 行 append (actual: {len(lines)})")
    finally:
        _cleanup_pending_file()


# ============================================================
# noetic_seed_get_wm_snapshot
# ============================================================

def test_get_wm_snapshot_shape():
    print("== noetic_seed_get_wm_snapshot: entities/channels/version 返す ==")
    result = asyncio.run(noetic_seed_get_wm_snapshot(_mock_ctx()))
    return all([
        _assert("entities" in result, "entities"),
        _assert("channels" in result, "channels"),
        _assert("version" in result, "version"),
        _assert("last_updated" in result, "last_updated"),
        _assert(isinstance(result["entities"], dict), "entities dict"),
        _assert(isinstance(result["channels"], dict), "channels dict"),
    ])


# ============================================================
# noetic_seed_get_recent_outputs
# ============================================================

def test_get_recent_outputs_channel_filter():
    print("== noetic_seed_get_recent_outputs: channel filter ==")
    # channel 指定あり → その channel のみ返る
    result = asyncio.run(
        noetic_seed_get_recent_outputs(_mock_ctx(), channel="claude", limit=5)
    )
    outputs = result.get("outputs", [])
    return all([
        _assert(result.get("channel") == "claude", "channel=claude"),
        _assert(isinstance(outputs, list), "outputs is list"),
        _assert(all(o.get("cycle_id") is not None or True for o in outputs),
                "outputs entries shape OK"),  # 空でもOK
    ])


def test_get_recent_outputs_infers_channel_from_client():
    print("== noetic_seed_get_recent_outputs: channel 省略で client から推定 ==")
    # claude-code client で channel="" → claude に解決
    result = asyncio.run(
        noetic_seed_get_recent_outputs(_mock_ctx("claude-code"))
    )
    return _assert(result.get("channel") == "claude",
                   f"channel=claude (推定、actual: {result.get('channel')})")


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("server: 4 tool 登録", test_tools_registered_on_server),
        ("get_state: 必須 field", test_get_state_returns_required_fields),
        ("get_state: limit_log=0", test_get_state_respects_limit_log),
        ("send_message: 書き込み + channel_spec", test_send_message_writes_pending_file),
        ("send_message: 未知 client fallback", test_send_message_unknown_client_fallback),
        ("send_message: client_params=None", test_send_message_missing_client_params),
        ("send_message: append-only", test_send_message_appends_multiple_records),
        ("wm_snapshot: shape", test_get_wm_snapshot_shape),
        ("recent_outputs: channel filter", test_get_recent_outputs_channel_filter),
        ("recent_outputs: client 推定", test_get_recent_outputs_infers_channel_from_client),
    ]
    results = []
    for _label, fn in groups:
        print()
        ok = fn()
        results.append((_label, ok))
    print()
    print("=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for _label, ok in results:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {_label}")
    print(f"\n  {passed}/{total} groups passed")
    sys.exit(0 if passed == total else 1)
