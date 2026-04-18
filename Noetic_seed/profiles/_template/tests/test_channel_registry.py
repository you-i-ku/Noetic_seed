"""channel_registry.py テスト (段階6-C v3 動的 channel 基盤)。

成功条件:
  - channel_from_device_input() が device spec を返す (id=device, type=direct)
  - channel_from_mcp_client("claude-code") が claude spec を返す
  - 大文字小文字混在でも claude に解決される
  - 未知 client 名は mcp_<safe_name> 形式で汎用 spec 生成
  - 空文字列は mcp_unknown に fallback
  - 長い name は 20 文字に切り詰め

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_channel_registry.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.channel_registry import (
    channel_from_device_input,
    channel_from_mcp_client,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# channel_from_device_input
# ============================================================

def test_channel_from_device_input_shape():
    print("== channel_from_device_input: spec の必須 field が揃う ==")
    spec = channel_from_device_input()
    return all([
        _assert(isinstance(spec, dict), "dict 返却"),
        _assert("id" in spec, "id"),
        _assert("type" in spec, "type"),
        _assert("tools_in" in spec, "tools_in"),
        _assert("tools_out" in spec, "tools_out"),
    ])


def test_channel_from_device_input_values():
    print("== channel_from_device_input: id=device, type=direct, output_display 含む ==")
    spec = channel_from_device_input()
    return all([
        _assert(spec["id"] == "device", "id=device"),
        _assert(spec["type"] == "direct", "type=direct"),
        _assert("[device_input]" in spec["tools_in"], "[device_input] in tools_in"),
        _assert("output_display" in spec["tools_out"], "output_display in tools_out"),
    ])


# ============================================================
# channel_from_mcp_client
# ============================================================

def test_channel_from_mcp_client_claude():
    print("== channel_from_mcp_client('claude-code'): claude channel spec ==")
    spec = channel_from_mcp_client("claude-code")
    return all([
        _assert(spec["id"] == "claude", "id=claude"),
        _assert(spec["type"] == "social", "type=social"),
        _assert("[claude_input]" in spec["tools_in"], "[claude_input] in tools_in"),
        _assert("output_display" in spec["tools_out"], "output_display in tools_out"),
    ])


def test_channel_from_mcp_client_case_insensitive():
    print("== channel_from_mcp_client: 大文字小文字混在でも claude 解決 ==")
    return all([
        _assert(channel_from_mcp_client("Claude-Code")["id"] == "claude",
                "Claude-Code → claude"),
        _assert(channel_from_mcp_client("CLAUDE")["id"] == "claude",
                "CLAUDE → claude"),
        _assert(channel_from_mcp_client("ClaudeDesktop")["id"] == "claude",
                "ClaudeDesktop → claude (部分一致)"),
    ])


def test_channel_from_mcp_client_unknown_fallback():
    print("== channel_from_mcp_client: 未知 client は mcp_<safe> 形式 ==")
    spec = channel_from_mcp_client("discord-bot")
    return all([
        _assert(spec["id"] == "mcp_discord-bot", f"id={spec['id']}"),
        _assert(spec["type"] == "social", "type=social (汎用)"),
        _assert("[discord-bot_input]" in spec["tools_in"],
                "[discord-bot_input] in tools_in"),
        _assert("output_display" in spec["tools_out"],
                "output_display in tools_out"),
    ])


def test_channel_from_mcp_client_empty_string():
    print("== channel_from_mcp_client: 空文字列で mcp_unknown ==")
    spec = channel_from_mcp_client("")
    return all([
        _assert(spec["id"] == "mcp_unknown", f"id={spec['id']}"),
        _assert(spec["type"] == "social", "type=social"),
    ])


def test_channel_from_mcp_client_long_name_truncated():
    print("== channel_from_mcp_client: 長い name は 20 文字に切り詰め ==")
    # 50 文字の name (小文字統一後) を投げる
    long_name = "a" * 50
    spec = channel_from_mcp_client(long_name)
    # mcp_ + safe_name (max 20 文字) を期待
    safe_part = spec["id"][4:]  # "mcp_" を除いた残り
    return all([
        _assert(spec["id"].startswith("mcp_"), f"mcp_ prefix (id={spec['id']})"),
        _assert(len(safe_part) <= 20, f"safe_name <= 20 chars (actual: {len(safe_part)})"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("device_input: spec shape", test_channel_from_device_input_shape),
        ("device_input: 値 (id=device, direct)", test_channel_from_device_input_values),
        ("mcp_client: claude-code → claude", test_channel_from_mcp_client_claude),
        ("mcp_client: 大小混在 OK", test_channel_from_mcp_client_case_insensitive),
        ("mcp_client: 未知 → mcp_<safe>", test_channel_from_mcp_client_unknown_fallback),
        ("mcp_client: 空文字列 → mcp_unknown", test_channel_from_mcp_client_empty_string),
        ("mcp_client: 長い name 切り詰め", test_channel_from_mcp_client_long_name_truncated),
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
