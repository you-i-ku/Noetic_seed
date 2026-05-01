"""world_model.py テスト (段階11-D Phase 0 後 — channels + render のみ)。

成功条件:
  - init_world_model() が {channels, version, last_updated} 構造を返す
  - **(v3) channels は空から始まる** (bootstrap 撤去)、観察で ensure_channel 経由で生える
  - ensure_channel accessor が冪等
  - channel accessor が存在・不存在を正しく返す
  - render_for_prompt が空 WM と None を正しく扱う
  - render_for_prompt が ensure_channel 後にチャネルセクションを出力する

段階11-D Phase 0 Step 0.1b で entity 関数群テスト (make_fact /
add_or_update_fact / ensure_entity / sync_from_memory_entities 等) を全削除。
entity 概念は B1 完全廃止、対応する module 実装も削除済。

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_world_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.world_model import (
    WM_SCHEMA_VERSION,
    init_world_model,
    get_channel,
    list_channels,
    render_for_prompt,
    observe_channel_activity,
    get_tool_channel,
    ensure_channel,
)
from core.channel_registry import (
    channel_from_device_input,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# init_world_model
# ============================================================

def test_init_world_model_shape():
    print("== init: トップレベル構造 (channels/version/last_updated) ==")
    wm = init_world_model()
    return all([
        _assert(isinstance(wm, dict), "dict 返却"),
        _assert("channels" in wm, "channels key"),
        _assert(wm.get("version") == WM_SCHEMA_VERSION, "version = 1"),
        _assert(isinstance(wm.get("last_updated"), str), "last_updated 文字列"),
    ])


def test_init_channels_empty():
    print("== (v3) init: channels は空から始まる (bootstrap 撤去、観察で生える) ==")
    wm = init_world_model()
    ch = wm["channels"]
    return all([
        _assert(isinstance(ch, dict), "channels dict"),
        _assert(len(ch) == 0, f"channels 空 (actual keys: {list(ch.keys())})"),
    ])


def test_init_idempotent():
    print("== init: 冪等 (2 回呼んでも同じ構造 ※timestamp 除く) ==")
    wm1 = init_world_model()
    wm2 = init_world_model()
    return all([
        _assert(set(wm1["channels"].keys()) == set(wm2["channels"].keys()),
                "channels key 同一"),
        _assert(wm1["version"] == wm2["version"], "version 同一"),
    ])


# ============================================================
# Channel アクセサ
# ============================================================

def test_get_channel_existing():
    print("== get_channel: ensure_channel 後に device を取得 ==")
    wm = init_world_model()
    ensure_channel(wm, **channel_from_device_input())
    ch = get_channel(wm, "device")
    return _assert(ch is not None and ch.get("id") == "device",
                   "device 取得 (ensure_channel 後)")


def test_get_channel_missing():
    print("== get_channel: 存在しない id で None (起動直後は device も未登録) ==")
    wm = init_world_model()
    return all([
        _assert(get_channel(wm, "device") is None, "device 起動直後は未登録"),
        _assert(get_channel(wm, "nonexistent") is None, "不存在で None"),
        _assert(get_channel(None, "device") is None, "wm=None で None"),
    ])


def test_list_channels_empty_and_none():
    print("== list_channels: 起動直後は空、wm=None で空リスト ==")
    wm = init_world_model()
    chs = list_channels(wm)
    return all([
        _assert(len(chs) == 0, f"起動直後 channel 0 件 (v3 bootstrap 撤去) actual: {len(chs)}"),
        _assert(list_channels(None) == [], "None で空リスト"),
    ])


# ============================================================
# render_for_prompt
# ============================================================

def test_render_with_none():
    print("== render: wm=None で空文字 ==")
    return _assert(render_for_prompt(None) == "", "空文字返却")


def test_render_empty_facts():
    print("== render: wm 中身空 → 空文字 ==")
    wm = init_world_model()
    s = render_for_prompt(wm)
    return _assert(s == "", "channels/dispositions/opinions すべて空なら空文字")


def test_render_after_ensure_channel():
    print("== (v3) render: ensure_channel 後にチャネルセクション出現 ==")
    wm = init_world_model()
    ensure_channel(wm, **channel_from_device_input())
    s = render_for_prompt(wm)
    return all([
        _assert("### チャネル" in s, "チャネル heading 出現"),
        _assert("device (direct)" in s, "device 行"),
    ])


# ============================================================
# Channel 活動追跡
# ============================================================

def test_observe_channel_activity():
    print("== observe_channel_activity: ensure_channel 後に count 更新 ==")
    wm = init_world_model()
    ensure_channel(wm, **channel_from_device_input())
    observe_channel_activity(wm, "device")
    observe_channel_activity(wm, "device")
    dev = wm["channels"]["device"]
    return all([
        _assert(dev.get("activity_count") == 2, "count=2"),
        _assert(dev.get("last_activity_at") is not None, "last_activity_at 設定"),
    ])


def test_observe_channel_nonexistent_silent():
    print("== observe_channel_activity: 不明 channel で silent (エラーなし) ==")
    wm = init_world_model()
    try:
        observe_channel_activity(wm, "nonexistent_channel")
        observe_channel_activity(wm, "device")  # 未 ensure なので skip
        observe_channel_activity(None, "device")
        return _assert(True, "例外なく終了")
    except Exception as e:
        return _assert(False, f"例外発生: {e}")


def test_get_tool_channel():
    print("== get_tool_channel: ensure_channel 後に逆引き成立 ==")
    wm = init_world_model()
    empty_result = get_tool_channel(wm, "output_display")
    ensure_channel(wm, **channel_from_device_input())
    return all([
        _assert(empty_result is None,
                "起動直後は output_display → None (channel 未登録)"),
        _assert(get_tool_channel(wm, "output_display") == "device",
                "ensure 後 output_display → device"),
        _assert(get_tool_channel(wm, "[device_input]") == "device",
                "ensure 後 [device_input] → device"),
        _assert(get_tool_channel(wm, "unknown_tool") is None,
                "tool なし → None"),
        _assert(get_tool_channel(None, "output_display") is None,
                "wm=None → None"),
    ])


# ============================================================
# 段階6-C v3: ensure_channel (動的 channel 登録)
# ============================================================

def test_ensure_channel_creates_new():
    print("== (v3) ensure_channel: 未存在 id で新規作成、channels 登録、last_updated 更新 ==")
    wm = init_world_model()
    wm["last_updated"] = "2000-01-01 00:00:00"
    ch = ensure_channel(wm, id="claude", type="social",
                        tools_in=["[claude_input]"],
                        tools_out=["output_display"])
    return all([
        _assert(ch["id"] == "claude", "返却 id=claude"),
        _assert(ch["type"] == "social", "type=social"),
        _assert(ch["tools_in"] == ["[claude_input]"], "tools_in"),
        _assert(ch["tools_out"] == ["output_display"], "tools_out"),
        _assert("claude" in wm["channels"], "wm.channels に登録"),
        _assert(wm["last_updated"] != "2000-01-01 00:00:00",
                "last_updated 更新"),
    ])


def test_ensure_channel_idempotent():
    print("== (v3) ensure_channel: 既存 id で 2 回目は既存を返却、重複作成なし ==")
    wm = init_world_model()
    ch1 = ensure_channel(wm, **channel_from_device_input())
    ch2 = ensure_channel(wm, id="device", type="social",
                         tools_in=["something_else"], tools_out=[])
    return all([
        _assert(ch1 is ch2, "同一インスタンス返却"),
        _assert(ch2["type"] == "direct", "既存の type 維持 (上書きしない)"),
        _assert("output_display" in ch2["tools_out"], "既存の tools_out 維持"),
        _assert(len(wm["channels"]) == 1, "channels に 1 件のみ"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("init: トップレベル構造", test_init_world_model_shape),
        ("(v3) init: channels 空", test_init_channels_empty),
        ("init: 冪等", test_init_idempotent),
        ("get_channel: ensure 後に存在", test_get_channel_existing),
        ("get_channel: 起動直後 device も None", test_get_channel_missing),
        ("list_channels: 空 / None", test_list_channels_empty_and_none),
        ("render: None", test_render_with_none),
        ("render: wm 中身空 → 空文字", test_render_empty_facts),
        ("(v3) render: ensure 後に channel 出現", test_render_after_ensure_channel),
        ("observe_channel_activity: 更新", test_observe_channel_activity),
        ("observe_channel_activity: silent skip", test_observe_channel_nonexistent_silent),
        ("get_tool_channel: 逆引き", test_get_tool_channel),
        ("(v3) ensure_channel: 新規作成", test_ensure_channel_creates_new),
        ("(v3) ensure_channel: 冪等", test_ensure_channel_idempotent),
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
