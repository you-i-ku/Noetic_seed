"""world_model.py テスト (段階2 ミニマル)。

成功条件:
  - init_world_model() が WorldModel 構造を返す
  - ent_self が構造的スロットとして予約されている
  - device/elyth/x/internal 4 channels が bootstrap されている
  - 特に device.tools_out に output_display が含まれる (回帰ガード)
  - accessor が存在・不存在を正しく返す
  - render_for_prompt が空 WM と None を正しく扱う

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
    get_entity,
    get_channel,
    list_entities,
    list_channels,
    render_for_prompt,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# init_world_model
# ============================================================

def test_init_world_model_shape():
    print("== init: トップレベル構造 (entities/channels/version/last_updated) ==")
    wm = init_world_model()
    return all([
        _assert(isinstance(wm, dict), "dict 返却"),
        _assert("entities" in wm, "entities key"),
        _assert("channels" in wm, "channels key"),
        _assert(wm.get("version") == WM_SCHEMA_VERSION, "version = 1"),
        _assert(isinstance(wm.get("last_updated"), str), "last_updated 文字列"),
    ])


def test_init_includes_ent_self():
    print("== init: ent_self 予約 (name='self' 固定 / facts=[]) ==")
    wm = init_world_model()
    ent = wm["entities"].get("ent_self")
    return all([
        _assert(ent is not None, "ent_self 存在"),
        _assert(ent.get("id") == "ent_self", "id"),
        _assert(ent.get("name") == "self", "name='self' 固定 (seed 透過なし)"),
        _assert(ent.get("facts") == [], "facts=[] (段階2 空)"),
        _assert("created_at" in ent, "created_at"),
        _assert("updated_at" in ent, "updated_at"),
    ])


def test_init_bootstrap_four_channels():
    print("== init: device/elyth/x/internal 4 channels bootstrap ==")
    wm = init_world_model()
    ch = wm["channels"]
    return all([
        _assert("device" in ch, "device"),
        _assert("elyth" in ch, "elyth"),
        _assert("x" in ch, "x"),
        _assert("internal" in ch, "internal"),
        _assert(ch["device"].get("type") == "direct", "device.type=direct"),
        _assert(ch["elyth"].get("type") == "social", "elyth.type=social"),
        _assert(ch["x"].get("type") == "social", "x.type=social"),
        _assert(ch["internal"].get("type") == "self", "internal.type=self"),
    ])


def test_device_channel_tools_out_includes_output_display():
    print("== init: device.tools_out に output_display 含む (回帰ガード) ==")
    wm = init_world_model()
    device = wm["channels"]["device"]
    return all([
        _assert("output_display" in device.get("tools_out", []),
                "output_display 含む"),
        _assert("[device_input]" in device.get("tools_in", []),
                "[device_input] が tools_in"),
    ])


def test_init_idempotent():
    print("== init: 冪等 (2 回呼んでも同じ構造 ※timestamp 除く) ==")
    wm1 = init_world_model()
    wm2 = init_world_model()
    return all([
        _assert(set(wm1["entities"].keys()) == set(wm2["entities"].keys()),
                "entities key 同一"),
        _assert(set(wm1["channels"].keys()) == set(wm2["channels"].keys()),
                "channels key 同一"),
        _assert(wm1["version"] == wm2["version"], "version 同一"),
    ])


# ============================================================
# アクセサ
# ============================================================

def test_get_entity_existing():
    print("== get_entity: 存在する ent_self を取得 ==")
    wm = init_world_model()
    ent = get_entity(wm, "ent_self")
    return _assert(ent is not None and ent.get("id") == "ent_self",
                   "ent_self 取得")


def test_get_entity_missing():
    print("== get_entity: 存在しない id で None ==")
    wm = init_world_model()
    return all([
        _assert(get_entity(wm, "ent_nobody") is None, "不存在で None"),
        _assert(get_entity(None, "ent_self") is None, "wm=None で None"),
    ])


def test_get_channel_existing():
    print("== get_channel: device を取得 ==")
    wm = init_world_model()
    ch = get_channel(wm, "device")
    return _assert(ch is not None and ch.get("id") == "device",
                   "device 取得")


def test_get_channel_missing():
    print("== get_channel: 存在しない id で None ==")
    wm = init_world_model()
    return all([
        _assert(get_channel(wm, "nonexistent") is None, "不存在で None"),
        _assert(get_channel(None, "device") is None, "wm=None で None"),
    ])


def test_list_entities_and_channels():
    print("== list_entities / list_channels: 全件返却 ==")
    wm = init_world_model()
    ents = list_entities(wm)
    chs = list_channels(wm)
    return all([
        _assert(len(ents) == 1, "entity 1 件 (ent_self)"),
        _assert(len(chs) == 4, "channel 4 件"),
        _assert(list_entities(None) == [], "None で空リスト"),
        _assert(list_channels(None) == [], "None で空リスト"),
    ])


# ============================================================
# render_for_prompt
# ============================================================

def test_render_with_none():
    print("== render: wm=None で空文字 ==")
    return _assert(render_for_prompt(None) == "", "空文字返却")


def test_render_empty_facts():
    print("== render: facts 空 → '(まだ観測されていない)' 表示 ==")
    wm = init_world_model()
    s = render_for_prompt(wm)
    return all([
        _assert("## 世界モデル" in s, "セクション heading"),
        _assert("### チャネル" in s, "チャネル heading"),
        _assert("device (direct)" in s, "device 行"),
        _assert("elyth (social)" in s, "elyth 行"),
        _assert("### 観測された存在" in s, "存在 heading"),
        _assert("まだ観測されていない" in s, "未観測メッセージ"),
    ])


def test_render_with_facts():
    print("== render: facts 入り entity が表示される ==")
    wm = init_world_model()
    # 段階3 の事前シミュレーションとして手動で facts を注入
    wm["entities"]["ent_yuu"] = {
        "id": "ent_yuu",
        "name": "ゆう",
        "facts": [
            {"key": "primary_channel", "value": "device"},
            {"key": "role", "value": "developer"},
        ],
        "created_at": "now",
        "updated_at": "now",
    }
    s = render_for_prompt(wm)
    return all([
        _assert("ゆう" in s, "ゆう name 表示"),
        _assert("primary_channel=device" in s, "fact 1 表示"),
        _assert("role=developer" in s, "fact 2 表示"),
        _assert("まだ観測されていない" not in s,
                "facts ありなら未観測メッセージは出ない"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("init: トップレベル構造", test_init_world_model_shape),
        ("init: ent_self 予約", test_init_includes_ent_self),
        ("init: 4 channels bootstrap", test_init_bootstrap_four_channels),
        ("init: device.tools_out 回帰ガード", test_device_channel_tools_out_includes_output_display),
        ("init: 冪等", test_init_idempotent),
        ("get_entity: 存在", test_get_entity_existing),
        ("get_entity: 不存在", test_get_entity_missing),
        ("get_channel: 存在", test_get_channel_existing),
        ("get_channel: 不存在", test_get_channel_missing),
        ("list_entities/channels", test_list_entities_and_channels),
        ("render: None", test_render_with_none),
        ("render: 空 facts", test_render_empty_facts),
        ("render: facts 入り", test_render_with_facts),
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
