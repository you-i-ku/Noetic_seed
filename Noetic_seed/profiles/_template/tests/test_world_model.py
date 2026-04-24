"""world_model.py テスト (段階2-3 + 段階6-C v3 動的 channel)。

成功条件:
  - init_world_model() が WorldModel 構造を返す
  - ent_self が構造的スロットとして予約されている
  - **(v3) channels は空から始まる** (bootstrap 撤去)、観察で ensure_channel 経由で生える
  - ensure_channel accessor が冪等かつ ensure_entity と対称
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
    # 段階3
    make_fact,
    update_fact_confidence,
    find_fact,
    add_or_update_fact,
    observe_channel_activity,
    get_tool_channel,
    ensure_entity,
    migrate_entity_fields,
    sync_from_memory_entities,
    # 段階6-C v3
    ensure_channel,
)
from core.channel_registry import (
    channel_from_device_input,
    channel_from_mcp_client,
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


def test_list_entities_and_channels():
    print("== list_entities / list_channels: 全件返却 (起動直後 channels 空) ==")
    wm = init_world_model()
    ents = list_entities(wm)
    chs = list_channels(wm)
    return all([
        _assert(len(ents) == 1, "entity 1 件 (ent_self)"),
        _assert(len(chs) == 0, f"channel 0 件 (v3 bootstrap 撤去) actual: {len(chs)}"),
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
# 段階3: Fact schema + β+ 更新
# ============================================================

def test_make_fact_structure():
    print("== make_fact: 8 field 揃う、confidence default 0.7 ==")
    f = make_fact("role", "developer")
    return all([
        _assert(f["key"] == "role", "key"),
        _assert(f["value"] == "developer", "value"),
        _assert(f["confidence"] == 0.7, "confidence default"),
        _assert(f["valid_to"] is None, "valid_to=None (現行)"),
        _assert(f["observation_count"] == 1, "observation_count=1"),
        _assert(isinstance(f["valid_from"], str), "valid_from 文字列"),
        _assert(isinstance(f["learned_at"], str), "learned_at 文字列"),
        _assert(isinstance(f["last_observed_at"], str), "last_observed_at 文字列"),
    ])


def test_update_fact_confidence_match_converges():
    print("== update_fact_confidence: match を 40 回で > 0.9 に漸近 ==")
    # 解析: conf_n = 1 - 0.95^n * (1 - conf_0)。
    # 0.5 から 0.9 到達には n ≈ 32、余裕を見て 40 回。
    f = make_fact("x", "v", confidence=0.5)
    for _ in range(40):
        update_fact_confidence(f, True)
    return all([
        _assert(f["confidence"] > 0.9, f"0.5 → {f['confidence']:.3f} > 0.9"),
        _assert(f["confidence"] <= 1.0, "上限 1.0"),
        _assert(f["observation_count"] == 41, "count=41 (初期1+追加40)"),
    ])


def test_update_fact_confidence_mismatch_drops():
    print("== update_fact_confidence: mismatch で -0.15 ==")
    f = make_fact("x", "v", confidence=0.7)
    update_fact_confidence(f, False)
    return _assert(abs(f["confidence"] - 0.55) < 1e-9,
                   f"0.7 → {f['confidence']:.3f} (期待 0.55)")


def test_update_fact_confidence_lower_bound():
    print("== update_fact_confidence: 下限 0.0 を超えない ==")
    f = make_fact("x", "v", confidence=0.1)
    update_fact_confidence(f, False)
    return _assert(f["confidence"] == 0.0, f"下限 0 (actual: {f['confidence']})")


def test_find_fact():
    print("== find_fact: 存在 / 不存在 / frozen のスキップ ==")
    ent = {"facts": [
        {"key": "role", "value": "dev", "valid_to": None},
        {"key": "old", "value": "legacy", "valid_to": "2026-01-01"},
    ]}
    return all([
        _assert(find_fact(ent, "role") is not None, "現行 fact 取得"),
        _assert(find_fact(ent, "old") is None, "frozen fact はスキップ"),
        _assert(find_fact(ent, "nonexistent") is None, "不存在で None"),
        _assert(find_fact(None, "role") is None, "entity=None で None"),
    ])


def test_add_or_update_fact_new():
    print("== add_or_update_fact: 未存在 key を新規追加 ==")
    ent = {"facts": [], "updated_at": "old"}
    f = add_or_update_fact(ent, "role", "developer")
    return all([
        _assert(len(ent["facts"]) == 1, "1 件追加"),
        _assert(f["key"] == "role" and f["value"] == "developer", "値"),
        _assert(ent["updated_at"] != "old", "updated_at 更新"),
    ])


def test_add_or_update_fact_matching_value():
    print("== add_or_update_fact: value 一致で β+ ==")
    ent = {"facts": [], "updated_at": "old"}
    add_or_update_fact(ent, "role", "developer")  # confidence=0.7, count=1
    f = add_or_update_fact(ent, "role", "developer")  # β+
    return all([
        _assert(len(ent["facts"]) == 1, "facts 増えない"),
        _assert(f["observation_count"] == 2, "count=2"),
        _assert(f["confidence"] > 0.7, "confidence 上昇"),
    ])


def test_add_or_update_fact_differing_value():
    print("== add_or_update_fact: value 異なる → bitemporal 更新 ==")
    ent = {"facts": [], "updated_at": "old"}
    add_or_update_fact(ent, "role", "developer")
    add_or_update_fact(ent, "role", "scientist")  # 違う値
    facts = ent["facts"]
    return all([
        _assert(len(facts) == 2, "旧 fact 保持 + 新 fact 追加"),
        _assert(facts[0]["valid_to"] is not None, "旧 fact 凍結"),
        _assert(facts[0]["confidence"] < 0.7, "旧 fact 信頼度降下"),
        _assert(facts[1]["value"] == "scientist", "新 fact value"),
        _assert(facts[1]["valid_to"] is None, "新 fact 現行"),
    ])


# ============================================================
# 段階3: Channel 活動追跡
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
        # (v3) 起動直後は device も未登録、silent skip される
        observe_channel_activity(wm, "nonexistent_channel")
        observe_channel_activity(wm, "device")  # 未 ensure なので skip
        observe_channel_activity(None, "device")
        return _assert(True, "例外なく終了")
    except Exception as e:
        return _assert(False, f"例外発生: {e}")


def test_get_tool_channel():
    print("== get_tool_channel: ensure_channel 後に逆引き成立 ==")
    wm = init_world_model()
    # (v3) 起動直後は channel 空、逆引きは全部 None
    empty_result = get_tool_channel(wm, "output_display")
    # device channel を ensure してから逆引き
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
# 段階3: Entity 作成 + migration
# ============================================================

def test_ensure_entity_creates():
    print("== ensure_entity: 未存在なら新規作成、段階3 schema 全 field ==")
    wm = init_world_model()
    ent = ensure_entity(wm, "ent_yuu", "ゆう")
    return all([
        _assert(ent["id"] == "ent_yuu", "id"),
        _assert(ent["name"] == "ゆう", "name"),
        _assert(ent["facts"] == [], "facts=[]"),
        _assert(ent["aliases"] == [], "aliases=[]"),
        _assert(ent["channels"] == [], "channels=[]"),
        _assert(ent["last_seen"] is None, "last_seen=None"),
        _assert("ent_yuu" in wm["entities"], "wm に登録"),
    ])


def test_ensure_entity_returns_existing():
    print("== ensure_entity: 既存なら返却のみ (上書きしない) ==")
    wm = init_world_model()
    # ent_self が既存 (段階2 初期化で作られる)
    ent = ensure_entity(wm, "ent_self", "something_else")
    return all([
        _assert(ent["id"] == "ent_self", "既存 id"),
        _assert(ent["name"] == "self", "name は上書きしない"),
    ])


def test_migrate_entity_fields_idempotent():
    print("== migrate_entity_fields: 冪等、既存 field 上書きしない ==")
    ent = {"id": "x", "name": "x", "facts": [], "aliases": ["a"]}
    migrate_entity_fields(ent)
    migrate_entity_fields(ent)  # 2 回目
    return all([
        _assert(ent["aliases"] == ["a"], "aliases 既存保持"),
        _assert(ent["channels"] == [], "channels 追加"),
        _assert(ent["last_seen"] is None, "last_seen 追加"),
    ])


# ============================================================
# 段階6-C v3: ensure_channel (entity と対称な動的 channel 登録)
# ============================================================

def test_ensure_channel_creates_new():
    print("== (v3) ensure_channel: 未存在 id で新規作成、channels 登録、last_updated 更新 ==")
    wm = init_world_model()
    before = wm["last_updated"]
    # ちょっと待つ代わりに last_updated を過去値に書き換えて検出可能にする
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
    # 2 回目は spec を変えても既存を返す (上書きしない、ensure_entity と対称)
    ch2 = ensure_channel(wm, id="device", type="social",
                         tools_in=["something_else"], tools_out=[])
    return all([
        _assert(ch1 is ch2, "同一インスタンス返却"),
        _assert(ch2["type"] == "direct", "既存の type 維持 (上書きしない)"),
        _assert("output_display" in ch2["tools_out"], "既存の tools_out 維持"),
        _assert(len(wm["channels"]) == 1, "channels に 1 件のみ"),
    ])


def test_ensure_channel_entity_symmetry():
    print("== (v3) ensure_channel: ensure_entity と対称な動作 ==")
    wm = init_world_model()
    # ensure_entity は既存 ent_self を返す (上書きしない)
    ent = ensure_entity(wm, "ent_self", "something_else")
    # ensure_channel も同じ挙動で、2 回呼んでも新規作成しない
    ch_a = ensure_channel(wm, id="test_ch", type="direct")
    ch_b = ensure_channel(wm, id="test_ch", type="social")
    return all([
        _assert(ent["name"] == "self", "ensure_entity: name 上書きされない"),
        _assert(ch_a is ch_b, "ensure_channel: 同一インスタンス"),
        _assert(ch_b["type"] == "direct", "ensure_channel: 既存 type 維持"),
    ])


# ============================================================
# 段階3: C-gradual 同期
# ============================================================

def test_sync_from_memory_entities_creates_new():
    print("== sync: memory/entity レコード → WM entity 新規作成 ==")
    wm = init_world_model()
    records = [
        {"id": "mem_1", "content": "ゆうは iku の開発者",
         "metadata": {"entity_name": "ゆう"},
         "created_at": "2026-04-10 10:00:00",
         "updated_at": "2026-04-10 10:00:00"},
        {"id": "mem_2", "content": "Claude は助けてくれる",
         "metadata": {"entity_name": "Claude"},
         "created_at": "2026-04-11 10:00:00",
         "updated_at": "2026-04-11 10:00:00"},
    ]
    created = sync_from_memory_entities(wm, records)
    return all([
        _assert(created == 2, "2 件新規作成"),
        _assert(any(e["name"] == "ゆう" for e in wm["entities"].values()),
                "ゆう entity 作成"),
        _assert(any(e["name"] == "Claude" for e in wm["entities"].values()),
                "Claude entity 作成"),
    ])


def test_sync_appends_description_fact():
    print("== sync: 最新レコード content が description fact として入る ==")
    wm = init_world_model()
    records = [
        {"id": "mem_old", "content": "古い情報",
         "metadata": {"entity_name": "A"},
         "updated_at": "2026-01-01 00:00:00"},
        {"id": "mem_new", "content": "新しい情報",
         "metadata": {"entity_name": "A"},
         "updated_at": "2026-04-18 00:00:00"},
    ]
    sync_from_memory_entities(wm, records)
    ent_a = [e for e in wm["entities"].values() if e["name"] == "A"][0]
    desc_fact = find_fact(ent_a, "description")
    return all([
        _assert(desc_fact is not None, "description fact 存在"),
        _assert(desc_fact["value"] == "新しい情報",
                "最新 content が使われる"),
    ])


def test_sync_empty_records():
    print("== sync: 空レコードで 0 返却、例外なし ==")
    wm = init_world_model()
    created = sync_from_memory_entities(wm, [])
    return _assert(created == 0, "0 件")


def test_sync_respects_limit():
    print("== sync: limit=2 で最初の 2 件のみ処理 ==")
    wm = init_world_model()
    records = [
        {"id": f"mem_{i}", "content": f"content {i}",
         "metadata": {"entity_name": f"Name{i}"},
         "updated_at": f"2026-04-{10+i:02d} 00:00:00"}
        for i in range(5)
    ]
    created = sync_from_memory_entities(wm, records, limit=2)
    return _assert(created == 2, f"2 件作成 (actual: {created})")


def test_sync_resolver_merges_similar_names():
    print("== sync: 段階4 resolver で ゆう と YOU が merge される ==")
    import math

    def mock_embed(texts):
        vecs = {"ゆう": [1.0, 0.0, 0.0], "YOU": [0.97, 0.05, 0.0]}
        return [vecs.get(t, [0.0] * 3) for t in texts]

    def mock_cosine(a, b):
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)

    wm = init_world_model()
    records = [
        {"id": "m1", "content": "開発者", "metadata": {"entity_name": "ゆう"},
         "updated_at": "2026-04-10 00:00:00"},
        {"id": "m2", "content": "同一人物", "metadata": {"entity_name": "YOU"},
         "updated_at": "2026-04-11 00:00:00"},
    ]
    created = sync_from_memory_entities(
        wm, records, embed_fn=mock_embed, cosine_fn=mock_cosine,
    )
    # ゆう と YOU が merge されるので新規は 1 つだけ
    return all([
        _assert(created == 1, f"新規 1 件 (actual: {created})"),
        _assert(
            any("YOU" in e.get("aliases", []) for e in wm["entities"].values()),
            "YOU が alias として登録",
        ),
    ])


def test_sync_without_embed_fn_no_merge():
    print("== sync: embed_fn なしで ゆう と YOU は別 entity ==")
    wm = init_world_model()
    records = [
        {"id": "m1", "content": "x", "metadata": {"entity_name": "ゆう"},
         "updated_at": "2026-04-10 00:00:00"},
        {"id": "m2", "content": "y", "metadata": {"entity_name": "YOU"},
         "updated_at": "2026-04-11 00:00:00"},
    ]
    created = sync_from_memory_entities(wm, records)  # embed_fn=None
    return _assert(created == 2, f"exact のみなので 2 件 (actual: {created})")


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        # 段階2
        ("init: トップレベル構造", test_init_world_model_shape),
        ("init: ent_self 予約", test_init_includes_ent_self),
        ("(v3) init: channels 空", test_init_channels_empty),
        ("init: 冪等", test_init_idempotent),
        ("get_entity: 存在", test_get_entity_existing),
        ("get_entity: 不存在", test_get_entity_missing),
        ("get_channel: ensure 後に存在", test_get_channel_existing),
        ("get_channel: 起動直後 device も None", test_get_channel_missing),
        ("list_entities/channels", test_list_entities_and_channels),
        ("render: None", test_render_with_none),
        ("render: wm 中身空 → 空文字", test_render_empty_facts),
        ("(v3) render: ensure 後に channel 出現", test_render_after_ensure_channel),
        # 段階3: Fact schema + β+
        ("make_fact: 構造", test_make_fact_structure),
        ("β+ match 収束", test_update_fact_confidence_match_converges),
        ("β+ mismatch 降下", test_update_fact_confidence_mismatch_drops),
        ("β+ 下限 0.0", test_update_fact_confidence_lower_bound),
        ("find_fact: 存在/不存在/frozen", test_find_fact),
        ("add_or_update_fact: 新規", test_add_or_update_fact_new),
        ("add_or_update_fact: 一致で β+", test_add_or_update_fact_matching_value),
        ("add_or_update_fact: 値違いで bitemporal", test_add_or_update_fact_differing_value),
        # 段階3: Channel
        ("observe_channel_activity: 更新", test_observe_channel_activity),
        ("observe_channel_activity: silent skip", test_observe_channel_nonexistent_silent),
        ("get_tool_channel: 逆引き", test_get_tool_channel),
        # 段階3: Entity
        ("ensure_entity: 新規作成", test_ensure_entity_creates),
        ("ensure_entity: 既存返却", test_ensure_entity_returns_existing),
        ("migrate_entity_fields: 冪等", test_migrate_entity_fields_idempotent),
        # 段階6-C v3: ensure_channel
        ("(v3) ensure_channel: 新規作成", test_ensure_channel_creates_new),
        ("(v3) ensure_channel: 冪等", test_ensure_channel_idempotent),
        ("(v3) ensure_channel: entity と対称", test_ensure_channel_entity_symmetry),
        # 段階3: C-gradual
        ("sync: 新規作成", test_sync_from_memory_entities_creates_new),
        ("sync: description fact", test_sync_appends_description_fact),
        ("sync: 空レコード", test_sync_empty_records),
        ("sync: limit 尊重", test_sync_respects_limit),
        ("sync: resolver で類似名 merge (段階4)", test_sync_resolver_merges_similar_names),
        ("sync: embed なしは merge しない (段階4)", test_sync_without_embed_fn_no_merge),
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
