"""世界モデル (WM) — schema 定義・初期化・アクセサ・prompt レンダリング。

WORLD_MODEL.md §5-§6 段階2-3 の実装 + 段階6-C v3 で動的 channel 化。

設計指針 (STAGE2/3/6C_IMPLEMENTATION_PLAN.md 準拠):
- ミニマリズム: 消費者のない field は段階を待って追加
- ent_self は構造的スロット (段階7 で state["self"] 統合予定)
- 生 dict アクセス禁止、必ず本モジュールの accessor 経由 (§10-1 accessor-only)
- 将来の unified memory_store 移行時は accessor 内部のみ書き換える
- **(v3) World is observed, not given**: channel は bootstrap せず、観察で ensure_channel から生える
- channel spec 生成ロジックは `core/channel_registry.py` の判定関数が所有

段階4 以降の契約:
- entity_id 生成は段階4 で Entity Resolver に差替予定
- Tool→entity fact 更新は段階4 で target entity 解決後に追加
- channel_mismatch 乗算は段階5
- **channel は動的登録 (段階6-C v3)**: 起動直後 channels={}、観察で ensure_channel 経由で生える
"""
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

WM_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 初期化
# ============================================================

def init_world_model() -> dict:
    """初期 world_model を返す。channels は空 (観察で ensure_channel 経由で生える)。

    ent_self は構造的スロットとして予約 (name="self" 固定)。
    個人名情報は state["self"]["name"] に任せる (段階7 で統合予定)。
    """
    now = _now()
    entities = {
        "ent_self": {
            "id": "ent_self",
            "name": "self",
            "facts": [],
            "aliases": [],
            "channels": [],
            "last_seen": None,
            "created_at": now,
            "updated_at": now,
        }
    }
    return {
        "entities": entities,
        "channels": {},   # (v3) 空から始まる。観察で channel_registry 判定関数 + ensure_channel 経由で動的に生える
        "version": WM_SCHEMA_VERSION,
        "last_updated": now,
    }


# ============================================================
# アクセサ (段階2 では read-only、段階3+ で update 系を追加予定)
# ============================================================

def get_entity(wm: Optional[dict], entity_id: str) -> Optional[dict]:
    """entity_id で entity を取得。存在しなければ None。"""
    if not wm:
        return None
    return wm.get("entities", {}).get(entity_id)


def get_channel(wm: Optional[dict], channel_id: str) -> Optional[dict]:
    """channel_id で channel を取得。存在しなければ None。"""
    if not wm:
        return None
    return wm.get("channels", {}).get(channel_id)


def list_entities(wm: Optional[dict]) -> list:
    """全 entity のリストを返す (順序不定)。"""
    if not wm:
        return []
    return list(wm.get("entities", {}).values())


def list_channels(wm: Optional[dict]) -> list:
    """全 channel のリストを返す (順序不定)。"""
    if not wm:
        return []
    return list(wm.get("channels", {}).values())


# ============================================================
# 段階3: Fact schema + β+ 更新
# ============================================================

def make_fact(key: str, value, confidence: float = 0.7) -> dict:
    """新規 Fact 構造を返す。"""
    now = _now()
    return {
        "key": key,
        "value": value,
        "confidence": max(0.0, min(1.0, confidence)),
        "valid_from": now,
        "valid_to": None,
        "learned_at": now,
        "observation_count": 1,
        "last_observed_at": now,
    }


def update_fact_confidence(fact: dict, observation_matches: bool) -> None:
    """β+ 更新。in-place で fact を書き換え。
    matches=True:  confidence += 0.05 * (1 - confidence), count+=1, last_observed 更新
    matches=False: confidence -= 0.15 (下限 0.0)
    """
    conf = float(fact.get("confidence", 0.7))
    if observation_matches:
        fact["confidence"] = conf + 0.05 * (1 - conf)
        fact["observation_count"] = int(fact.get("observation_count", 0)) + 1
        fact["last_observed_at"] = _now()
    else:
        fact["confidence"] = max(0.0, conf - 0.15)


def find_fact(entity: Optional[dict], key: str) -> Optional[dict]:
    """entity.facts から key 一致かつ valid_to=None (現行) の fact を返す。"""
    if not entity:
        return None
    for f in entity.get("facts", []):
        if f.get("key") == key and f.get("valid_to") is None:
            return f
    return None


def add_or_update_fact(entity: dict, key: str,
                       value, confidence: float = 0.7) -> dict:
    """既存 fact があれば β+、なければ新規追加。
    既存 value と異なる場合は bitemporal: 旧 fact の valid_to=now で凍結 +
    信頼度削減、新 fact を追加。
    """
    existing = find_fact(entity, key)
    entity.setdefault("facts", [])
    if existing is None:
        new_fact = make_fact(key, value, confidence)
        entity["facts"].append(new_fact)
        entity["updated_at"] = _now()
        return new_fact
    if str(existing.get("value")) == str(value):
        update_fact_confidence(existing, True)
        entity["updated_at"] = _now()
        return existing
    # 値が違う: bitemporal 更新
    existing["valid_to"] = _now()
    update_fact_confidence(existing, False)
    new_fact = make_fact(key, value, confidence)
    entity["facts"].append(new_fact)
    entity["updated_at"] = _now()
    return new_fact


# ============================================================
# 段階3: Channel 活動追跡
# ============================================================

def observe_channel_activity(wm: Optional[dict], channel_id: str,
                             timestamp: Optional[str] = None) -> None:
    """channel の last_activity_at / activity_count を更新。in-place。
    wm=None or channel 未登録なら silent skip (段階4+で動的 channel 追加に備え)。
    """
    if not wm:
        return
    channel = wm.get("channels", {}).get(channel_id)
    if not channel:
        return
    channel["last_activity_at"] = timestamp or _now()
    channel["activity_count"] = int(channel.get("activity_count", 0)) + 1
    wm["last_updated"] = _now()


def get_tool_channel(wm: Optional[dict], tool_name: str) -> Optional[str]:
    """ツール名から所属 channel id を逆引き。
    tools_in / tools_out いずれかに含まれる channel を返す。
    どこにも属さなければ None (internal な自作 tool 等)。
    """
    if not wm or not tool_name:
        return None
    for ch_id, ch in wm.get("channels", {}).items():
        if tool_name in ch.get("tools_in", []):
            return ch_id
        if tool_name in ch.get("tools_out", []):
            return ch_id
    return None


# ============================================================
# 段階3: Entity 作成 + lazy migration
# ============================================================

def _slugify(name: str) -> str:
    """entity ID 用 slug。日本語は保持、記号類は _ に。
    段階4 で Entity Resolver に置換予定。
    """
    s = re.sub(r'[^\w\u3000-\u9fff\u4e00-\u9fff]+', '_', name.strip())
    s = s.strip('_')
    return s[:32] if s else "unknown"


def ensure_entity(wm: dict, entity_id: str, name: str) -> dict:
    """entity 存在なら返却、未存在なら新規作成して返却。in-place。
    新規作成時は段階3 schema の全 field を付与。
    """
    entities = wm.setdefault("entities", {})
    if entity_id in entities:
        # 段階2 entity に段階3 field 欠損があれば補完
        migrate_entity_fields(entities[entity_id])
        return entities[entity_id]
    now = _now()
    entity = {
        "id": entity_id,
        "name": name,
        "facts": [],
        "aliases": [],
        "channels": [],
        "last_seen": None,
        "created_at": now,
        "updated_at": now,
    }
    entities[entity_id] = entity
    wm["last_updated"] = now
    return entity


def migrate_entity_fields(entity: dict) -> None:
    """段階2 entity に段階3 追加 field を付与 (in-place, 冪等)。"""
    if "aliases" not in entity:
        entity["aliases"] = []
    if "channels" not in entity:
        entity["channels"] = []
    if "last_seen" not in entity:
        entity["last_seen"] = None


# ============================================================
# 段階6-C v3: Channel 動的登録 (ensure_entity と対称)
# ============================================================

def ensure_channel(wm: dict, id: str, type: str,
                   tools_in=None, tools_out=None) -> dict:
    """channel id 存在なら既存を返却、未存在なら新規作成して返却。in-place。

    `ensure_entity` と対称設計、冪等。
    spec は `core/channel_registry.py` の判定関数が生成して渡してくる前提
    (caller は生 dict を組み立てない)。

    Args:
        wm: world_model
        id: channel id (例: "device", "claude", "mcp_discord-bot")
        type: "direct" / "social" / "self" など
        tools_in / tools_out: 観察ツール / 出力ツール名のリスト

    Returns:
        channel dict (新規作成 or 既存)
    """
    channels = wm.setdefault("channels", {})
    if id in channels:
        return channels[id]
    channel = {
        "id": id,
        "type": type,
        "tools_in": list(tools_in or []),
        "tools_out": list(tools_out or []),
    }
    channels[id] = channel
    wm["last_updated"] = _now()
    return channel


# ============================================================
# 段階3: C-gradual 同期 (memory/entity → WM.entities 片方向ミラー)
# ============================================================

def sync_from_memory_entities(wm: Optional[dict], memory_entity_records: list,
                              limit: int = 20,
                              embed_fn=None, cosine_fn=None,
                              llm_call_fn=None) -> int:
    """memory/entity レコード群から WM entity を片方向ミラー作成/更新。

    entity_name でグループ化 → Entity Resolver (段階4) で既存解決
    → 最新 content を description fact として add_or_update。

    Args:
        wm: world_model
        memory_entity_records: memory/entity の dict レコード群
        limit: 処理件数上限
        embed_fn / cosine_fn: Entity Resolver Tier 2 で使う embedding 関数
        llm_call_fn: Tier 3 LLM tiebreak (省略時は ambiguous は新規扱い)

    戻り値: 新規作成された entity 数。
    """
    if not wm or not memory_entity_records:
        return 0
    # 循環 import 回避のため関数内 import
    from core.entity_resolver import resolve_or_create_entity

    groups = defaultdict(list)
    for rec in memory_entity_records[:limit]:
        name = rec.get("metadata", {}).get("entity_name", "").strip()
        if not name:
            continue
        groups[name].append(rec)
    created_count = 0
    for name, records in groups.items():
        entity, is_new = resolve_or_create_entity(
            wm, name,
            embed_fn=embed_fn,
            cosine_fn=cosine_fn,
            llm_call_fn=llm_call_fn,
        )
        if entity is None:
            continue
        if is_new:
            created_count += 1
        latest = max(records,
                     key=lambda r: r.get("updated_at", r.get("created_at", "")))
        content = str(latest.get("content", "")).strip()
        if content:
            add_or_update_fact(entity, "description", content)
    return created_count


# ============================================================
# prompt レンダリング (prompt_assembly から呼ばれる)
# ============================================================

def render_for_prompt(wm: Optional[dict], max_entities: int = 10) -> str:
    """[世界モデル] セクションを system_prompt 用にレンダリング。

    wm=None または空の場合は空文字を返す (prompt_assembly 側で
    セクションごと省略される)。
    段階3: 各 fact に confidence 値を括弧で付与。
    """
    if not wm:
        return ""

    lines = ["## 世界モデル"]

    # channels サマリ
    channels = list_channels(wm)
    if channels:
        lines.append("### チャネル")
        for c in channels:
            lines.append(f"- {c['id']} ({c['type']})")

    # entities サマリ (facts を持つ entity のみ表示)
    entities = list_entities(wm)
    entities_with_facts = [e for e in entities if e.get("facts")]
    lines.append("### 観測された存在")
    if entities_with_facts:
        for e in entities_with_facts[:max_entities]:
            fact_summaries = []
            for f in e.get("facts", [])[:3]:
                # 凍結済 fact (valid_to 設定済) はスキップ
                if f.get("valid_to") is not None:
                    continue
                key = f.get("key", "?")
                val = f.get("value", "?")
                conf = f.get("confidence")
                if conf is not None:
                    fact_summaries.append(f"{key}={val}({float(conf):.2f})")
                else:
                    fact_summaries.append(f"{key}={val}")
            if fact_summaries:
                lines.append(f"- {e.get('name', e['id'])}: {', '.join(fact_summaries)}")
        # フィルタ後 fact が全件 frozen/空だった場合の処理
        if all(not any(f.get("valid_to") is None for f in e.get("facts", []))
               for e in entities_with_facts):
            # 可能性としてほぼ起きないけど念のため
            pass
    else:
        lines.append("(まだ観測されていない — 段階3 で自動登録される)")

    return "\n".join(lines)
