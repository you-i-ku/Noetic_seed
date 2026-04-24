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

def make_fact(key: str, value, confidence: float = 0.7,
              perspective: Optional[dict] = None) -> dict:
    """新規 Fact 構造を返す。

    段階11-A: perspective 属性を追加 (fact が誰の視点由来かを記録)。
    None なら default_self_perspective (self/actual) で補完。
    """
    from core.perspective import default_self_perspective
    now = _now()
    if perspective is None:
        perspective = default_self_perspective()
    return {
        "key": key,
        "value": value,
        "confidence": max(0.0, min(1.0, confidence)),
        "valid_from": now,
        "valid_to": None,
        "learned_at": now,
        "observation_count": 1,
        "last_observed_at": now,
        "perspective": perspective,
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
                       value, confidence: float = 0.7,
                       perspective: Optional[dict] = None) -> dict:
    """既存 fact があれば β+、なければ新規追加。
    既存 value と異なる場合は bitemporal: 旧 fact の valid_to=now で凍結 +
    信頼度削減、新 fact を追加。

    段階11-A: perspective 属性を新規 fact に付与 (既存 fact の perspective は
    不変維持、bitemporal 哲学整合)。
    """
    existing = find_fact(entity, key)
    entity.setdefault("facts", [])
    if existing is None:
        new_fact = make_fact(key, value, confidence, perspective=perspective)
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
    new_fact = make_fact(key, value, confidence, perspective=perspective)
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

# ============================================================
# 段階7: Materialized view 化 (memory/wm.jsonl ↔ in-memory WM)
# ============================================================

def store_wm_fact(wm: dict, entity_name: str, fact_key: str,
                  fact_value, confidence: float = 0.7,
                  entity_id: Optional[str] = None,
                  perspective: Optional[dict] = None) -> dict:
    """wm タグの記憶を保存 (materialized view 同期の唯一の accessor)。

    段階7 Step 3: memory/wm.jsonl (source of truth) + state["world_model"] (派生) を
    同時に更新。β+ / bitemporal は既存 add_or_update_fact に委譲。

    段階11-A: perspective を add_or_update_fact と memory_store 両方に伝播。
    None なら default_self_perspective (memory_store 側で補完)。

    Args:
        wm: state["world_model"] dict
        entity_name: 対象エンティティ名
        fact_key: fact キー (e.g. "description", "role")
        fact_value: fact 値
        confidence: 確信度 (0.0-1.0、既定 0.7)
        entity_id: 省略時は _slugify(entity_name) で生成
        perspective: 段階11-A — fact がどの視点由来かの metadata。
            None なら default_self_perspective (self/actual) で補完。

    Returns:
        add_or_update_fact の戻り値 (new fact dict)
    """
    from core.memory import memory_store  # 循環 import 回避

    if entity_id is None:
        entity_id = f"ent_{_slugify(entity_name)}"
    entity = ensure_entity(wm, entity_id, entity_name)
    new_fact = add_or_update_fact(entity, fact_key, fact_value, confidence,
                                  perspective=perspective)

    content = f"{entity_name}.{fact_key} = {fact_value}"
    metadata = {
        "entity_name": entity_name,
        "entity_id": entity_id,
        "fact_key": fact_key,
        "fact_value": fact_value,
        "confidence": new_fact["confidence"],
        "valid_from": new_fact["valid_from"],
        "valid_to": new_fact.get("valid_to"),
        "observation_count": new_fact["observation_count"],
    }
    memory_store("wm", content, metadata,
                 origin="store_wm_fact",
                 source_context="wm_materialized",
                 perspective=perspective)
    return new_fact


def rebuild_wm_from_jsonl(wm: dict, wm_records: list) -> int:
    """memory/wm.jsonl のレコード群から WM entities を再構築。

    段階7 Step 3: 起動時 / reflect 時に呼ばれる materialized view 再構築。
    既存 entities は保持 (additive)、同じ fact は β+ 再適用、
    矛盾 fact は bitemporal で旧 fact valid_to 凍結 + 新 fact 追加。

    Args:
        wm: state["world_model"]
        wm_records: memory/wm.jsonl から読み出したレコード群 (古い順推奨)

    Returns:
        処理に成功した record 数 (不正 record は skip)
    """
    from core.perspective import default_self_perspective
    count = 0
    for rec in wm_records:
        meta = rec.get("metadata", {})
        entity_name = str(meta.get("entity_name", "")).strip()
        fact_key = str(meta.get("fact_key", "")).strip()
        if not entity_name or not fact_key:
            continue
        fact_value = meta.get("fact_value")
        try:
            confidence = float(meta.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        entity_id = meta.get("entity_id") or f"ent_{_slugify(entity_name)}"
        # 段階11-A: 専用キー perspective 優先、欠落時は default (backward compat)
        perspective = rec.get("perspective") or default_self_perspective()
        entity = ensure_entity(wm, entity_id, entity_name)
        add_or_update_fact(entity, fact_key, fact_value, confidence,
                           perspective=perspective)
        count += 1
    return count


def _pkey_matches_filter(perspective: Optional[dict],
                         view_filter: Optional[dict]) -> bool:
    """perspective dict が view_filter 条件を満たすか。

    段階11-A: fact / entry の perspective を filter にかける helper。
    view_filter=None → 常に True (全視点表示)
    view_filter={"viewer": "self"} → viewer=self のみ
    view_filter={"viewer_type": "actual"} → 仮想除外
    view_filter={"viewer": "X", "viewer_type": "Y"} → AND
    perspective 欠落 (旧 entry) は default_self_perspective 相当で判定。
    """
    if view_filter is None:
        return True
    from core.perspective import default_self_perspective
    p = perspective or default_self_perspective()
    for key, expected in view_filter.items():
        if p.get(key) != expected:
            return False
    return True


def _pkey_str_to_perspective(pkey: str) -> dict:
    """state["dispositions"] の key str → perspective dict 逆変換。

    段階11-A: dispositions の view_filter 判定で _pkey_matches_filter を
    使い回すための helper。
      "self" → {"viewer": "self", "viewer_type": "actual"}
      "attributed:X" → {"viewer": "X", "viewer_type": "actual"}
      "imagined:X" / "past_self:X" / "future_self:X" → 対応する type に
    """
    if pkey == "self":
        return {"viewer": "self", "viewer_type": "actual"}
    if ":" in pkey:
        prefix, viewer = pkey.split(":", 1)
        if prefix == "attributed":
            return {"viewer": viewer, "viewer_type": "actual"}
        return {"viewer": viewer, "viewer_type": prefix}
    return {"viewer": pkey, "viewer_type": "actual"}


def _is_perspective_keyed_dispositions(dispositions: dict) -> bool:
    """dispositions dict が perspective-keyed 形式か判定 (dual support)。

    段階11-A: Step 5 までの過渡期、既存 flat dict (段階10.5 Fix 4 δ')
    と新 perspective-keyed dict の両方を受け取って綺麗に動くため。
      flat: {"curiosity": 0.8}  → 値が数値
      perspective-keyed: {"self": {"curiosity": {"value": 0.8,...}}} → 値が dict
    """
    if not dispositions:
        return False
    first_val = next(iter(dispositions.values()), None)
    return isinstance(first_val, dict)


def render_for_prompt(wm: Optional[dict], max_entities: int = 10,
                      *, opinions: Optional[list] = None,
                      dispositions: Optional[dict] = None,
                      view_filter: Optional[dict] = None) -> str:
    """[世界モデル] セクションを system_prompt 用にレンダリング。

    wm=None または中身全部空 (channels / dispositions / opinions いずれも
    空) の場合は空文字を返す (prompt_assembly 側でセクションごと省略)。
    段階3: 各 fact に confidence 値を括弧で付与。
    段階10.5 Fix 4 δ' (PLAN §6-2 準拠): opinions / dispositions を追加
    セクションで表示して「構造化自己認識」を完成させる。

    段階11-A:
      - view_filter kwarg 追加 (perspective filter)
        None → 全視点表示 (debug / inspect_wm_view の default/free case)
        {"viewer": "self"} → self 視点のみ (system_prompt での default)
        {"viewer_type": "actual"} → 仮想視点除外
      - dispositions の dual support: flat dict (段階10.5 既存) と
        perspective-keyed dict (Step 5 以降) 両方を受けられる
      - fact の valid_to 凍結フィルタと view_filter の 2 段フィルタ
    """
    if not wm:
        return ""

    lines = ["## 世界モデル"]

    # channels サマリ (view_filter 非適用 — channel は物理的結合、視点分離対象外)
    channels = list_channels(wm)
    if channels:
        lines.append("### チャネル")
        for c in channels:
            lines.append(f"- {c['id']} ({c['type']})")

    # 段階10.5 Fix 4 δ' + 段階11-A: dispositions dual support (flat / perspective-keyed)
    if dispositions:
        is_pkeyed = _is_perspective_keyed_dispositions(dispositions)
        if is_pkeyed:
            # perspective-keyed 形式 (Step 5 以降)
            # outer key: "self" / "attributed:X" / "imagined:X" 等
            # inner: {trait_key: {"value": 0.8, "confidence": None, ...}}
            rendered_disp_header = False
            for pkey in sorted(dispositions.keys()):
                traits = dispositions[pkey]
                if not isinstance(traits, dict) or not traits:
                    continue
                # view_filter 適用 (pkey → perspective 逆変換で判定)
                if not _pkey_matches_filter(
                        _pkey_str_to_perspective(pkey), view_filter):
                    continue
                if not rendered_disp_header:
                    lines.append("### 傾向 (dispositions)")
                    rendered_disp_header = True
                # sub-header
                if pkey == "self":
                    lines.append("#### 自己視点")
                elif pkey.startswith("attributed:"):
                    lines.append(f"#### {pkey[len('attributed:'):]} 視点 (attributed)")
                else:
                    lines.append(f"#### {pkey}")
                for trait in sorted(traits.keys()):
                    info = traits[trait]
                    val = info.get("value") if isinstance(info, dict) else info
                    try:
                        lines.append(f"- {trait}: {float(val):.2f}")
                    except (TypeError, ValueError):
                        continue
        else:
            # flat 形式 (段階10.5 Fix 4 δ'、backward compat)
            lines.append("### 傾向 (dispositions)")
            for key, val in sorted(dispositions.items()):
                try:
                    lines.append(f"- {key}: {float(val):.2f}")
                except (TypeError, ValueError):
                    continue

    # 段階10.5 Fix 4 δ': opinions (iku の意見、memory tag="opinion" から上位 N 件)
    # 段階11-A: opinion 側の view_filter は Step 6 で search_memory 層に追加予定。
    # ここでは opinion entry の perspective を見て個別 filter する最小対応のみ。
    if opinions:
        rendered_op_header = False
        for op in opinions[:5]:
            content = str(op.get("content", ""))[:120]
            if not content:
                continue
            # 段階11-A: opinion 自体が持つ perspective を filter にかける
            if not _pkey_matches_filter(op.get("perspective"), view_filter):
                continue
            if not rendered_op_header:
                lines.append("### 意見 (opinions)")
                rendered_op_header = True
            conf = op.get("metadata", {}).get("confidence")
            if conf is not None:
                try:
                    lines.append(f"- {content} ({float(conf):.2f})")
                    continue
                except (TypeError, ValueError):
                    pass
            lines.append(f"- {content}")

    if len(lines) == 1:
        return ""
    return "\n".join(lines)
