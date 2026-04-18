"""世界モデル (WM) — schema 定義・初期化・アクセサ・prompt レンダリング。

WORLD_MODEL.md §5-§6 段階2 の実装。
段階2 では read-only; auto-update は段階3 で実装される。

設計指針 (STAGE2_IMPLEMENTATION_PLAN.md 準拠):
- ミニマリズム: 段階2 で消費者のない field は含めない
- ent_self は構造的スロットとして予約 (段階7 で state["self"] 統合予定)
- 生 dict アクセス禁止、必ず本モジュールの accessor 経由 (§10-1 accessor-only ルール)
- 将来の unified memory_store 移行 (§14) 時は accessor 内部のみ書き換える
"""
from datetime import datetime
from typing import Optional

WM_SCHEMA_VERSION = 1

# 段階2 で bootstrap する channel 定義。type / tools_in / tools_out のみ。
# health / last_error / tags は段階3+ で追加。
_CHANNEL_BOOTSTRAP = {
    "device": {
        "type": "direct",
        "tools_in": ["[device_input]"],
        "tools_out": ["output_display", "camera_stream", "screen_peek",
                      "view_image", "listen_audio", "mic_record"],
    },
    "elyth": {
        "type": "social",
        "tools_in": ["elyth_info", "elyth_get"],
        "tools_out": ["elyth_post", "elyth_reply", "elyth_like", "elyth_follow"],
    },
    "x": {
        "type": "social",
        "tools_in": ["x_timeline", "x_search", "x_get_notifications"],
        "tools_out": ["x_post", "x_reply", "x_quote", "x_like"],
    },
    "internal": {
        "type": "self",
        "tools_in": [],
        "tools_out": [],
    },
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 初期化
# ============================================================

def init_world_model() -> dict:
    """初期 world_model を返す。冪等 (何度呼んでも同構造)。

    ent_self は構造的スロットとして予約 (name="self" 固定)。
    個人名情報は state["self"]["name"] に任せる (段階7 で統合予定)。
    """
    now = _now()
    entities = {
        "ent_self": {
            "id": "ent_self",
            "name": "self",
            "facts": [],
            "created_at": now,
            "updated_at": now,
        }
    }
    channels = {}
    for cid, meta in _CHANNEL_BOOTSTRAP.items():
        channels[cid] = {
            "id": cid,
            "type": meta["type"],
            "tools_in": list(meta["tools_in"]),
            "tools_out": list(meta["tools_out"]),
        }
    return {
        "entities": entities,
        "channels": channels,
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
# prompt レンダリング (prompt_assembly から呼ばれる)
# ============================================================

def render_for_prompt(wm: Optional[dict], max_entities: int = 10) -> str:
    """[世界モデル] セクションを system_prompt 用にレンダリング。

    wm=None または空の場合は空文字を返す (prompt_assembly 側で
    セクションごと省略される)。
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
                # 段階3 で Fact schema 定義後、ここを拡張
                fact_summaries.append(f"{f.get('key','?')}={f.get('value','?')}")
            lines.append(f"- {e.get('name', e['id'])}: {', '.join(fact_summaries)}")
    else:
        lines.append("(まだ観測されていない — 段階3 で自動登録される)")

    return "\n".join(lines)
