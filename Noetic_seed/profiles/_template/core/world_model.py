"""世界モデル (WM) — schema 定義・初期化・アクセサ・prompt レンダリング。

段階11-D Phase 0 で entity 関数群 (ensure_entity / make_fact /
add_or_update_fact / store_wm_fact / rebuild_wm_from_jsonl /
sync_from_memory_entities 等) を全削除し、channels + perspective filter helper
のみの最小構成にした。entity 概念は B1 完全廃止 (PLAN §11-3)、後続 Phase
(memory_graph / link graph / Physarum / cluster 推定) が役割を継承する。

維持する要素:
- channels (物理結合、tag 非依存): ensure_channel / get_channel / list_channels /
  observe_channel_activity / get_tool_channel
- perspective filter helper (11-A 視点層): _pkey_matches_filter /
  _pkey_str_to_perspective / _is_perspective_keyed_dispositions
- render_for_prompt (channels / dispositions / opinions、view_filter 付き)

**(v3) World is observed, not given**: channel は bootstrap せず、観察で
ensure_channel から生える (段階6-C v3)。
"""
from datetime import datetime
from typing import Optional

WM_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 初期化
# ============================================================

def init_world_model() -> dict:
    """初期 world_model を返す。channels は空 (観察で ensure_channel 経由で生える)。"""
    now = _now()
    return {
        "channels": {},
        "version": WM_SCHEMA_VERSION,
        "last_updated": now,
    }


# ============================================================
# Channel アクセサ
# ============================================================

def get_channel(wm: Optional[dict], channel_id: str) -> Optional[dict]:
    """channel_id で channel を取得。存在しなければ None。"""
    if not wm:
        return None
    return wm.get("channels", {}).get(channel_id)


def list_channels(wm: Optional[dict]) -> list:
    """全 channel のリストを返す (順序不定)。"""
    if not wm:
        return []
    return list(wm.get("channels", {}).values())


# ============================================================
# Channel 活動追跡
# ============================================================

def observe_channel_activity(wm: Optional[dict], channel_id: str,
                             timestamp: Optional[str] = None) -> None:
    """channel の last_activity_at / activity_count を更新。in-place。
    wm=None or channel 未登録なら silent skip。
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
# 段階6-C v3: Channel 動的登録
# ============================================================

def ensure_channel(wm: dict, id: str, type: str,
                   tools_in=None, tools_out=None) -> dict:
    """channel id 存在なら既存を返却、未存在なら新規作成して返却。in-place。

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
# β+ confidence 更新 (汎用 helper、predictor 等 caller に開放)
# ============================================================

def update_fact_confidence(fact: dict, observation_matches: bool) -> None:
    """β+ 更新。in-place で fact を書き換え。

    匹名は entity fact 由来だが、"confidence" / "observation_count" /
    "last_observed_at" key を持つ任意 dict に使える汎用 helper。段階11-D
    Phase 0 で entity 本体は撤去されたが、本関数は残された
    (caller: core/predictor.py update_predictor_confidence)。

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


# ============================================================
# 段階11-A: Perspective filter helper
# ============================================================

def _pkey_matches_filter(perspective: Optional[dict],
                         view_filter: Optional[dict]) -> bool:
    """perspective dict が view_filter 条件を満たすか。

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


# ============================================================
# prompt レンダリング (prompt_assembly から呼ばれる)
# ============================================================

def render_for_prompt(wm: Optional[dict], max_entities: int = 10,
                      *, opinions: Optional[list] = None,
                      dispositions: Optional[dict] = None,
                      view_filter: Optional[dict] = None) -> str:
    """[世界モデル] セクションを system_prompt 用にレンダリング。

    wm=None または中身全部空 (channels / dispositions / opinions いずれも
    空) の場合は空文字を返す (prompt_assembly 側でセクションごと省略)。

    段階10.5 Fix 4 δ' (PLAN §6-2 準拠): opinions / dispositions を追加
    セクションで表示して「構造化自己認識」を完成させる。

    段階11-A:
      - view_filter kwarg 追加 (perspective filter)
        None → 全視点表示 (debug / free case)
        {"viewer": "self"} → self 視点のみ (system_prompt での default)
        {"viewer_type": "actual"} → 仮想視点除外
      - dispositions の dual support: flat dict (段階10.5 既存) と
        perspective-keyed dict (Step 5 以降) 両方を受けられる

    max_entities kwarg は legacy signature 互換のため残置 (entity 廃止後は
    未使用、呼び手への影響回避)。
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
            rendered_disp_header = False
            for pkey in sorted(dispositions.keys()):
                traits = dispositions[pkey]
                if not isinstance(traits, dict) or not traits:
                    continue
                if not _pkey_matches_filter(
                        _pkey_str_to_perspective(pkey), view_filter):
                    continue
                if not rendered_disp_header:
                    lines.append("### 傾向 (dispositions)")
                    rendered_disp_header = True
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
    if opinions:
        rendered_op_header = False
        for op in opinions[:5]:
            content = str(op.get("content", ""))[:120]
            if not content:
                continue
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
