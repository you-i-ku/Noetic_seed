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


# ============================================================
# 段階14 Step D: Basin Migration / Phase Transition
# ============================================================
#
# STAGE14 PLAN §6 の実装。iku が同 basin (= subject cluster) に N cycle 滞留
# したら強制脱出でなく phase transition として状態を表現、Step C の β 動的化
# と相補して自然な basin 移行を促す。Ramstead 系 FEP の局所エルゴード性違反
# を「memory 形成中→転移」の二相モデルで表現。
#
# basin_id は cluster_estimation の cluster_id (uuid hex 8 文字 str) を流用
# (memo line 154-156「触らない」literal、Phase 5 非永続 posterior 整合)。
# state は JSON-safe (memo line 144「再起動でも保持」literal、Step B BLOCKER
# 同種注意点)。

BASIN_TRANSITION_THRESHOLD = 5     # 同 basin 滞留閾値 (PLAN §10-7 確定値)
BASIN_VISIT_HISTORY_CAP = 50       # FIFO 履歴上限 (PLAN §6-2 literal)


def update_basin_state(
    state: dict,
    current_subject_id: Optional[str] = None,
    clusters_snapshot: Optional[list] = None,
) -> dict:
    """同 basin 滞留検知 + phase transition pending 判定 (PLAN §6-2 literal)。

    呼出経路 2 つ:
    - reflect 発火時: clusters_snapshot=estimate_clusters() 結果 +
      current_subject_id=直近 memory_id を渡す。subject の cluster 所属を
      再計算、basin 切替検知 + dwell 更新 + history append。
    - 非 reflect cycle: 引数なし or None で呼出、basin 不変で dwell カウント
      のみ (basin_id 不在時は no-op)。

    state["basin_state"] フィールド (JSON-safe、memo line 144「再起動でも保持」):
      current_basin_id (str): cluster_id (uuid hex 8 文字) or "" (unknown)
      current_basin_dwell (int): 同 basin 滞留 cycle 数
      basin_transition_threshold (int): 閾値 (default BASIN_TRANSITION_THRESHOLD)
      phase_transition_pending (bool): 閾値超過 = Step C β boost trigger
      basin_visit_history (list[str]): 過去 basin_id (FIFO)

    Args:
        state: state dict (破壊的更新)
        current_subject_id: reflect 発火時の subject memory_id (= 直近 memory_id)
        clusters_snapshot: estimate_clusters() 戻り値 list

    Returns:
        更新後 basin_state dict (state["basin_state"] と同参照)
    """
    bs = state.setdefault("basin_state", {
        "current_basin_id": "",
        "current_basin_dwell": 0,
        "basin_transition_threshold": BASIN_TRANSITION_THRESHOLD,
        "phase_transition_pending": False,
        "basin_visit_history": [],
    })

    if clusters_snapshot is not None and current_subject_id:
        # reflect 経路: subject の cluster 所属を再計算
        new_basin = _classify_basin_from_snapshot(
            current_subject_id, clusters_snapshot,
        )
        if not new_basin:
            # subject が cluster の memory_ids に含まれない (新規 memory 等):
            # basin 不変、dwell も update せず safe no-op (basin_id="" のまま、
            # phase_transition_pending も既存値維持)
            pass
        elif new_basin == bs["current_basin_id"]:
            bs["current_basin_dwell"] += 1
        else:
            # basin 切替: 旧 basin を history に保存 (空 id は履歴に入れない)
            if bs["current_basin_id"]:
                bs["basin_visit_history"].append(bs["current_basin_id"])
                if len(bs["basin_visit_history"]) > BASIN_VISIT_HISTORY_CAP:
                    del bs["basin_visit_history"][0]
            bs["current_basin_id"] = new_basin
            bs["current_basin_dwell"] = 1
            bs["phase_transition_pending"] = False  # transition 完了
    else:
        # 非 reflect cycle: dwell カウントのみ (basin 不変)
        if bs["current_basin_id"]:
            bs["current_basin_dwell"] += 1

    # 閾値超過判定 (両経路共通、Step C _compute_dynamic_beta の boost trigger)
    if bs["current_basin_dwell"] >= bs["basin_transition_threshold"]:
        bs["phase_transition_pending"] = True

    return bs


def _classify_basin_from_snapshot(
    subject_id: str,
    clusters: list,
) -> str:
    """subject_id がどの cluster の memory_ids に含まれるかで basin 判定。

    PLAN §6-2 概念 literal の `_classify_basin` 概念実装、cluster_estimation
    流用 (memo line 154-156「触らない」literal、Phase 5 非永続 posterior 整合)。
    実コード `cluster_id` は uuid hex 8 文字 str (PLAN literal int との乖離は
    Step B/C 同様の概念呼称解釈)。

    Args:
        subject_id: subject memory_id (str)
        clusters: estimate_clusters の戻り値
            [{"cluster_id": str, "label": str, "memory_ids": [...], "method": str}, ...]

    Returns:
        cluster_id (str) or "" (subject が cluster に含まれない / 入力空)
    """
    if not subject_id or not clusters:
        return ""
    # V07.5 commit 5 (A4 literal fix、PLAN v1.4 §3-5): basin id 堅牢化
    # subject_id 文字列正規化で type mismatch (旧 str vs int 混在 case) literal 防止、
    # cluster snapshot との対応取れない bug literal 解消 (段階14 K2/K4/K6 dormancy は
    # smoke data 依存、本 fix は構造的不具合の literal 解消のみ literal 保証)。
    subject_id_str = str(subject_id)
    for cluster in clusters:
        memory_ids_set = {str(mid) for mid in cluster.get("memory_ids", [])}
        if subject_id_str in memory_ids_set:
            return str(cluster.get("cluster_id", ""))
    return ""
