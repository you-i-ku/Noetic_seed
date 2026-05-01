"""内省モジュール v3 — 定期的な自己省察 (Generative Agents + Reflexion 統合)
N サイクルごと or 高 prediction_error 時に発火。
NOTES (raw 気づき) と Disposition を更新。

段階11-D Phase 5 Step 5.2 改修 (正典 PLAN §5 Phase 5):
- OPINIONS / ENTITIES ハードコード枠を撤去、cluster 推定 (posterior、永続化
  しない事後整理) を prompt に挿入する構造に置換
- _build_reflect_sections (tag_registry.reflect_section 駆動) を撤去、
  _build_cluster_sections (estimate_clusters の結果整形) を新設
- reflect 出力は NOTES のみ、network=None で memory_store (untagged path、
  rules 不要、Phase 1 で開けた経路の本格運用)
- reconciliation hook (`_state=state`) は NOTES の memory_store にも継承
- SELF_DISPOSITION / ATTRIBUTED_DISPOSITION 経路は不変 (段階11-A 設計維持)
- ENTITY 経路 (memory_update / memory_network_search via find_similar_facts) 撤去
  (B1 entity 概念完全廃止と整合)

段階11-A Step 4 で導入された経路 (G3 log 分離 / G1 reflect_section) のうち、
G3 (perspective ベース log 分離) は維持、G1 は Phase 5 で撤去。
"""
from datetime import datetime, timezone

from core.cluster_estimation import estimate_clusters
from core.memory import load_all_memories, memory_store
from core.memory_links import list_links
from core.tag_emergence_monitor import compute_cluster_mutual_information
from core.perspective import (
    default_self_perspective,
    is_self_view,
    make_perspective,
)
from core.state import append_debug_log


def should_reflect(state: dict, interval: int = 10) -> bool:
    """内省を実行すべきか判定。"""
    cycles_since = state.get("reflection_cycle", 0)
    if cycles_since >= interval:
        return True
    # 高prediction_errorで前倒し（ただし最低3サイクル間隔）
    # 段階11-D smoke 1 hotfix (2026-04-26): main.py:883 が last_prediction_error を
    # 0-100 scale (e2 の差分絶対値) で書き込むのに対し、本式は 0-1 scale 想定で
    # `> 0.8` 判定だった = 段階10 (a582bc5) 以降ほぼ常時 pre-fire 発動 bug。
    # entropy.py:137 と同じ /100.0 正規化方式に統一して 80% 相当を正しく判定。
    if cycles_since >= 3 and (state.get("last_prediction_error", 0) / 100.0) > 0.8:
        return True
    return False


# ============================================================
# 段階11-A Step 4: log 分離 helpers (G3)
# ============================================================

def _split_log_by_perspective(recent_log: list) -> tuple:
    """log を self 視点 / 非 self 視点に分離 (SEAL 原理)。

    段階11-A G3: tool 名 hardcode に依存せず log entry の perspective 属性で
    分離。perspective 欠落 entry は default_self_perspective() 扱い
    (backward compat)。

    Returns:
        (self_actions, observations): tuple of lists
    """
    self_actions, observations = [], []
    for e in recent_log:
        p = e.get("perspective") or default_self_perspective()
        if is_self_view(p):
            self_actions.append(e)
        else:
            observations.append(e)
    return self_actions, observations


def _format_self_actions(entries: list) -> str:
    """self_actions を prompt 用にフォーマット。"""
    if not entries:
        return "(なし)"
    lines = []
    for e in entries:
        t = e.get("time", "")
        tool = e.get("tool", "?")
        intent = str(e.get("intent", ""))[:100]
        result = str(e.get("result", ""))[:200]
        ev = e.get("eval", {})
        ach = ev.get("achievement", "?") if isinstance(ev, dict) else "?"
        lines.append(f"  {t} {tool}: {intent} → {result} (ach={ach})")
    return "\n".join(lines)


def _format_observations(entries: list) -> str:
    """observations (非 self 視点 log) を viewer と channel 付きでフォーマット。"""
    if not entries:
        return "(なし)"
    lines = []
    for e in entries:
        t = e.get("time", "")
        p = e.get("perspective") or default_self_perspective()
        viewer = p.get("viewer", "?")
        channel = e.get("channel", "?")
        content = str(e.get("result", ""))[:200]
        lines.append(f"  {t} [{viewer} @{channel}] {content}")
    return "\n".join(lines)


# ============================================================
# 段階11-D Phase 5 Step 5.2: cluster 推定セクション
# ============================================================

def _build_cluster_sections(clusters: list, memory_index: dict) -> str:
    """estimate_clusters の結果を reflect prompt 用に整形。

    段階11-D Phase 5 Step 5.2: tag_registry.reflect_section 駆動の旧
    _build_reflect_sections に代わる新経路。cluster は posterior、
    永続化せず reflect 毎に再推定 (PLAN §5 Phase 5)。

    各 cluster は label + 件数 + 代表 sample 2 件の content (60 字 cap) を表示。
    LLM② が「この memory 群の整理」を見て NOTES を返す材料になる。

    Args:
        clusters: estimate_clusters の戻り値
            [{"cluster_id", "label", "memory_ids", "method"}, ...]
        memory_index: memory_id -> memory entry の dict (content 取得用)

    Returns:
        prompt に挿入する文字列。clusters 空 → 空文字列。
    """
    if not clusters:
        return ""
    lines = ["[現在の memory cluster 推定 (posterior、永続化しない事後整理)]"]
    for c in clusters:
        label = c.get("label") or "(未分類)"
        mids = c.get("memory_ids") or []
        lines.append(f"- クラスタ「{label}」 ({len(mids)} 件):")
        for mid in mids[:2]:
            m = memory_index.get(mid)
            if m:
                content = (m.get("content") or "").replace("\n", " ")[:60]
                lines.append(f"    · {content}")
    return "\n".join(lines)


# ============================================================
# Reflect 本体
# ============================================================

def _gather_dispositions_for_prompt(state: dict) -> tuple:
    """prompt 表示用の self_disp / attributed_disps を dict で返す。

    段階11-A Step 5 以降: perspective-keyed state["dispositions"] 単一ソース。
    flat state["disposition"] は load_state の _migrate_disposition_v11a で
    起動時に撤去済み (dual write 期間は Step 4 で終了)。
    """
    dispositions = state.get("dispositions") or {}
    self_disp_raw = dispositions.get("self") or {}
    self_disp_values = {
        k: (v.get("value") if isinstance(v, dict) else v)
        for k, v in self_disp_raw.items()
    }

    # attributed のみ (self を除く)
    attr_disps = {}
    for pkey, traits in dispositions.items():
        if pkey == "self":
            continue
        if isinstance(traits, dict):
            attr_disps[pkey] = {
                k: (v.get("value") if isinstance(v, dict) else v)
                for k, v in traits.items()
            }
    return self_disp_values, attr_disps


def reflect(state: dict, call_llm_fn) -> dict:
    """内省サイクル実行。LLM に直近の行動 + memory cluster 整理を振り返らせる。

    段階11-D Phase 5 Step 5.2: 入力に cluster 推定 (memory list 全体の
    posterior 整理) を追加、出力枠を NOTES (raw 気づき自由形式) に統一。
    OPINIONS / ENTITIES 固定枠は撤去 (B1 entity 廃止 + tag 廃止徹底と整合)。

    戻り値: {"notes": [...], "self_disp_delta": {...}, "attr_disp_delta": {...}}
    """
    import json

    # 段階11-A G3: 直近 log の材料分離 (self/observation)
    recent_log = state.get("log", [])[-10:]
    self_actions, observations = _split_log_by_perspective(recent_log)
    self_actions_text = _format_self_actions(self_actions)
    observations_text = _format_observations(observations)

    # 現在の自己モデル
    self_text = json.dumps(state.get("self", {}), ensure_ascii=False)[:300]

    # dispositions 材料分離
    self_disp_values, attr_disps = _gather_dispositions_for_prompt(state)

    # pending
    pending = state.get("pending", [])
    pending_text = f"{len(pending)}件未対応" if pending else "なし"

    # 段階11-D Phase 5 Step 5.2: cluster 推定 (posterior、永続化しない)
    memories = load_all_memories()
    memory_index = {m.get("id", ""): m for m in memories if m.get("id")}
    clusters = estimate_clusters(
        memories,
        method="hybrid",
        n_clusters=None,
        llm_call_fn=call_llm_fn,
    )
    cluster_sections = _build_cluster_sections(clusters, memory_index)

    # 段階11-D Phase 6 Step 6.2: cluster MI 計算 (観察のみ、log_cycle_metrics 経由 jsonl 永続化)
    # state["phase6_metrics"] に書き、log_cycle_metrics 側が次 cycle 以降拾う
    mi_metrics = compute_cluster_mutual_information(clusters, list_links(limit=10000))
    if isinstance(state, dict):
        state.setdefault("phase6_metrics", {})
        state["phase6_metrics"]["cluster_mi"] = mi_metrics["cluster_mi"]
        state["phase6_metrics"]["cluster_inter_ratio"] = mi_metrics["cluster_inter_ratio"]
        state["phase6_metrics"]["last_cluster_link_pairs"] = mi_metrics["cluster_link_pairs"]
        state["phase6_metrics"]["last_reflect_cycle"] = state.get("cycle_id")

    prompt = f"""あなたは自律AIシステムの内省モジュールです。以下の直近の行動 + memory cluster 整理を振り返り、各項目を出力してください。

[自己モデル]
{self_text}

[自己の傾向 (self)]
{json.dumps(self_disp_values, ensure_ascii=False)}

[帰属された傾向 (attributed)]
{json.dumps(attr_disps, ensure_ascii=False)}

[直近の自己行動]
{self_actions_text}

[直近観察した対象]
{observations_text}

[未対応事項]
{pending_text}

{cluster_sections}

## 振り返りのガイド (情報提示、命令ではない)
- 観察を記述する際、視点属性 (viewer, viewer_type) を付与できる
- 自分視点 (自分が見た何か) と他者視点の想像 (他者がどう見てると自分が思うか) は別エントリになる
- 自分の disposition 変化は、自分の行動と結果から判断 (他者の性質から直接変動させない)
- cluster は posterior 推定 (永続化されない事後整理)、固定カテゴリではない
- 書く/書かないは自由

以下の形式で出力してください。直近の行動と memory 整理から生まれた気づきを自由に記述。

NOTES:
- 〜について〜と気づいた (confidence: 0.7)
- 〜と〜の関連性に気づいた (confidence: 0.8)
- (自由形式、固定カテゴリなし)

SELF_DISPOSITION:
- [観点]_delta: [-0.1~+0.1 の数値]
  ([観点]は今回の内省内容に整合する観点を自由に記述。自分の直近行動の結果から変化した傾向のみ、該当があれば書く、なければ書かない)

ATTRIBUTED_DISPOSITION:
- viewer: 対象の entity name, key: 傾向名, delta: -0.1~+0.1, confidence: 0.0-1.0
  (観察した対象の傾向を self が推定。該当があれば書く、なければ書かない)

短く、具体的に。各 NOTE / DISPOSITION は独立に。"""

    try:
        text = call_llm_fn(prompt, max_tokens=1000, temperature=0.3)
        append_debug_log("Reflection", text)
        parsed = _parse_reflection(text, state)
        # 段階11-D Phase 6 Step 6.2: 戻り値に MI 観察値を含める (新キー追加のみ、既存 test 影響なし)
        parsed.setdefault("cluster_mi", mi_metrics["cluster_mi"])
        parsed.setdefault("cluster_inter_ratio", mi_metrics["cluster_inter_ratio"])
        parsed.setdefault("cluster_link_pairs", mi_metrics["cluster_link_pairs"])
        return parsed
    except Exception as e:
        print(f"  [reflection] エラー: {e}")
        return {
            "notes": [],
            "self_disp_delta": {}, "attr_disp_delta": {},
            "cluster_mi": mi_metrics.get("cluster_mi", 0.0),
            "cluster_inter_ratio": mi_metrics.get("cluster_inter_ratio", 0.0),
            "cluster_link_pairs": mi_metrics.get("cluster_link_pairs", 0),
        }


def _parse_reflection(text: str, state: dict) -> dict:
    """内省結果をパースして記憶・dispositionに反映。

    段階11-D Phase 5 Step 5.2: NOTES 単一枠 (raw 気づき自由形式) で memory_store
    (network=None、untagged path、rules 不要)。OPINIONS / ENTITIES 固定枠は
    撤去 (B1 entity 廃止 + tag 廃止徹底と整合)。
    SELF_DISPOSITION / ATTRIBUTED_DISPOSITION 経路は不変 (段階11-A 設計維持)。
    reconciliation hook (`_state=state`) は NOTES の memory_store にも継承
    (矛盾検出 → EC 誤差、段階11-B Phase 3 設計の延長)。
    """
    import re
    notes = []
    self_disp_delta = {}        # {trait_key: delta}
    attr_disp_delta = {}        # {viewer: {trait_key: (delta, confidence)}}

    # セクションフラグ
    in_notes = False
    in_self_disp = False
    in_attr_disp = False

    for line in text.splitlines():
        stripped = line.strip()
        # ヘッダ判定 (大文字化で順序問題なく)
        upper = stripped.upper()
        if "SELF_DISPOSITION" in upper:
            in_notes = False
            in_self_disp, in_attr_disp = True, False
            continue
        elif "ATTRIBUTED_DISPOSITION" in upper:
            in_notes = False
            in_self_disp, in_attr_disp = False, True
            continue
        elif "NOTES" in upper:
            in_notes = True
            in_self_disp, in_attr_disp = False, False
            continue

        if in_notes and stripped.startswith("-"):
            content = stripped.lstrip("- ").strip()
            confidence = 0.7
            m = re.search(r'confidence:\s*([\d.]+)', content, re.IGNORECASE)
            if m:
                confidence = float(m.group(1))
                content = re.sub(r'\(?\s*confidence:\s*[\d.]+\s*\)?', '', content).strip()
            if content:
                entry = memory_store(
                    network=None, content=content,
                    metadata={"confidence": confidence},
                    origin="reflection", source_context="self_inference",
                    perspective=default_self_perspective(),
                    _state=state,  # 段階11-B Phase 3: reconciliation hook (矛盾検出 → EC 誤差)
                )
                notes.append(entry)
                print(f"  [reflection] note: {content[:60]} (conf={confidence})")

        elif in_self_disp and stripped.startswith("-"):
            m = re.search(r'(\w+)_delta:\s*([-+]?[\d.]+)', stripped)
            if m:
                key = m.group(1)
                delta = max(-0.1, min(0.1, float(m.group(2))))
                self_disp_delta[key] = delta

        elif in_attr_disp and stripped.startswith("-"):
            # 形式: viewer: X, key: Y, delta: +0.05, confidence: 0.6
            m = re.search(
                r'viewer:\s*([^,]+?),\s*key:\s*([^,]+?),\s*delta:\s*([-+]?[\d.]+)'
                r'(?:,\s*confidence:\s*([\d.]+))?',
                stripped,
            )
            if m:
                viewer = m.group(1).strip()
                key = m.group(2).strip()
                delta = max(-0.1, min(0.1, float(m.group(3))))
                conf = float(m.group(4)) if m.group(4) else 0.5
                if viewer and key:
                    attr_disp_delta.setdefault(viewer, {})[key] = (delta, conf)

    # ====================================================================
    # SELF_DISPOSITION 反映 (perspective-keyed 単一ソース、Step 5 以降)
    # ====================================================================
    # 段階11-A Step 5: flat state["disposition"] への dual write を撤去、
    # perspective-keyed state["dispositions"]["self"] を単一 source of truth。
    # 起動時 migration (_migrate_disposition_v11a) で旧 flat dict は削除済み。
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dispositions = state.setdefault("dispositions", {})
    self_disp = dispositions.setdefault("self", {})

    for key, delta in self_disp_delta.items():
        cur = self_disp.get(key)
        cur_val = cur.get("value", 0.5) if isinstance(cur, dict) else 0.5
        new_val = max(0.1, min(0.9, cur_val + delta))
        self_disp[key] = {
            "value": new_val,
            "confidence": None,
            "perspective": default_self_perspective(),
            "updated_at": now_iso,
        }

    if self_disp_delta:
        print(f"  [reflection] self_disposition delta: {self_disp_delta}")

    # ====================================================================
    # ATTRIBUTED_DISPOSITION 反映 (perspective-keyed 専用、flat dual write なし)
    # ====================================================================
    for viewer, deltas in attr_disp_delta.items():
        pkey = f"attributed:{viewer}"
        viewer_disp = dispositions.setdefault(pkey, {})
        viewer_persp = make_perspective(viewer=viewer, viewer_type="actual")
        for key, (delta, conf) in deltas.items():
            cur = viewer_disp.get(key)
            if isinstance(cur, dict):
                cur_val = cur.get("value", 0.5)
            else:
                cur_val = 0.5
            new_val = max(0.1, min(0.9, cur_val + delta))
            viewer_disp[key] = {
                "value": new_val,
                "confidence": conf,
                "perspective": viewer_persp,
                "updated_at": now_iso,
            }

    if attr_disp_delta:
        print(f"  [reflection] attributed_disposition delta: {attr_disp_delta}")

    return {
        "notes": notes,
        "self_disp_delta": self_disp_delta,
        "attr_disp_delta": attr_disp_delta,
    }
