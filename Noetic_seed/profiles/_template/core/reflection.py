"""内省モジュール v2 — 定期的な自己省察（Generative Agents + Reflexion統合）
Nサイクルごと or 高prediction_error時に発火。
Opinion/Entity/Dispositionを更新。

段階11-A Step 4 改修 (正典 PLAN §6):
- G1: reflect prompt を tag_registry.reflect_section 駆動化 (OPINIONS/ENTITIES
  ハードコード撤廃、11-B の動的 tag 発明に同経路で吸収される構造)
- G3: log 材料を log entry.perspective 属性でフロート分離 (tool 名 hardcode 排除、
  _split_log_by_perspective)。SEAL 原理 — self_actions が自己 disposition 更新の
  材料、observations が他者視点 (attributed) 推定の材料
- perspective-keyed dispositions への書き込み (state["dispositions"]["self"] +
  attributed:<viewer>)、Step 4→5 移行期間 flat state["disposition"] への dual write
- LLM② prompt に視点概念の情報を追加 (指示ではなく知識として、原則 P2)
"""
from datetime import datetime, timezone

from core.memory import memory_store, memory_update, memory_network_search
from core.perspective import (
    default_self_perspective,
    is_self_view,
    make_perspective,
)
from core.state import load_state, save_state, append_debug_log


def should_reflect(state: dict, interval: int = 10) -> bool:
    """内省を実行すべきか判定。"""
    cycles_since = state.get("reflection_cycle", 0)
    if cycles_since >= interval:
        return True
    # 高prediction_errorで前倒し（ただし最低3サイクル間隔）
    if cycles_since >= 3 and state.get("last_prediction_error", 0) > 0.8:
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
# 段階11-A Step 4: tag_registry 駆動 reflect セクション (G1)
# ============================================================

def _build_reflect_sections(visibility_mode: str = "visible") -> str:
    """tag_registry.reflect_section から reflect prompt の動的セクション組立。

    段階11-A G1: opinion/entity セクションを tag_registry 駆動化
    (hardcode 撤廃)。段階11-B で AI が新 tag を発明し reflect_section を
    付けた場合も、同経路で自動的に prompt に載る (11-B 受け皿)。

    段階11-C G-lite Phase 3: visibility_mode で tag prior の可視性を切替可能。

    Args:
        visibility_mode:
            "visible"    — 既存挙動 (デフォルト、reflect_section を全組立)
            "cold_start" — 空文字列を返す、iku が既存 tag prior なしに reflect
                            できる実験モード (settings.reflection.reflect_cold_start_mode
                            から配線)。reflect_section ハードコードな OPINIONS /
                            ENTITIES のみ影響、prompt 本体の指示文は変化なし。
                            reflect 抽象再設計は G-full (11-D Phase 5)。
    """
    if visibility_mode == "cold_start":
        return ""
    from core.tag_registry import list_registered_tags, get_tag_rules
    sections = []
    for tag in list_registered_tags():
        rules = get_tag_rules(tag) or {}
        sec = rules.get("reflect_section")
        if not sec or not sec.get("enabled_in_reflect"):
            continue
        header = sec.get("header", tag.upper())
        template = sec.get("template", "- (自由記述)")
        sections.append(f"{header}:\n{template}")
    return "\n\n".join(sections)


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
    """内省サイクル実行。LLMに最近の経験を振り返らせる。

    段階11-A Step 4: 材料を self_actions / observations に分離 (G3)、
    OPINIONS/ENTITIES セクションを tag_registry 駆動 (G1)、
    SELF_DISPOSITION / ATTRIBUTED_DISPOSITION 2 セクションで disposition
    更新材料を分離 (PLAN §6-3)。

    戻り値: {"opinions": [...], "entities": [...],
             "self_disp_delta": {...}, "attr_disp_delta": {...}}
    """
    import json

    # 段階11-B Phase 5 hotfix: reflect が生成する opinion/entity tag を未登録なら
    # inline register (register_standard_tags() 撤去後の silent fail 対策)。
    # iku 自発 tag 発明 (memory_store 経由) の余地を残すため wm/experience は
    # 登録せず、reflect が実際に書き込む 2 tag のみ限定。
    from core.tag_registry import is_tag_registered, register_tag, STANDARD_TAGS
    for _tag in ("opinion", "entity"):
        if not is_tag_registered(_tag):
            _cfg = STANDARD_TAGS.get(_tag)
            if _cfg:
                try:
                    register_tag(
                        _tag,
                        learning_rules=_cfg["learning_rules"],
                        display_format=_cfg.get("display_format", ""),
                        origin="standard",
                        reflect_section=_cfg.get("reflect_section"),
                    )
                except ValueError:
                    pass

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

    # 段階11-A G1: tag_registry 駆動の動的セクション
    # 段階11-C G-lite Phase 3: settings.reflection.reflect_cold_start_mode で
    # 既存 tag prior (OPINIONS/ENTITIES section) の可視性を切替、opt-in 実験。
    try:
        from core.config import llm_cfg
        _cold_start = bool(
            (llm_cfg.get("reflection", {}) or {}).get("reflect_cold_start_mode", False)
        )
    except Exception:
        _cold_start = False
    dynamic_sections = _build_reflect_sections(
        visibility_mode="cold_start" if _cold_start else "visible"
    )

    prompt = f"""あなたは自律AIシステムの内省モジュールです。以下の直近の行動を振り返り、各項目を出力してください。

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

## 振り返りのガイド (情報提示、命令ではない)
- 観察を記述する際、視点属性 (viewer, viewer_type) を付与できる
- 自分視点 (自分が見た何か) と他者視点の想像 (他者がどう見てると自分が思うか) は別エントリになる
- 自分の disposition 変化は、自分の行動と結果から判断 (他者の性質から直接変動させない)
- 書く/書かないは自由

以下の形式で出力してください。直近の行動の文脈に沿った内容のみ。無関係なエンティティの更新は不要。

{dynamic_sections}

SELF_DISPOSITION:
- curiosity_delta: -0.1~+0.1 (自分の直近行動の結果から変化した自分の傾向のみ)
- skepticism_delta: -0.1~+0.1
- sociality_delta: -0.1~+0.1

ATTRIBUTED_DISPOSITION:
- viewer: 対象の entity name, key: 傾向名, delta: -0.1~+0.1, confidence: 0.0-1.0
  (観察した対象の傾向を self が推定。空でも良い)

短く、具体的に。繰り返し禁止。"""

    try:
        text = call_llm_fn(prompt, max_tokens=1000, temperature=0.3)
        append_debug_log("Reflection", text)
        return _parse_reflection(text, state)
    except Exception as e:
        print(f"  [reflection] エラー: {e}")
        return {
            "opinions": [], "entities": [],
            "self_disp_delta": {}, "attr_disp_delta": {},
        }


def _parse_reflection(text: str, state: dict) -> dict:
    """内省結果をパースして記憶・dispositionに反映。

    段階11-A Step 4: SELF_DISPOSITION / ATTRIBUTED_DISPOSITION の 2 セクションを
    別々にパース。OPINIONS / ENTITIES の memory_store 時に perspective=self を
    明示付与。
    disposition 反映は perspective-keyed state["dispositions"] に書き込み、
    Step 4→5 移行期間は flat state["disposition"] にも dual write。
    """
    import re
    opinions = []
    entities = []
    self_disp_delta = {}        # {trait_key: delta}
    attr_disp_delta = {}        # {viewer: {trait_key: (delta, confidence)}}

    # セクションフラグ
    in_opinions = False
    in_entities = False
    in_self_disp = False
    in_attr_disp = False

    for line in text.splitlines():
        stripped = line.strip()
        # ヘッダ判定 (大文字化で順序問題なく)
        upper = stripped.upper()
        if "SELF_DISPOSITION" in upper:
            in_opinions, in_entities = False, False
            in_self_disp, in_attr_disp = True, False
            continue
        elif "ATTRIBUTED_DISPOSITION" in upper:
            in_opinions, in_entities = False, False
            in_self_disp, in_attr_disp = False, True
            continue
        elif "OPINIONS" in upper:
            in_opinions, in_entities = True, False
            in_self_disp, in_attr_disp = False, False
            continue
        elif "ENTITIES" in upper:
            in_opinions, in_entities = False, True
            in_self_disp, in_attr_disp = False, False
            continue

        if in_opinions and stripped.startswith("-"):
            content = stripped.lstrip("- ").strip()
            confidence = 0.7
            m = re.search(r'confidence:\s*([\d.]+)', content, re.IGNORECASE)
            if m:
                confidence = float(m.group(1))
                content = re.sub(r'\(?\s*confidence:\s*[\d.]+\s*\)?', '', content).strip()
            if content:
                entry = memory_store(
                    "opinion", content, {"confidence": confidence},
                    origin="reflection", source_context="self_inference",
                    perspective=default_self_perspective(),
                    _state=state,  # 段階11-B Phase 3: reconciliation hook (矛盾検出 → EC 誤差)
                )
                opinions.append(entry)
                print(f"  [reflection] opinion: {content[:60]} (conf={confidence})")

        elif in_entities and stripped.startswith("-"):
            content = stripped.lstrip("- ").strip()
            m = re.search(r'name:\s*([^,]+),?\s*content:\s*(.*)', content, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                desc = m.group(2).strip()
                if name and desc:
                    from core.tag_registry import get_tags_with_rule
                    entity_tags = get_tags_with_rule("c_gradual_source")
                    if entity_tags:
                        existing = memory_network_search(name, networks=entity_tags, limit=3)
                    else:
                        existing = []
                    matched = [e for e in existing
                               if e.get("metadata", {}).get("entity_name", "") == name]
                    if matched:
                        entry_id = matched[0].get("id", "")
                        memory_update(entry_id, content=desc)
                        entities.append(matched[0])
                        print(f"  [reflection] entity update: {name} = {desc[:60]}")
                    else:
                        entry = memory_store(
                            "entity", desc, {"entity_name": name},
                            origin="reflection", source_context="self_inference",
                            perspective=default_self_perspective(),
                            _state=state,  # 段階11-B Phase 3: reconciliation hook (矛盾検出 → EC 誤差)
                        )
                        entities.append(entry)
                        print(f"  [reflection] entity new: {name} = {desc[:60]}")

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

    # WM 段階3-4: C-gradual 同期 (memory/entity → state["world_model"].entities)
    # 段階4: Entity Resolver で "ゆう" と "YOU" 等を embedding 経由で merge。
    # LLM tiebreak は off (コスト優先、ambiguous は新規扱いで安全側)。
    # 失敗しても reflect 継続 (WM は特権化しない方針)。
    try:
        from core.memory import list_records
        from core.embedding import _embed_sync, cosine_similarity, is_vector_ready
        from core.world_model import sync_from_memory_entities
        wm = state.get("world_model")
        if wm:
            from core.tag_registry import get_tags_with_rule
            c_gradual_tags = get_tags_with_rule("c_gradual_source")
            records = []
            for tag in c_gradual_tags:
                records.extend(list_records(tag, limit=20))
            _vr = is_vector_ready()
            _embed = _embed_sync if _vr else None
            _cosine = cosine_similarity if _vr else None
            created = sync_from_memory_entities(
                wm, records, limit=20,
                embed_fn=_embed, cosine_fn=_cosine,
                llm_call_fn=None,
            )
            if created:
                print(f"  [reflection] WM C-gradual: {created} 件新規取込")
    except Exception as e:
        print(f"  [reflection] WM C-gradual スキップ (エラー: {e})")

    return {
        "opinions": opinions,
        "entities": entities,
        "self_disp_delta": self_disp_delta,
        "attr_disp_delta": attr_disp_delta,
    }
