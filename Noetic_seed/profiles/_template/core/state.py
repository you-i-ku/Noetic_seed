"""State管理・好み関数・デバッグログ"""
import json
import os
import tempfile
from datetime import datetime, timezone
from core.config import STATE_FILE, PREF_FILE, DEBUG_LOG, SEED_FILE, MEMORY_DIR


def _migrate_disposition_v11a(state: dict) -> None:
    """段階11-A Step 5: 旧 state['disposition'] (flat) → state['dispositions']['self']
    (perspective-keyed) へ移行する起動時 migration。冪等。

    正典 PLAN: STAGE11A_PERSPECTIVE_FOUNDATION_PLAN.md §5-1

    移行挙動:
      - state['dispositions'] 未存在 → 初期化 ({"self": {}})
      - state['disposition'] (flat) 存在 → self に未反映の trait のみ移行
        (conflict 時 dispositions 側優先 = 既存 Step4 書き込みを尊重)
      - 移行後、state['disposition'] (flat) を完全撤去 (`pop`)
      - 既に dispositions だけの state → no-op (冪等)
    """
    from core.perspective import default_self_perspective
    dispositions = state.setdefault("dispositions", {})
    dispositions.setdefault("self", {})

    old = state.pop("disposition", None)
    if isinstance(old, dict) and old:
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for k, v in old.items():
            if k in dispositions["self"]:
                continue  # 既に pkeyed 側にある → 上書きしない (Step 4 書き込み尊重)
            try:
                val = float(v)
            except (TypeError, ValueError):
                val = 0.5
            dispositions["self"][k] = {
                "value": max(0.1, min(0.9, val)),
                "confidence": None,
                "perspective": default_self_perspective(),
                "updated_at": now_iso,
            }


def _rebuild_views_from_jsonl(state: dict) -> None:
    """段階13 Phase 0.1.B: raw_events.jsonl / subjective_entries.jsonl から
    state["raw_events"] / state["subjective_entries"] view を再構築する。

    PLAN §18-2 「段階7 materialized view パターン継承」の起動時 in-memory rebuild。
    jsonl が source of truth (PLAN §20-2 #4 raw 永続 / subjective compaction 可)
    なので state.json の view 値は信頼せず rebuild で完全置換する。

    挙動:
      - jsonl 不在 (clean profile) → state[key] = []
      - 壊れた行 (json parse 失敗) → skip して続行 (load_all_memories 流儀)
      - 読み出し例外 (encoding 等) → state[key] = [] にフォールバック
    """
    for key, filename in (
        ("raw_events", "raw_events.jsonl"),
        ("subjective_entries", "subjective_entries.jsonl"),
    ):
        fpath = MEMORY_DIR / filename
        if not fpath.exists():
            state[key] = []
            continue
        try:
            lines = fpath.read_text(encoding="utf-8").splitlines()
        except Exception:
            state[key] = []
            continue
        rebuilt = []
        for line in lines:
            if not line.strip():
                continue
            try:
                rebuilt.append(json.loads(line))
            except Exception:
                continue
        state[key] = rebuilt


def merge_log_view(state: dict) -> list:
    """段階13 Phase 0.1.D: raw_events と subjective_entries を id で zip した
    合本 view を返す。controller / prompt 等、両層の field を同時に見たい
    consumer 用 (旧 state["log"] 撤去後の代替)。

    哲学整合: raw (immutable, source of truth) と subjective (mutable, materialized
    view) は別 jsonl で永続、同 id で対応関係を保つ。merge は read 時のみの
    on-the-fly 合成、永続化しない (段階7 materialized view pattern 同精神)。

    片側のみ存在する id (raw のみ / subjective のみ) は欠落 field を空 dict 補完で
    そのまま含める (defensive、起動直後 jsonl 同期前等の過渡期向け)。

    注意 (Codex review 0.1.D Hypothesis):
      - id collision (タイムスタンプ modulo 系の id 衝突) が起きると、raw_by_id
        辞書化で earlier raw event が silently 上書きされる。現状 id 生成は
        cycle_id + ms (main.py:606/1014/1119/1188/1250) で衝突確率は実質ゼロ
        だが、id 生成器側で uniqueness を保つ前提で本 helper は dict 構築を採用。
      - raw-only entry (subjective 未生成) は subjective 主軸の後に append される
        ため、`[-N:]` 系 reader からは「最新」のように見える。過渡期 only の状態
        だが、raw-only が大量残存する状況なら別途 sort 戦略を検討。
    """
    raw_by_id = {e.get("id"): e for e in state.get("raw_events", []) if e.get("id")}
    subj_by_id = {e.get("id"): e for e in state.get("subjective_entries", []) if e.get("id")}
    merged: list = []
    seen: set = set()
    # subjective_entries の順序を主軸にする (compaction 対象 = 「iku の経験」順)
    for s in state.get("subjective_entries", []):
        sid = s.get("id")
        if sid is None:
            merged.append(dict(s))
            continue
        r = raw_by_id.get(sid, {})
        merged.append({**r, **s})
        seen.add(sid)
    # raw のみ存在する entry (subjective 未生成、過渡期 only) を末尾に
    for r in state.get("raw_events", []):
        rid = r.get("id")
        if rid is not None and rid not in seen:
            merged.append(dict(r))
    return merged


def _atomic_write(path, text: str):
    """tmp ファイルに書いてから os.replace でアトミックに差し替える。"""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _get_name_from_seed() -> str:
    """seed.txtの1行目からnameを取得。「name:」形式なら:の前、なければ1行目全体。空なら空文字。"""
    if SEED_FILE.exists():
        try:
            first_line = SEED_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()
            if ":" in first_line:
                return first_line.split(":")[0].strip()
            if first_line:
                return first_line
        except Exception:
            pass
    return ""


def load_state() -> dict:
    _name = _get_name_from_seed()
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            # 段階13 Phase 0.1.D: state["log"] 撤去済。raw_events.jsonl (immutable
            # source of truth) + subjective_entries.jsonl (materialized view、
            # compaction 対象) の二層分離。in-memory view は _rebuild_views_from_jsonl
            # で起動時に必ず jsonl から rebuild する (PLAN §20-2 #4)。
            if "raw_events" not in data:
                data["raw_events"] = []
            if "subjective_entries" not in data:
                data["subjective_entries"] = []
            # 旧 state["log"] が残ってる profile は無視 (jsonl が source of truth)
            data.pop("log", None)
            if "self" not in data:
                data["self"] = {"name": _name}
            elif "name" not in data["self"]:
                data["self"]["name"] = _name
            if "energy" not in data:
                data["energy"] = 50
            if "summaries" not in data:
                data["summaries"] = []
            if "cycle_id" not in data:
                data["cycle_id"] = 0
            if "tool_level" not in data:
                data["tool_level"] = 0
            # 段階11-D Phase 0 Step 0.4: memory_graph affordance ガード (B2)
            # 自発 memory_store 経験 counter (失敗は count しない、Z2)
            if "voluntary_memory_store_count" not in data:
                data["voluntary_memory_store_count"] = 0
            if "files_read" not in data:
                data["files_read"] = []
            if "files_written" not in data:
                data["files_written"] = []
            if "last_notification_fetch" not in data:
                data["last_notification_fetch"] = ""
            if "pressure" not in data:
                data["pressure"] = 0.0
            if "last_e1" not in data:
                data["last_e1"] = 0.5
            if "last_e2" not in data:
                data["last_e2"] = 0.5
            if "last_e3" not in data:
                data["last_e3"] = 0.5
            if "last_e4" not in data:
                data["last_e4"] = 0.5
            if "entropy" not in data:
                data["entropy"] = 0.65
            if "drives_state" not in data:
                data["drives_state"] = {}
            if "world_model" not in data:
                from core.world_model import init_world_model
                data["world_model"] = init_world_model()
            # 段階10 柱 B: Predictor 自己学習の state 拡張
            if "predictor_confidence" not in data:
                data["predictor_confidence"] = {}
            # Slice 3 (orchestration §3 P1 #3): info_gain 前 cycle snapshot
            if "_info_gain_prev" not in data:
                data["_info_gain_prev"] = {}
            if "prediction_error_history_e2" not in data:
                data["prediction_error_history_e2"] = []
            if "prediction_error_history_ec" not in data:
                data["prediction_error_history_ec"] = []
            # 段階11-A Step 5: disposition (flat) → dispositions (perspective-keyed) 移行
            _migrate_disposition_v11a(data)
            # 段階13 Phase 0.1.B: jsonl が source of truth、in-memory view は
            # 起動時に必ず rebuild (state.json 内の値は信頼せず上書き、PLAN §18-2)
            _rebuild_views_from_jsonl(data)
            # Slice 5 (orchestration §5 Slice 5): Goal Shadow Observer
            # 観測のみ、controller 不変 (Slice 6 で resonance multiplier 弱接続予定)
            # Codex 3 周目 P2 fix (2026-05-10): snapshot init は _rebuild_views_from_jsonl
            # の後で行う必要がある。rebuild が subjective_entries を上書きするため、
            # rebuild 前に snapshot 取ると state.json なし / stale 復旧パスで 0 (or stale)
            # 値となり、次 cycle で全履歴が "新規 evidence" として goal_shadow 量産される
            # silent bug を防ぐ。既存 state.json で前 cycle 末の値を継承してる場合は
            # setdefault でスキップ (in_data 判定により尊重)。
            if "goal_shadows" not in data:
                data["goal_shadows"] = []
            if "_files_written_count_prev" not in data:
                data["_files_written_count_prev"] = len(data.get("files_written", []) or [])
            if "_subjective_entries_count_prev" not in data:
                data["_subjective_entries_count_prev"] = len(data.get("subjective_entries", []) or [])
            return data
        except json.JSONDecodeError:
            pass
    from core.world_model import init_world_model
    fresh = {"raw_events": [], "subjective_entries": [], "self": {"name": _name}, "energy": 50, "summaries": [], "cycle_id": 0, "tool_level": 0, "voluntary_memory_store_count": 0, "files_read": [], "files_written": [], "last_notification_fetch": "", "pressure": 0.0, "last_e1": 0.5, "last_e2": 0.5, "last_e3": 0.5, "last_e4": 0.5, "entropy": 0.65, "drives_state": {}, "world_model": init_world_model(), "predictor_confidence": {}, "prediction_error_history_e2": [], "prediction_error_history_ec": [], "dispositions": {"self": {}}, "goal_shadows": [], "_files_written_count_prev": 0, "_subjective_entries_count_prev": 0}
    # 段階13 Phase 0.1.B: state.json が無くても jsonl があれば rebuild
    # (state.json 削除 + memory/ 残存ケースの safety net)
    _rebuild_views_from_jsonl(fresh)
    # Slice 5 Codex 3 周目 P2 fix: rebuild 後に snapshot を実 jsonl 件数で更新。
    # state.json 不在 + jsonl に履歴残存ケースで _subjective_entries_count_prev=0
    # のままだと、次 cycle で全履歴を "新規 evidence" として goal_shadow 量産する
    # silent bug を防ぐ (load_state 経路と同精神、対称性維持)。
    fresh["_files_written_count_prev"] = len(fresh.get("files_written", []) or [])
    fresh["_subjective_entries_count_prev"] = len(fresh.get("subjective_entries", []) or [])
    return fresh


def save_state(state: dict):
    # F-005: dirty な materialized view (subjective_entries 等) を atomic rewrite。
    # save_state は cycle 末以外にも複数回呼ばれるが、flush は dirty 立ってる view
    # のみ書くので no-op cost は無視できる範囲 (set membership check + 早期 return)。
    # state.json 書き出しより先に flush する: jsonl が source of truth (load_state
    # 側で _rebuild_views_from_jsonl が state.json view を ignore するため)、jsonl
    # を先に persist しておけば state.json 失敗でも view 整合は保たれる。
    try:
        from core.view_persistence import flush_dirty_views
        flush_dirty_views(state)
    except Exception as e:
        # flush 失敗は state.json 保存を阻害しない (defensive、reflect 継続原則)。
        print(f"  [view_persistence] flush skip (error: {e})")
    _atomic_write(STATE_FILE, json.dumps(state, ensure_ascii=False, indent=2))


def load_pref() -> dict:
    if PREF_FILE.exists():
        try:
            return json.loads(PREF_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_pref(pref: dict):
    _atomic_write(PREF_FILE, json.dumps(pref, ensure_ascii=False, indent=2))


def append_debug_log(phase: str, text: str):
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {phase} =====\n{text}\n")
    except Exception:
        pass
