"""State管理・好み関数・デバッグログ"""
import json
import os
import tempfile
import threading
import time
import uuid
import re
from functools import wraps
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


def _rebuild_views_from_jsonl(state: dict, *, raw_read_report=None) -> None:
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
        from core.memory_read import read_memory_jsonl
        state[key] = read_memory_jsonl(
            MEMORY_DIR / filename,
            report=raw_read_report if key == "raw_events" else None,
        )


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


STATE_LOCK = threading.RLock()
_live_state = None
_live_path = None


def state_locked(func):
    """One reentrant lock for tool transactions and state persistence."""
    @wraps(func)
    def locked(*args, **kwargs):
        with STATE_LOCK:
            return func(*args, **kwargs)
    return locked


@state_locked
def bind_live_state(state):
    global _live_state, _live_path
    _live_state = state
    _live_path = STATE_FILE.resolve() if state is not None else None


def _bound_live():
    return _live_state if _live_path == STATE_FILE.resolve() else None


def _migrate_state(data):
    if not isinstance(data, dict):
        raise ValueError("state must be an object")
    if type(data.get("cycle_id", 0)) is not int or data.get("cycle_id", 0) < 0:
        raise ValueError("invalid cycle_id")
    if not isinstance(data.get("_persistence", {}), dict):
        raise ValueError("invalid persistence record")
    _name = _get_name_from_seed()
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
    # Slice 6.5 Step 6 (PLAN §5.4): preference distribution C の動的更新 state
    # _efe_self_confidence: update_self 時の per-key confidence 蓄積 (NAME_KEY 除外)
    # _efe_C: compute_C_from_self の返り値 dict、初回 update_self で構築、それまで None
    # _efe_C_update_cycle: 最後に C 更新した cycle_id (未更新は -1 sentinel)
    if "_efe_self_confidence" not in data:
        data["_efe_self_confidence"] = {}
    if "_efe_C" not in data:
        data["_efe_C"] = None
    if "_efe_C_update_cycle" not in data:
        data["_efe_C_update_cycle"] = -1
    # 段階11-A Step 5: disposition (flat) → dispositions (perspective-keyed) 移行
    _migrate_disposition_v11a(data)
    return data


def _read_state(path):
    return _decode_state(path.read_text(encoding="utf-8"))


def _decode_state(text):
    def invalid_constant(value):
        raise ValueError("non-finite JSON number")
    return _migrate_state(json.loads(text, parse_constant=invalid_constant))


def _fresh_state():
    _name = _get_name_from_seed()
    from core.world_model import init_world_model
    fresh = {"raw_events": [], "subjective_entries": [], "self": {"name": _name}, "energy": 50, "summaries": [], "cycle_id": 0, "tool_level": 0, "voluntary_memory_store_count": 0, "files_read": [], "files_written": [], "last_notification_fetch": "", "pressure": 0.0, "last_e1": 0.5, "last_e2": 0.5, "last_e3": 0.5, "last_e4": 0.5, "entropy": 0.65, "drives_state": {}, "world_model": init_world_model(), "predictor_confidence": {}, "prediction_error_history_e2": [], "prediction_error_history_ec": [], "_efe_self_confidence": {}, "_efe_C": None, "_efe_C_update_cycle": -1, "dispositions": {"self": {}}}
    # 段階13 Phase 0.1.B: state.json が無くても jsonl があれば rebuild
    # (state.json 削除 + memory/ 残存ケースの safety net)
    _rebuild_views_from_jsonl(fresh, raw_read_report=None)
    return fresh


@state_locked
def load_state() -> dict:
    """Main's live object, or a read-only load for standalone readers."""
    live = _bound_live()
    if live is not None:
        return live
    try:
        data = _read_state(STATE_FILE)
    except (FileNotFoundError, ValueError, TypeError, AttributeError, UnicodeError):
        return _fresh_state()
    _rebuild_views_from_jsonl(data)
    return data


def _generations():
    return sorted((MEMORY_DIR / "state_generations").glob("state-*.json"), reverse=True)


def _retain_generation():
    try:
        text = STATE_FILE.read_text(encoding="utf-8")
        _decode_state(text)
    except (FileNotFoundError, ValueError, TypeError, AttributeError, UnicodeError):
        return
    directory = MEMORY_DIR / "state_generations"
    directory.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("state-%Y%m%dT%H%M%S%f-") + uuid.uuid4().hex + ".json"
    _atomic_write(directory / name, text)
    for path in _generations()[3:]:
        path.unlink()


def _persistence_failure(state, exc, phase):
    fact = {"time": datetime.now(timezone.utc).isoformat(),
            "exception_type": type(exc).__name__, "phase": phase}
    state.setdefault("_persistence", {})["last_failure"] = fact
    print(f"  [state] persistence failed: {phase} ({type(exc).__name__})")


def _preserve_corrupt(fact):
    preserved = STATE_FILE.with_name("state.corrupt-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
    preserved.write_bytes(STATE_FILE.read_bytes())
    fact["preserved"] = preserved.name


@state_locked
def save_state(state: dict) -> bool:
    phase = "identity"
    try:
        live = _bound_live()
        if live is not None and state is not live:
            raise ValueError("save requires the bound live state")
        phase = "preserve_corrupt"
        startup = state.get("_persistence", {}).get("startup", {})
        if startup.get("reason") == "invalid_state" and not startup.get("preserved"):
            _preserve_corrupt(startup)
        phase = "generation"
        if live is not None:
            _retain_generation()
        phase = "views"
        from core.view_persistence import flush_dirty_views
        flush_dirty_views(state)
        phase = "state"
        _atomic_write(STATE_FILE, json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False))
        return True
    except Exception as exc:
        target = _bound_live()
        _persistence_failure(target if target is not None else state, exc, phase)
        return False


@state_locked
def load_startup_state() -> dict:
    """Main only: recover; common load_state never writes or recovers files."""
    bind_live_state(None)
    failure = None
    data = None
    source = "state.json"
    for attempt in range(3):
        try:
            data = _read_state(STATE_FILE)
            if failure is not None:
                failure["attempts"] = attempt + 1
            break
        except FileNotFoundError:
            failure = {"reason": "missing"}
            break
        except OSError as exc:
            failure = {"reason": "os_error", "exception_type": type(exc).__name__, "attempts": attempt + 1}
            if attempt < 2:
                time.sleep(0.05)
        except (ValueError, TypeError, AttributeError, UnicodeError) as exc:
            failure = {"reason": "invalid_state", "exception_type": type(exc).__name__}
            break
    if data is None:
        try:
            candidates = _generations()
        except OSError:
            candidates = []
        for path in candidates:
            try:
                data = _read_state(path)
                source = str(path.relative_to(MEMORY_DIR))
                break
            except (OSError, ValueError, TypeError, AttributeError, UnicodeError):
                continue
        if failure["reason"] == "invalid_state":
            try:
                _preserve_corrupt(failure)
            except OSError as exc:
                failure["preservation_error"] = type(exc).__name__
        if data is None:
            data = _fresh_state()
            source = "initial"
    _rebuild_views_from_jsonl(data)
    latest = int(data.get("cycle_id", 0) or 0)
    try:
        paths = list(MEMORY_DIR.glob("*.jsonl"))
    except OSError:
        paths = []
    for path in paths:
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            continue
                        for key in ("cycle_id", "cycle"):
                            value = row.get(key)
                            if type(value) is int:
                                latest = max(latest, value)
                        # main cycle entries: <session>_<cycle>; auxiliary events
                        # add another suffix and do not match this pattern.
                        if path.name == "raw_events.jsonl":
                            match = re.fullmatch(r"[^_]+_(\d+)", str(row.get("id", "")))
                            if match:
                                latest = max(latest, int(match[1]))
                    except (ValueError, TypeError):
                        continue
        except (OSError, UnicodeError):
            continue
    data["cycle_id"] = latest
    data.setdefault("_persistence", {}).pop("startup", None)
    if failure is not None:
        data["_persistence"]["startup"] = {**failure, "source": source,
            "time": datetime.now(timezone.utc).isoformat(), "cycle_id": latest}
    return data


def record_startup_recovery(state):
    fact = state.get("_persistence", {}).get("startup")
    if fact is None:
        return
    event = {**fact, "event_type": "state_recovery", "run_id": state.get("run_id")}
    from core.metrics import _emit_error_observation
    from core.memory import _record_entry
    _emit_error_observation(event)
    entry = {"id": "state-recovery-" + uuid.uuid4().hex,
             "time": fact["time"], "channel": "system", "tool": "state_recovery",
             "result": json.dumps(event, ensure_ascii=False)}
    try:
        _record_entry(state, entry)
    except Exception as exc:
        if not any(row.get("id") == entry["id"] for row in state.get("raw_events", [])):
            state.setdefault("raw_events", []).append(entry)
        _persistence_failure(state, exc, "recovery_record")



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
