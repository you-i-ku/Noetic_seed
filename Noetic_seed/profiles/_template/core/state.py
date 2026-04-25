"""State管理・好み関数・デバッグログ"""
import json
import os
import tempfile
from datetime import datetime, timezone
from core.config import STATE_FILE, PREF_FILE, DEBUG_LOG, SEED_FILE


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
            if "log" not in data:
                data["log"] = []
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
            if "tools_created" not in data:
                data["tools_created"] = []
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
            if "prediction_error_history_e2" not in data:
                data["prediction_error_history_e2"] = []
            if "prediction_error_history_ec" not in data:
                data["prediction_error_history_ec"] = []
            # 段階11-A Step 5: disposition (flat) → dispositions (perspective-keyed) 移行
            _migrate_disposition_v11a(data)
            return data
        except json.JSONDecodeError:
            pass
    from core.world_model import init_world_model
    return {"log": [], "self": {"name": _name}, "energy": 50, "summaries": [], "cycle_id": 0, "tool_level": 0, "voluntary_memory_store_count": 0, "files_read": [], "files_written": [], "last_notification_fetch": "", "pressure": 0.0, "last_e1": 0.5, "last_e2": 0.5, "last_e3": 0.5, "last_e4": 0.5, "tools_created": [], "entropy": 0.65, "drives_state": {}, "world_model": init_world_model(), "predictor_confidence": {}, "prediction_error_history_e2": [], "prediction_error_history_ec": [], "dispositions": {"self": {}}}


def save_state(state: dict):
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
