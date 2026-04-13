"""State管理・好み関数・デバッグログ"""
import json
import os
import tempfile
from datetime import datetime
from core.config import STATE_FILE, PREF_FILE, DEBUG_LOG, SEED_FILE


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
            return data
        except json.JSONDecodeError:
            pass
    return {"log": [], "self": {"name": _name}, "energy": 50, "summaries": [], "cycle_id": 0, "tool_level": 0, "files_read": [], "files_written": [], "last_notification_fetch": "", "pressure": 0.0, "last_e1": 0.5, "last_e2": 0.5, "last_e3": 0.5, "last_e4": 0.5, "tools_created": [], "entropy": 0.65, "drives_state": {}}


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
