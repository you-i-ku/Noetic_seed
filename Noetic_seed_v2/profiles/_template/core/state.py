"""State管理（v2: pending, disposition, 4-network memory refs）"""
import json
from datetime import datetime
from core.config import STATE_FILE, SEED_FILE, DEBUG_LOG


def _get_name_from_seed() -> str:
    """seed.txtの1行目からnameを取得。"""
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


_DEFAULT_STATE = {
    "session_id": "",
    "cycle_id": 0,
    "tool_level": 0,
    "energy": 50,
    "entropy": 0.65,
    "pressure": 0.0,
    "log": [],
    "summaries": [],
    "self": {},
    "plan": {"goal": "", "steps": [], "current": 0},
    "files_read": [],
    "files_written": [],
    "tools_created": [],
    "pending": [],
    "responded_posts": [],
    "disposition": {
        "curiosity": 0.5,
        "skepticism": 0.5,
        "sociality": 0.5,
    },
    "last_e_values": {
        "achievement": 0.5,
        "prediction": 0.5,
        "diversity": 0.5,
        "coherence": 1.0,
        "negentropy": 0.0,
    },
    "last_prediction_error": 0.0,
    "last_coherence": 1.0,
    "reflection_cycle": 0,
    "drives_state": {},
    "unresponded_external_count": 0,
    "unresolved_external": 0.0,
}


def load_state() -> dict:
    _name = _get_name_from_seed()
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            # 欠けているキーをデフォルトで埋める
            for k, v in _DEFAULT_STATE.items():
                if k not in data:
                    data[k] = v if not isinstance(v, (dict, list)) else type(v)(v)
            if "self" not in data or not isinstance(data["self"], dict):
                data["self"] = {"name": _name}
            elif "name" not in data["self"]:
                data["self"]["name"] = _name
            return data
        except Exception:
            pass
    state = {k: (v if not isinstance(v, (dict, list)) else type(v)(v)) for k, v in _DEFAULT_STATE.items()}
    state["self"] = {"name": _name}
    return state


def save_state(state: dict):
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def append_debug_log(label: str, content: str):
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {label} =====\n{content}\n")
    except Exception:
        pass
