"""設定・定数・パス・DualLogger"""
import json
import sys
import threading
from pathlib import Path

# === パス ===
BASE_DIR = Path(__file__).parent.parent  # profile directory (e.g. profiles/iku/)
SEED_FILE = BASE_DIR / "seed.txt"
RAW_LOG_FILE = BASE_DIR / "raw_log.txt"
STATE_FILE = BASE_DIR / "state.json"
SANDBOX_DIR = BASE_DIR / "sandbox"
SANDBOX_TOOLS_DIR = BASE_DIR / "sandbox" / "tools"
LLM_SETTINGS = BASE_DIR / "settings.json"
DEBUG_LOG = BASE_DIR / "llm_debug.log"
MEMORY_DIR = BASE_DIR / "memory"
PREF_FILE = BASE_DIR / "pref.json"

# === 定数 ===
BASE_INTERVAL = 20  # 秒（エラー回復用に残す）
MAX_LOG_IN_PROMPT = 10
ENV_INJECT_INTERVAL = 10  # 秒: ログ表示間隔
_NOTIFICATION_HOURS = {13, 17, 21, 1}

LOG_HARD_LIMIT = 150    # logがこの件数に達したらTrigger1
LOG_KEEP = 120          # Trigger1後に保持する生ログ件数（古い LOG_HARD_LIMIT - LOG_KEEP 件が要約される）
SUMMARY_HARD_LIMIT = 10 # summariesがこの件数に達したらTrigger2
META_SUMMARY_RAW = 15   # Trigger2でrawから使う件数

# === Prompt budget（settings.json の prompt_budget で上書き可）===
DEFAULT_PROMPT_BUDGET = {
    "context_window": 32768,
    "completion_reserve": 8192,
    "safety_margin": 512,
    "log_gradient": {
        "boundaries": [5, 15, 45],
        "caps": [20000, 3000, 800, 200],
        "intent_cap": 300,
    },
    "block_budgets": {
        "ltm_self": 800,
        "pending": 300,
        "related_memory": 800,
        "summaries": 1000,
        "tools": 1500,
        "instructions": 600,
    },
}


def _deep_merge(default: dict, override: dict) -> dict:
    """settings.json の prompt_budget を DEFAULT にマージ。dict は再帰、それ以外は上書き。"""
    out = dict(default)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def estimate_tokens(text: str) -> int:
    """雑な token 見積もり（日本語は 3 文字 ≒ 1 tok で上に寄せる）。"""
    if not text:
        return 0
    return len(text) // 3 + 1

# === 電脳気候パラメータのデフォルト（pref.jsonで上書き可）===
DEFAULT_PRESSURE_PARAMS = {
    "decay": 0.97,
    "clock_base": 0.15,
    "threshold": 12.0,
    "post_fire_reset": 0.3,
    "e2_pressure_scale": 3.0,
    "e3_pressure_scale": 0.6,
    "weights": {
        "info_velocity": 0.3,
        "info_entropy": 0.3,
        "channel_state": 0.3,
        "noise": 0.1,
    },
}

# === ネットワーク計測キャッシュ（電脳気候コード残存用）===
_net_cache: dict = {"avg": 50.0, "jitter": 0.0}
_net_lock = threading.Lock()

# === DualLogger ===
class DualLogger:
    """標準出力（ターミナル）へのprintとファイルへの追記を同時に行うクラス"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(message)
        except Exception:
            pass

    def flush(self):
        self.terminal.flush()

# === LLM設定読み込み ===
with open(LLM_SETTINGS, encoding="utf-8") as f:
    llm_cfg = json.load(f)

# === Prompt budget 読み込み（settings.json の prompt_budget を DEFAULT にマージ）===
prompt_budget = _deep_merge(DEFAULT_PROMPT_BUDGET, llm_cfg.get("prompt_budget", {}))
