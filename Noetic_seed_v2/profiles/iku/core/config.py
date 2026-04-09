"""設定・定数・パス・DualLogger"""
import json
import sys
import threading
from pathlib import Path

# === パス ===
BASE_DIR = Path(__file__).parent.parent  # profile directory
SEED_FILE = BASE_DIR / "seed.txt"
RAW_LOG_FILE = BASE_DIR / "raw_log.txt"
STATE_FILE = BASE_DIR / "state.json"
SANDBOX_DIR = BASE_DIR / "sandbox"
SANDBOX_TOOLS_DIR = BASE_DIR / "sandbox" / "tools"
LLM_SETTINGS = BASE_DIR / "settings.json"
DEBUG_LOG = BASE_DIR / "llm_debug.log"
MEMORY_DIR = BASE_DIR / "memory"

# === 定数 ===
ENV_INJECT_INTERVAL = 10        # 秒: pressureログ表示間隔
_NOTIFICATION_HOURS = {13, 17, 21, 1}
LOG_HARD_LIMIT = 50             # logがこの件数に達したら圧縮
LOG_KEEP = 30                   # 圧縮後に保持する生ログ件数
MEMORY_NETWORKS = ["world", "experience", "opinion", "entity"]
REFLECTION_INTERVAL_DEFAULT = 10  # サイクル

# === 圧力パラメータ ===
DEFAULT_PRESSURE_PARAMS = {
    "decay": 0.97,
    "threshold": 12.0,
    "post_fire_reset": 0.3,
}

# === DualLogger ===
class DualLogger:
    """標準出力とファイルへの同時書き出し"""
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
