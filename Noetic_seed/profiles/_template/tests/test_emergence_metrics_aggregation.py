"""test_emergence_metrics_aggregation.py — 段階11-B Phase 5 Step 5.7。

検証対象 (log_cycle_metrics):
  - 期待 schema で jsonl に append (cycle / timestamp / stats 全 field /
    memory_count / link_count / link_grad_density / reconciliation_ec_count)
  - 複数 cycle 連続 append (行増加)
  - memory_store で書込 → memory_count 増加
  - reconciliation 由来 history → reconciliation_ec_count 反映
  - 空 state (None) で graceful 集計 (全 count 0)
  - log_file 明示指定で任意 path に書ける

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_emergence_metrics_aggregation.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_metrics_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

import core.memory_links as _ml
_ml.MEMORY_DIR = _tmp_memory

import core.tag_emergence_monitor as _tem
_tem.MEMORY_DIR = _tmp_memory

from core.memory import memory_store
from core.perspective import default_self_perspective
from core.tag_emergence_monitor import log_cycle_metrics
from core.tag_registry import register_standard_tags, register_tag


register_standard_tags()  # Phase 5 test でも標準 4 タグ登録で base 作る


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


# =========================================================================
# Section A: 期待 schema で jsonl append
# =========================================================================
print("=== Section A: 期待 schema ===")

log_file_a = _tmp_root / "logs_a" / "phase5_emergence.jsonl"
state_a = {}

m_a = log_cycle_metrics(1, state_a, log_file=log_file_a)

expected_fields = {
    "cycle", "timestamp", "total_registered", "standard_count", "dynamic_count",
    "write_protected_count", "memory_count", "link_count", "link_grad_density",
    "reconciliation_ec_count",
}
_assert(
    set(m_a.keys()) == expected_fields,
    f"A-1 全 field 揃い (got {set(m_a.keys())})",
)
_assert(m_a["cycle"] == 1, "A-2 cycle 番号 反映")
_assert(m_a["total_registered"] == 4, f"A-3 standard 4 tag (got {m_a['total_registered']})")
_assert(m_a["memory_count"] == 0, "A-4 memory_count=0 (まだ書込なし)")
_assert(m_a["link_count"] == 0, "A-5 link_count=0")
_assert(
    m_a["link_grad_density"] == 0.0,
    f"A-6 link_grad_density=0.0 (link/memory=0/0→0、max(1,0) で分母補正、got {m_a['link_grad_density']})",
)
_assert(m_a["reconciliation_ec_count"] == 0, "A-7 reconciliation_ec_count=0")

# jsonl 永続化確認
_assert(log_file_a.exists(), "A-8 jsonl ファイル生成")
lines_a = log_file_a.read_text(encoding="utf-8").splitlines()
_assert(len(lines_a) == 1, f"A-9 1 行記録 (got {len(lines_a)})")


# =========================================================================
# Section B: 複数 cycle 連続 append
# =========================================================================
print("=== Section B: 複数 cycle 累積 ===")

log_file_b = _tmp_root / "logs_b" / "phase5_emergence.jsonl"
state_b = {}

for i in range(1, 6):
    log_cycle_metrics(i, state_b, log_file=log_file_b)

lines_b = log_file_b.read_text(encoding="utf-8").splitlines()
_assert(len(lines_b) == 5, f"B-1 5 cycle 分 append (got {len(lines_b)})")

parsed = [json.loads(l) for l in lines_b]
_assert(
    [p["cycle"] for p in parsed] == [1, 2, 3, 4, 5],
    f"B-2 cycle 1-5 順序保持 (got {[p['cycle'] for p in parsed]})",
)


# =========================================================================
# Section C: memory_store 書込で memory_count 反映
# =========================================================================
print("=== Section C: memory_count 反映 ===")

log_file_c = _tmp_root / "logs_c" / "phase5_emergence.jsonl"
state_c = {}

m_c_before = log_cycle_metrics(1, state_c, log_file=log_file_c)

memory_store(
    "entity", "test content C1", {"entity_name": "ent_C1"},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)
memory_store(
    "experience", "test content C2", {},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)

m_c_after = log_cycle_metrics(2, state_c, log_file=log_file_c)

_assert(
    m_c_after["memory_count"] == m_c_before["memory_count"] + 2,
    f"C-1 memory_count +2 ({m_c_before['memory_count']} → {m_c_after['memory_count']})",
)


# =========================================================================
# Section D: reconciliation_ec_count 反映
# =========================================================================
print("=== Section D: reconciliation_ec_count ===")

log_file_d = _tmp_root / "logs_d" / "phase5_emergence.jsonl"
state_d = {
    "prediction_error_history_by_source": {
        "reconciliation": [
            {"magnitude": 0.5, "reason": "r1", "context": {}, "timestamp": ""},
            {"magnitude": 0.7, "reason": "r2", "context": {}, "timestamp": ""},
            {"magnitude": 0.3, "reason": "r3", "context": {}, "timestamp": ""},
        ],
    },
}

m_d = log_cycle_metrics(1, state_d, log_file=log_file_d)
_assert(
    m_d["reconciliation_ec_count"] == 3,
    f"D-1 reconciliation 3 件 reflect (got {m_d['reconciliation_ec_count']})",
)


# =========================================================================
# Section E: state=None で graceful (全 count 0 ほぼ)
# =========================================================================
print("=== Section E: state=None graceful ===")

log_file_e = _tmp_root / "logs_e" / "phase5_emergence.jsonl"
m_e = log_cycle_metrics(1, None, log_file=log_file_e)
_assert(m_e["cycle"] == 1, "E-1 cycle 番号")
_assert(m_e["reconciliation_ec_count"] == 0, "E-2 state None で reconciliation_ec_count=0")


# =========================================================================
# Section F: dynamic tag register で dynamic_count 反映
# =========================================================================
print("=== Section F: dynamic tag metric 反映 ===")

log_file_f = _tmp_root / "logs_f" / "phase5_emergence.jsonl"

register_tag(
    "new_dynamic",
    learning_rules={"beta_plus": False, "bitemporal": False},
    origin="dynamic",
)
m_f = log_cycle_metrics(1, {}, log_file=log_file_f)

_assert(
    m_f["dynamic_count"] >= 1,
    f"F-1 dynamic_count 反映 (got {m_f['dynamic_count']})",
)


# =========================================================================
print("=" * 60)
_pass = sum(1 for r, _ in results if r)
_total = len(results)
print(f"結果: {_pass}/{_total} passed")
for ok, msg in results:
    if not ok:
        print(f"  FAIL: {msg}")
print("=" * 60)

try:
    shutil.rmtree(_tmp_root, ignore_errors=True)
except Exception:
    pass

sys.exit(0 if _pass == _total else 1)
