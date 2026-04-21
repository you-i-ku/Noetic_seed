"""test_disposition_migration.py — 段階11-A Step 5: disposition 移行検証。

正典 PLAN: WORLD_MODEL_DESIGN/STAGE11A_PERSPECTIVE_FOUNDATION_PLAN.md §5-1, §7 Step 5

検証対象:
  Section 1: _migrate_disposition_v11a 単体
    - flat only → dispositions.self 移行 + state.disposition 完全撤去
    - 既存 dispositions.self あり → flat から未反映 trait のみ追加 (既存尊重)
    - 空 state → dispositions.self={} 初期化
    - 空 flat ({}) → state.disposition 削除のみ
    - 冪等 (2 回呼んでも結果不変)
    - value clamp (0.1-0.9)
  Section 2: load_state 経由の自動 migration
    - state.json に旧 flat disposition 書いて load_state → 移行済 state
  Section 3: 新規起動時の default dict に dispositions 含まれる

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_disposition_migration.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# tempdir に STATE_FILE を向ける
_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_migrate_"))
_tmp_state_file = _tmp_root / "state.json"
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.STATE_FILE = _tmp_state_file
_cfg.MEMORY_DIR = _tmp_memory
_cfg.SEED_FILE = _tmp_root / "seed.txt"

import core.state as _state_mod
_state_mod.STATE_FILE = _tmp_state_file
_state_mod.SEED_FILE = _tmp_root / "seed.txt"

from core.perspective import is_self_view
from core.state import _migrate_disposition_v11a, load_state


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    return bool(cond)


# =========================================================================
# Section 1: _migrate_disposition_v11a 単体
# =========================================================================
print("=== Section 1: _migrate_disposition_v11a ===")

# 1-A: flat only state → dispositions.self に移行 + state.disposition 撤去
state1 = {
    "disposition": {"curiosity": 0.7, "skepticism": 0.4, "sociality": 0.6},
}
_migrate_disposition_v11a(state1)
_assert(
    "disposition" not in state1,
    "1-1 flat state['disposition'] 撤去済 (pop)",
)
_assert(
    "dispositions" in state1 and "self" in state1["dispositions"],
    "1-2 state['dispositions']['self'] 作成",
)
self_disp = state1["dispositions"]["self"]
_assert(
    "curiosity" in self_disp and self_disp["curiosity"]["value"] == 0.7,
    "1-3 curiosity=0.7 移行",
)
_assert(
    self_disp["curiosity"]["confidence"] is None,
    "1-4 self entry confidence=None",
)
_assert(
    is_self_view(self_disp["curiosity"]["perspective"]),
    "1-5 self entry perspective=self/actual",
)
_assert(
    "updated_at" in self_disp["curiosity"],
    "1-6 self entry updated_at 存在",
)

# 1-B: 既存 dispositions.self あり、flat も追加 → 既存優先、flat の新 key のみ追加
state2 = {
    "disposition": {"curiosity": 0.99, "new_trait": 0.55},  # flat 側
    "dispositions": {
        "self": {
            "curiosity": {"value": 0.5, "confidence": None, "perspective": {},
                          "updated_at": "prev"},  # 既存
        },
    },
}
_migrate_disposition_v11a(state2)
_assert(
    state2["dispositions"]["self"]["curiosity"]["value"] == 0.5,
    "1-7 既存 curiosity=0.5 保持 (flat 0.99 で上書きしない)",
)
_assert(
    state2["dispositions"]["self"]["new_trait"]["value"] == 0.55,
    "1-8 flat の新 key は self に追加",
)
_assert("disposition" not in state2, "1-9 flat key 撤去")

# 1-C: 空 state → dispositions.self={} 初期化
state3 = {}
_migrate_disposition_v11a(state3)
_assert(
    state3.get("dispositions", {}).get("self") == {},
    "1-10 空 state → dispositions.self={} 初期化",
)
_assert("disposition" not in state3, "1-11 空 state → disposition key 不在")

# 1-D: 空 flat ({}) → state.disposition 削除
state4 = {"disposition": {}, "dispositions": {"self": {"x": {"value": 0.3}}}}
_migrate_disposition_v11a(state4)
_assert("disposition" not in state4, "1-12 空 flat も pop で撤去")
_assert(
    state4["dispositions"]["self"]["x"]["value"] == 0.3,
    "1-13 既存 self 保持",
)

# 1-E: 冪等 (2 回呼んでも結果不変)
state5 = {"disposition": {"curiosity": 0.6}}
_migrate_disposition_v11a(state5)
snap = json.dumps(state5, default=str, sort_keys=True)
_migrate_disposition_v11a(state5)
snap2 = json.dumps(state5, default=str, sort_keys=True)
_assert(snap == snap2, "1-14 2 回呼出で結果不変 (冪等)")

# 1-F: value clamp (flat 側の異常値を 0.1-0.9 に収める)
state6 = {"disposition": {"too_high": 99.0, "too_low": -50.0, "normal": 0.5}}
_migrate_disposition_v11a(state6)
_assert(
    state6["dispositions"]["self"]["too_high"]["value"] == 0.9,
    "1-15 異常 high → 0.9 clamp",
)
_assert(
    state6["dispositions"]["self"]["too_low"]["value"] == 0.1,
    "1-16 異常 low → 0.1 clamp",
)
_assert(
    state6["dispositions"]["self"]["normal"]["value"] == 0.5,
    "1-17 normal value 保持",
)

# 1-G: 非数値の flat value は 0.5 fallback
state7 = {"disposition": {"weird": "not_a_number"}}
_migrate_disposition_v11a(state7)
_assert(
    state7["dispositions"]["self"]["weird"]["value"] == 0.5,
    "1-18 非数値 flat → 0.5 fallback",
)


# =========================================================================
# Section 2: load_state 経由の自動 migration
# =========================================================================
print("\n=== Section 2: load_state 経由 ===")

# state.json に旧 flat disposition 書き込み
legacy_state = {
    "log": [],
    "self": {"name": "test"},
    "energy": 50,
    "cycle_id": 0,
    "disposition": {"curiosity": 0.6, "skepticism": 0.4},
    # Step 5 以前の state を想定 (dispositions キー無い)
}
_tmp_state_file.write_text(
    json.dumps(legacy_state, ensure_ascii=False), encoding="utf-8",
)

loaded = load_state()
_assert(
    "disposition" not in loaded,
    "2-1 load_state 後 flat disposition 撤去済",
)
_assert(
    "dispositions" in loaded and "self" in loaded["dispositions"],
    "2-2 load_state 後 dispositions.self 存在",
)
_assert(
    abs(loaded["dispositions"]["self"]["curiosity"]["value"] - 0.6) < 0.01,
    "2-3 load_state 経由で curiosity=0.6 移行済",
)
_assert(
    abs(loaded["dispositions"]["self"]["skepticism"]["value"] - 0.4) < 0.01,
    "2-4 load_state 経由で skepticism=0.4 移行済",
)


# =========================================================================
# Section 3: 新規起動時の default dict に dispositions
# =========================================================================
print("\n=== Section 3: 新規起動 default ===")

# state.json 削除して新規扱い
_tmp_state_file.unlink()
fresh = load_state()
_assert(
    "dispositions" in fresh,
    "3-1 新規起動 default dict に dispositions キー",
)
_assert(
    fresh["dispositions"] == {"self": {}},
    "3-2 新規 dispositions={'self': {}} 初期化",
)
_assert(
    "disposition" not in fresh,
    "3-3 新規 default に flat disposition なし",
)


# =========================================================================
# Summary + Cleanup
# =========================================================================
print("\n=== Summary ===")
passed = sum(1 for r, _ in results if r)
failed = sum(1 for r, _ in results if not r)
for r, m in results:
    if not r:
        print(f"  FAIL: {m}")
print(f"\nPASSED: {passed} / {passed + failed}")

shutil.rmtree(_tmp_root, ignore_errors=True)

if failed:
    sys.exit(1)
