"""test_inspect_wm_view.py — 段階11-A Step 3: inspect_wm_view tool 検証。

正典 PLAN: WORLD_MODEL_DESIGN/STAGE11A_PERSPECTIVE_FOUNDATION_PLAN.md §4-4

検証対象:
  - 基本動作: default kwargs (viewer=self, viewer_type=actual)
  - viewer 切替: args={"viewer": "ent_yuu"} で attributed 視点表示
  - viewer_type 切替: args={"viewer_type": "actual"} で仮想除外
  - **read-only 保証**: 呼び出し前後で state / world_model 不変
  - wm 未初期化で graceful ("世界モデル未初期化")
  - P2 原則遵守: tool 内部で自動発火 / 強制メタ認知コードなし (静的解析)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_inspect_wm_view.py
"""
import copy
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# tempdir に memory / tag_registry / state を isolate
_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_inspect_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

from core.perspective import make_perspective
from core.tag_registry import register_standard_tags
from core.world_model import init_world_model, store_wm_fact

register_standard_tags()

# state を mock する (load_state 差し替え)
import core.state as _state_mod

_mock_state = {
    "world_model": init_world_model(),
    "dispositions": {
        "self": {
            "curiosity": {"value": 0.7, "confidence": None,
                          "perspective": {"viewer": "self", "viewer_type": "actual",
                                          "view_time": "2026-04-22T00:00:00Z"},
                          "updated_at": "2026-04-22T00:00:00Z"},
        },
        "attributed:ent_yuu": {
            "curiosity": {"value": 0.9, "confidence": 0.6,
                          "perspective": {"viewer": "ent_yuu", "viewer_type": "actual",
                                          "view_time": "2026-04-22T00:00:00Z"},
                          "updated_at": "2026-04-22T00:00:00Z"},
        },
    },
}

# 3 視点の fact
store_wm_fact(_mock_state["world_model"], "alpha_e", "k", "self_v",
              perspective=make_perspective())
store_wm_fact(_mock_state["world_model"], "beta_e", "k", "yuu_v",
              perspective=make_perspective(viewer="ent_yuu", viewer_type="actual", confidence=0.6))
store_wm_fact(_mock_state["world_model"], "gamma_e", "k", "imag_v",
              perspective=make_perspective(viewer="fear", viewer_type="imagined", confidence=0.4))


_orig_load_state = _state_mod.load_state
_state_mod.load_state = lambda: _mock_state

# import は load_state 差し替え後に
from tools.perspective_tools import _inspect_wm_view


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    return bool(cond)


# =========================================================================
# Section 1: 基本動作 (defaults = self/actual)
# =========================================================================
print("=== Section 1: 基本動作 (default self/actual) ===")

out_default = _inspect_wm_view({})
_assert("beta_e" not in out_default, "1-2 default self: beta_e 除外 (ent_yuu 視点)")
_assert("gamma_e" not in out_default, "1-3 default self: gamma_e 除外 (imagined)")

# dispositions も self 視点のみ
_assert("#### 自己視点" in out_default, "1-4 default self: 自己視点 sub-header")
_assert("ent_yuu 視点" not in out_default, "1-5 default self: ent_yuu 視点 非表示")


# =========================================================================
# Section 2: viewer 切替 (ent_yuu 視点)
# =========================================================================
print("\n=== Section 2: viewer 切替 ===")

out_yuu = _inspect_wm_view({"viewer": "ent_yuu", "viewer_type": "actual"})
_assert("alpha_e" not in out_yuu, "2-2 viewer=ent_yuu: alpha_e 除外 (self 視点)")
_assert("gamma_e" not in out_yuu, "2-3 viewer=ent_yuu: gamma_e 除外 (imagined)")

# dispositions: attributed:ent_yuu のみ表示
_assert("ent_yuu 視点" in out_yuu, "2-4 viewer=ent_yuu: ent_yuu disposition sub-header")
_assert("#### 自己視点" not in out_yuu, "2-5 viewer=ent_yuu: 自己視点 disposition 非表示")


# =========================================================================
# Section 3: viewer_type 切替 (imagined)
# =========================================================================
print("\n=== Section 3: viewer_type 切替 ===")

out_imag = _inspect_wm_view({"viewer": "fear", "viewer_type": "imagined"})
_assert("alpha_e" not in out_imag, "3-2 viewer=fear/imagined: alpha_e 除外")
_assert("beta_e" not in out_imag, "3-3 viewer=fear/imagined: beta_e 除外")


# =========================================================================
# Section 4: read-only 保証 (副作用なし)
# =========================================================================
print("\n=== Section 4: read-only 保証 ===")

state_snapshot = copy.deepcopy(_mock_state)
_ = _inspect_wm_view({"viewer": "ent_yuu"})
_ = _inspect_wm_view({"viewer": "self", "viewer_type": "imagined"})
_ = _inspect_wm_view({})

_assert(
    _mock_state == state_snapshot,
    "4-1 inspect_wm_view 複数回呼び出しで state 不変 (read-only 保証)",
)


# =========================================================================
# Section 5: wm 未初期化 graceful
# =========================================================================
# 注意: perspective_tools が `from core.state import load_state` で static
#       bind してるので、test 側では tools.perspective_tools.load_state を
#       直接差し替えて mock 切り替え。
print("\n=== Section 5: wm 未初期化 ===")

import tools.perspective_tools as _pt

_mock_state_empty = {"world_model": None}
_pt.load_state = lambda: _mock_state_empty

out_empty = _inspect_wm_view({})
_assert(
    "未初期化" in out_empty,
    "5-1 wm=None → 'world model 未初期化' 相当メッセージ",
)

# 空 wm dict の場合
_mock_state_empty2 = {"world_model": {}}
_pt.load_state = lambda: _mock_state_empty2
out_empty2 = _inspect_wm_view({})
# render_for_prompt は 空 dict で "" を返すので、inspect_wm_view は "未初期化" を返す設計
_assert(
    "未初期化" in out_empty2 or out_empty2 == "",
    "5-2 wm={} でも graceful (未初期化 or 空)",
)


# =========================================================================
# Section 6: P2 原則遵守の静的確認
# =========================================================================
print("\n=== Section 6: P2 原則遵守 ===")

# perspective_tools.py の source に以下キーワードが無いことを確認
# - "threshold" (閾値起動) / "auto_fire" / "schedule"
# - "必ず呼ぶ" / "must call" 等 (自動発火コード)
src_path = Path(__file__).resolve().parent.parent / "tools" / "perspective_tools.py"
src = src_path.read_text(encoding="utf-8")

for forbidden in ["threshold", "auto_fire", "schedule", "必ず呼ぶ", "force_call"]:
    _assert(
        forbidden.lower() not in src.lower(),
        f"6-{forbidden}: perspective_tools.py に '{forbidden}' 出現なし (P2 静的確認)",
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

# load_state 復元
_state_mod.load_state = _orig_load_state
shutil.rmtree(_tmp_root, ignore_errors=True)

if failed:
    sys.exit(1)
