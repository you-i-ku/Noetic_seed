"""test_wm_perspective.py — 段階11-A Step 3: WM perspective 伝播 + view_filter +
dispositions dual support。

正典 PLAN: WORLD_MODEL_DESIGN/STAGE11A_PERSPECTIVE_FOUNDATION_PLAN.md §4-3, §4-4, §7 Step 3

検証対象:
  Section 1: store_wm_fact perspective 伝播
    - kwarg 指定 → fact / memory_store 両方に perspective 入る
    - kwarg None → default_self_perspective 補完
  Section 2: rebuild_wm_from_jsonl
    - 新 entry (perspective 付) → fact に perspective 保持
    - legacy entry (perspective 欠落) → default 補完
    - 混在でも壊れない (backward compat)
  Section 3: render_for_prompt view_filter
    - view_filter=None: 全視点表示 (既存互換)
    - view_filter={"viewer": "self"}: self 視点 fact のみ
    - view_filter={"viewer_type": "actual"}: 仮想視点除外
  Section 4: dispositions dual support
    - flat dict (段階10.5 Fix 4 δ' 形式) がそのまま表示
    - perspective-keyed dict (Step 5 以降) が sub-header 付きで表示
    - view_filter 適用で disposition も絞られる
  Section 5: 後方互換 (view_filter / perspective 未指定で既存挙動維持)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_wm_perspective.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# MEMORY_DIR を tempdir に差し替え (tag_registry と memory を isolate)
_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_wm_persp_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

from core.perspective import default_self_perspective, make_perspective
from core.tag_registry import register_standard_tags
from core.world_model import (
    init_world_model,
    rebuild_wm_from_jsonl,
    render_for_prompt,
    store_wm_fact,
    _pkey_matches_filter,
    _pkey_str_to_perspective,
    _is_perspective_keyed_dispositions,
)


register_standard_tags()


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    return bool(cond)


# =========================================================================
# Section 1: store_wm_fact perspective 伝播
# =========================================================================
print("=== Section 1: store_wm_fact perspective 伝播 ===")

wm = init_world_model()

# 1-A: perspective kwarg 指定 (他者視点)
p_yuu = make_perspective(viewer="device", viewer_type="actual", confidence=0.6)
fact_other = store_wm_fact(wm, "ゆう", "role", "確認相手",
                           confidence=0.8, perspective=p_yuu)
_assert(
    "perspective" in fact_other,
    "1-1 fact に perspective 属性",
)
_assert(
    fact_other["perspective"]["viewer"] == "device",
    "1-2 fact.perspective.viewer=device",
)

# 1-B: perspective None → default (self/actual) 補完
fact_default = store_wm_fact(wm, "iku_self", "state", "観察中",
                             confidence=0.9)
_assert(
    fact_default["perspective"]["viewer"] == "self",
    "1-3 kwarg 未指定 → default self 補完",
)
_assert(
    fact_default["perspective"]["viewer_type"] == "actual",
    "1-4 default viewer_type=actual",
)

# 1-C: memory_store 側にも伝播 (wm.jsonl の entry に perspective 入る)
from core.memory import list_records
wm_recs = list_records("wm", limit=10)
_assert(
    any("perspective" in r for r in wm_recs),
    "1-5 memory_store 経由の wm.jsonl entry に perspective",
)
# 最新 entry (iku_self, state) が self 視点
_assert(
    wm_recs[0]["perspective"]["viewer"] == "self",
    "1-6 最新 wm entry perspective=self (iku_self entry)",
)
# 2 件目 (ゆう, role) が device 視点
_assert(
    wm_recs[1]["perspective"]["viewer"] == "device",
    "1-7 2 件目 wm entry perspective=device",
)


# =========================================================================
# Section 2: rebuild_wm_from_jsonl
# =========================================================================
print("\n=== Section 2: rebuild_wm_from_jsonl ===")

# 新 entry (perspective 付) と legacy entry (perspective 欠落) を混ぜる
legacy_rec = {
    "id": "mem_legacy_001",
    "network": "wm",
    "content": "test.key1 = legacy",
    "metadata": {
        "entity_name": "legacy_entity",
        "fact_key": "key1",
        "fact_value": "legacy_val",
        "confidence": 0.7,
    },
    # perspective 欠落 (段階11-A 以前)
}
new_rec = {
    "id": "mem_new_001",
    "network": "wm",
    "content": "test.key2 = new",
    "metadata": {
        "entity_name": "new_entity",
        "fact_key": "key2",
        "fact_value": "new_val",
        "confidence": 0.8,
    },
    "perspective": make_perspective(viewer="claude", viewer_type="actual", confidence=0.5),
}

wm2 = init_world_model()
count = rebuild_wm_from_jsonl(wm2, [legacy_rec, new_rec])
_assert(count == 2, f"2-1 2 records 処理 (got {count})")

# legacy_entity の fact は default self 補完
legacy_ent = wm2["entities"].get("ent_legacy_entity")
_assert(legacy_ent is not None, "2-2 legacy_entity が entity 化された")
legacy_fact = legacy_ent["facts"][0]
_assert(
    legacy_fact["perspective"]["viewer"] == "self",
    "2-3 legacy rec → fact.perspective.viewer=self (default 補完)",
)

# new_entity の fact は claude 視点保持
new_ent = wm2["entities"].get("ent_new_entity")
_assert(new_ent is not None, "2-4 new_entity が entity 化された")
new_fact = new_ent["facts"][0]
_assert(
    new_fact["perspective"]["viewer"] == "claude",
    "2-5 new rec → fact.perspective.viewer=claude",
)


# =========================================================================
# Section 3: render_for_prompt view_filter
# =========================================================================
print("\n=== Section 3: render_for_prompt view_filter ===")

# 3 種 fact を持つ wm を用意
wm3 = init_world_model()
store_wm_fact(wm3, "alpha", "key_a", "self_obs",
              perspective=make_perspective())  # self/actual
store_wm_fact(wm3, "beta", "key_b", "yuu_obs",
              perspective=make_perspective(viewer="device", viewer_type="actual"))
store_wm_fact(wm3, "gamma", "key_g", "imagined_obs",
              perspective=make_perspective(viewer="future_fear", viewer_type="imagined"))

# 3-A: view_filter=None → 全視点表示
rendered_all = render_for_prompt(wm3, view_filter=None)
_assert("alpha" in rendered_all, "3-1 view_filter=None: alpha 表示")
_assert("beta" in rendered_all, "3-2 view_filter=None: beta 表示")
_assert("gamma" in rendered_all, "3-3 view_filter=None: gamma 表示")

# 3-B: view_filter={"viewer": "self"} → self 視点のみ (alpha のみ)
rendered_self = render_for_prompt(wm3, view_filter={"viewer": "self"})
_assert("alpha" in rendered_self, "3-4 view_filter=self: alpha 表示")
_assert("beta" not in rendered_self, "3-5 view_filter=self: beta 除外 (device 視点)")
_assert("gamma" not in rendered_self, "3-6 view_filter=self: gamma 除外 (imagined)")

# 3-C: view_filter={"viewer_type": "actual"} → 仮想除外 (alpha + beta)
rendered_actual = render_for_prompt(wm3, view_filter={"viewer_type": "actual"})
_assert("alpha" in rendered_actual, "3-7 view_filter=actual: alpha 表示")
_assert("beta" in rendered_actual, "3-8 view_filter=actual: beta 表示")
_assert("gamma" not in rendered_actual, "3-9 view_filter=actual: gamma 除外 (imagined)")


# =========================================================================
# Section 4: dispositions dual support
# =========================================================================
print("\n=== Section 4: dispositions dual support ===")

wm4 = init_world_model()
store_wm_fact(wm4, "dummy", "k", "v")

# 4-A: flat dict (段階10.5 Fix 4 δ' 形式)
flat_disp = {"curiosity": 0.8, "skepticism": 0.3}
rendered_flat = render_for_prompt(wm4, dispositions=flat_disp)
_assert(
    "curiosity" in rendered_flat and "0.80" in rendered_flat,
    "4-1 flat dispositions: curiosity 表示",
)
_assert(
    "skepticism" in rendered_flat and "0.30" in rendered_flat,
    "4-2 flat dispositions: skepticism 表示",
)
# flat は sub-header なし
_assert(
    "#### 自己視点" not in rendered_flat,
    "4-3 flat 形式では perspective sub-header なし",
)

# 4-B: perspective-keyed dict (Step 5 以降)
pkeyed_disp = {
    "self": {
        "curiosity": {"value": 0.7, "confidence": None,
                      "perspective": default_self_perspective(),
                      "updated_at": "2026-04-22T00:00:00Z"},
        "skepticism": {"value": 0.5, "confidence": None,
                       "perspective": default_self_perspective(),
                       "updated_at": "2026-04-22T00:00:00Z"},
    },
    "attributed:ent_yuu": {
        "curiosity": {"value": 0.9, "confidence": 0.6,
                      "perspective": make_perspective(viewer="ent_yuu", viewer_type="actual"),
                      "updated_at": "2026-04-22T00:00:00Z"},
    },
}
# view_filter なし → 全視点 dispositions
rendered_pkeyed_all = render_for_prompt(wm4, dispositions=pkeyed_disp)
_assert(
    "#### 自己視点" in rendered_pkeyed_all,
    "4-4 perspective-keyed: '自己視点' sub-header",
)
_assert(
    "ent_yuu 視点" in rendered_pkeyed_all,
    "4-5 perspective-keyed: ent_yuu sub-header",
)
_assert("0.70" in rendered_pkeyed_all, "4-6 pkeyed: self.curiosity=0.70 表示")
_assert("0.90" in rendered_pkeyed_all, "4-7 pkeyed: ent_yuu.curiosity=0.90 表示")

# view_filter={"viewer": "self"} → self dispositions のみ
rendered_pkeyed_self = render_for_prompt(
    wm4, dispositions=pkeyed_disp, view_filter={"viewer": "self"},
)
_assert(
    "#### 自己視点" in rendered_pkeyed_self,
    "4-8 pkeyed+self filter: 自己視点 表示",
)
_assert(
    "ent_yuu 視点" not in rendered_pkeyed_self,
    "4-9 pkeyed+self filter: ent_yuu 除外",
)


# =========================================================================
# Section 5: 後方互換 (view_filter 未指定での既存挙動)
# =========================================================================
print("\n=== Section 5: 後方互換 ===")

# view_filter / perspective 何も指定しない既存呼び出しで同等結果
wm5 = init_world_model()
# perspective を積極的に指定しない (自動 default)
store_wm_fact(wm5, "testent", "role", "テスト")
rendered_legacy = render_for_prompt(wm5, max_entities=10)
_assert(
    "testent" in rendered_legacy,
    "5-1 view_filter 未指定で既存通り entity 表示",
)
_assert(
    "role=テスト" in rendered_legacy,
    "5-2 既存通り fact 表示",
)

# wm=None / 空の場合
_assert(render_for_prompt(None) == "", "5-3 wm=None → 空文字")


# =========================================================================
# Section 6: helpers (_pkey_matches_filter / _pkey_str_to_perspective / _is_perspective_keyed_dispositions)
# =========================================================================
print("\n=== Section 6: helpers ===")

# _pkey_matches_filter
_assert(
    _pkey_matches_filter(default_self_perspective(), None) is True,
    "6-1 view_filter=None → True",
)
_assert(
    _pkey_matches_filter(default_self_perspective(), {"viewer": "self"}) is True,
    "6-2 self perspective + viewer=self filter → True",
)
_assert(
    _pkey_matches_filter(
        make_perspective(viewer="ent_yuu", viewer_type="actual"),
        {"viewer": "self"},
    ) is False,
    "6-3 ent_yuu perspective + viewer=self filter → False",
)
_assert(
    _pkey_matches_filter(None, {"viewer": "self"}) is True,
    "6-4 perspective=None → default self 補完で True",
)

# _pkey_str_to_perspective
p1 = _pkey_str_to_perspective("self")
_assert(p1["viewer"] == "self" and p1["viewer_type"] == "actual", "6-5 'self' 逆変換")
p2 = _pkey_str_to_perspective("attributed:ent_yuu")
_assert(
    p2["viewer"] == "ent_yuu" and p2["viewer_type"] == "actual",
    "6-6 'attributed:X' 逆変換",
)
p3 = _pkey_str_to_perspective("imagined:fear_future")
_assert(
    p3["viewer"] == "fear_future" and p3["viewer_type"] == "imagined",
    "6-7 'imagined:X' 逆変換",
)

# _is_perspective_keyed_dispositions
_assert(
    _is_perspective_keyed_dispositions({"curiosity": 0.8}) is False,
    "6-8 flat dict → is_pkeyed False",
)
_assert(
    _is_perspective_keyed_dispositions({"self": {"curiosity": {"value": 0.8}}})
    is True,
    "6-9 perspective-keyed → is_pkeyed True",
)
_assert(
    _is_perspective_keyed_dispositions({}) is False,
    "6-10 空 dict → is_pkeyed False",
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
