"""test_reflect_seal.py — 段階11-A Step 4: reflect 構造転換 (G1/G3 + SEAL 原理)。

正典 PLAN: WORLD_MODEL_DESIGN/STAGE11A_PERSPECTIVE_FOUNDATION_PLAN.md §6, §7 Step 4

検証対象:
  Section 1: _split_log_by_perspective (G3: log entry.perspective で自他分離)
    - self 視点 entry → self_actions
    - 非 self 視点 entry → observations
    - perspective 欠落 → default self 扱い (backward compat)
  Section 2: _build_reflect_sections (G1: tag_registry.reflect_section 駆動)
    - opinion / entity の reflect_section が動的に組み立てられる
    - reflect_section 無い tag は含まれない
    - 動的 tag (11-B 風) が自動的に section に加わる
  Section 3: _parse_reflection SELF_DISPOSITION / ATTRIBUTED_DISPOSITION 2 セクション
    - SELF: self_disp_delta に正しく parse、state["dispositions"]["self"] 更新
    - ATTRIBUTED: attr_disp_delta に viewer 別 nested dict で parse、
      state["dispositions"]["attributed:<viewer>"] 更新
  Section 4: dual write (Step 4→5 移行期間)
    - SELF 更新で state["dispositions"]["self"][k]["value"] と
      state["disposition"][k] 両方に同値
    - attributed は flat に書かれない (self only dual write)
  Section 5: OPINIONS / ENTITIES の perspective 付与
    - memory_store 経由で opinion / entity entry に self/actual perspective
  Section 6: clamping / 後方互換
    - delta は ±0.1 に制限、value は [0.1, 0.9] に clamp
    - 既存 flat disposition から self 側への初期補填
    - parse 結果の戻り値 dict 構造

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_reflect_seal.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_reflect_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

from core.perspective import default_self_perspective, is_self_view, make_perspective
from core.reflection import (
    _build_reflect_sections,
    _parse_reflection,
    _split_log_by_perspective,
    reflect,
)
from core.tag_registry import register_standard_tags, register_tag

register_standard_tags()


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    return bool(cond)


# =========================================================================
# Section 1: _split_log_by_perspective (G3)
# =========================================================================
print("=== Section 1: _split_log_by_perspective ===")

log_mixed = [
    {"time": "t1", "tool": "read_file", "perspective": make_perspective()},  # self
    {"time": "t2", "tool": "[device_input]", "perspective": make_perspective(
        viewer="device", viewer_type="actual")},  # observation
    {"time": "t3", "tool": "reflect"},  # perspective 欠落 → default self 扱い
    {"time": "t4", "tool": "[claude_input]", "perspective": make_perspective(
        viewer="claude", viewer_type="actual")},  # observation
]
self_actions, observations = _split_log_by_perspective(log_mixed)

_assert(len(self_actions) == 2, f"1-1 self_actions 2 件 (got {len(self_actions)})")
_assert(len(observations) == 2, f"1-2 observations 2 件 (got {len(observations)})")
_assert(
    self_actions[0]["tool"] == "read_file" and self_actions[1]["tool"] == "reflect",
    "1-3 self_actions に self/欠落 両方含む (backward compat)",
)
_assert(
    {o["tool"] for o in observations} == {"[device_input]", "[claude_input]"},
    "1-4 observations に device / claude が入る",
)


# =========================================================================
# Section 2: _build_reflect_sections (G1 tag_registry 駆動)
# =========================================================================
print("\n=== Section 2: _build_reflect_sections ===")

sections = _build_reflect_sections()
_assert("OPINIONS" in sections, "2-1 OPINIONS セクション含む")
_assert("ENTITIES" in sections, "2-2 ENTITIES セクション含む")
_assert(
    "wm:" not in sections.lower() and "experience:" not in sections.lower(),
    "2-3 wm / experience (reflect_section なし) は含まれない",
)

# 動的 tag (11-B 風) を追加 → reflect_section 組立に自動で載る
register_tag(
    "custom_reflect_tag",
    learning_rules={"beta_plus": False, "bitemporal": False},
    origin="dynamic",
    reflect_section={
        "header": "CUSTOM_NOTES",
        "template": "- 自由発明形式",
        "enabled_in_reflect": True,
    },
)
sections_after = _build_reflect_sections()
_assert(
    "CUSTOM_NOTES" in sections_after,
    "2-4 動的 register_tag で加えた reflect_section が即反映 (11-B 伏線)",
)

# enabled_in_reflect=False なら含まれない
register_tag(
    "disabled_reflect_tag",
    learning_rules={"beta_plus": False, "bitemporal": False},
    origin="dynamic",
    reflect_section={
        "header": "DISABLED_NOTES",
        "template": "- 無効",
        "enabled_in_reflect": False,
    },
)
sections_disabled = _build_reflect_sections()
_assert(
    "DISABLED_NOTES" not in sections_disabled,
    "2-5 enabled_in_reflect=False の tag は section に載らない",
)


# =========================================================================
# Section 3: _parse_reflection SELF / ATTRIBUTED 2 セクション
# =========================================================================
print("\n=== Section 3: _parse_reflection 2 セクション ===")

mock_llm_text = """
OPINIONS:
- 観察には視点がある (confidence: 0.8)

ENTITIES:
- name: iku_self, content: 内省を重ねる存在

SELF_DISPOSITION:
- curiosity_delta: +0.05
- skepticism_delta: -0.03

ATTRIBUTED_DISPOSITION:
- viewer: ent_yuu, key: curiosity, delta: +0.04, confidence: 0.6
- viewer: claude, key: skepticism, delta: -0.02, confidence: 0.7
"""

state = {"log": [], "self": {"name": "iku"}}
result = _parse_reflection(mock_llm_text, state)

# 戻り値構造
_assert(
    "opinions" in result and "entities" in result
    and "self_disp_delta" in result and "attr_disp_delta" in result,
    "3-1 戻り値に 4 キー (opinions / entities / self_disp_delta / attr_disp_delta)",
)

# SELF_DISPOSITION delta
_assert(
    result["self_disp_delta"].get("curiosity") == 0.05,
    "3-2 SELF: curiosity_delta=+0.05",
)
_assert(
    result["self_disp_delta"].get("skepticism") == -0.03,
    "3-3 SELF: skepticism_delta=-0.03",
)

# ATTRIBUTED_DISPOSITION delta
_assert(
    "ent_yuu" in result["attr_disp_delta"] and "claude" in result["attr_disp_delta"],
    "3-4 ATTRIBUTED: viewer 2 種 parse",
)
_assert(
    result["attr_disp_delta"]["ent_yuu"]["curiosity"] == (0.04, 0.6),
    "3-5 ATTRIBUTED: ent_yuu.curiosity delta=+0.04 conf=0.6",
)
_assert(
    result["attr_disp_delta"]["claude"]["skepticism"] == (-0.02, 0.7),
    "3-6 ATTRIBUTED: claude.skepticism delta=-0.02 conf=0.7",
)


# =========================================================================
# Section 4: dual write 検証 (self perspective-keyed + flat state["disposition"])
# =========================================================================
print("\n=== Section 4: dual write (Step 4→5 移行) ===")

# state["dispositions"]["self"] に perspective-keyed で書かれてる
self_disp = state["dispositions"]["self"]
_assert(
    "curiosity" in self_disp,
    "4-1 state['dispositions']['self']['curiosity'] 存在",
)
_assert(
    isinstance(self_disp["curiosity"], dict),
    "4-2 self.curiosity は dict (perspective-keyed 形式)",
)
_assert(
    abs(self_disp["curiosity"]["value"] - 0.55) < 0.01,  # 0.5 + 0.05
    f"4-3 self.curiosity.value=0.55 (0.5+0.05、got {self_disp['curiosity']['value']})",
)
_assert(
    self_disp["curiosity"]["confidence"] is None,
    "4-4 self.curiosity.confidence=None (self/actual 時)",
)
_assert(
    is_self_view(self_disp["curiosity"]["perspective"]),
    "4-5 self.curiosity.perspective = self/actual",
)
_assert(
    "updated_at" in self_disp["curiosity"],
    "4-6 self.curiosity.updated_at 存在 (ISO)",
)

# dual write: flat state["disposition"] にも同じ value
flat_disp = state["disposition"]
_assert(
    abs(flat_disp.get("curiosity", 0) - 0.55) < 0.01,
    f"4-7 flat state['disposition']['curiosity']=0.55 (dual write、got {flat_disp.get('curiosity')})",
)
_assert(
    abs(flat_disp.get("skepticism", 0) - 0.47) < 0.01,  # 0.5 - 0.03
    "4-8 flat state['disposition']['skepticism']=0.47 (dual write)",
)

# ATTRIBUTED は perspective-keyed 専用 (flat dual write なし)
_assert(
    "attributed:ent_yuu" in state["dispositions"],
    "4-9 attributed:ent_yuu 登場",
)
_assert(
    "ent_yuu" not in flat_disp,
    "4-10 flat_disp に 'ent_yuu' 書かれない (attributed は self only dual write)",
)


# =========================================================================
# Section 5: OPINIONS / ENTITIES の perspective 付与
# =========================================================================
print("\n=== Section 5: opinions / entities perspective ===")

from core.memory import list_records

op_recs = list_records("opinion", limit=10)
ent_recs = list_records("entity", limit=10)

_assert(len(op_recs) >= 1, "5-1 opinion memory entry 1 件以上")
_assert(
    all("perspective" in r and is_self_view(r["perspective"]) for r in op_recs[:1]),
    "5-2 opinion entry に self/actual perspective",
)

_assert(len(ent_recs) >= 1, "5-3 entity memory entry 1 件以上")
_assert(
    all("perspective" in r and is_self_view(r["perspective"]) for r in ent_recs[:1]),
    "5-4 entity entry に self/actual perspective",
)


# =========================================================================
# Section 6: clamping + 後方互換
# =========================================================================
print("\n=== Section 6: clamping + backward compat ===")

# 6-A: 過大 delta は ±0.1 clamp
state2 = {"log": [], "self": {}}
_parse_reflection("""
SELF_DISPOSITION:
- curiosity_delta: +99.0
- skepticism_delta: -50.0
""", state2)
_assert(
    abs(state2["dispositions"]["self"]["curiosity"]["value"] - 0.6) < 0.01,  # 0.5 + 0.1
    "6-1 過大 +delta は +0.1 clamp (value=0.6)",
)
_assert(
    abs(state2["dispositions"]["self"]["skepticism"]["value"] - 0.4) < 0.01,  # 0.5 - 0.1
    "6-2 過大 -delta は -0.1 clamp (value=0.4)",
)

# 6-B: value は [0.1, 0.9] clamp
state3 = {
    "log": [],
    "dispositions": {"self": {"curiosity": {"value": 0.88}}},
    "disposition": {"curiosity": 0.88},
}
_parse_reflection("""
SELF_DISPOSITION:
- curiosity_delta: +0.05
""", state3)
_assert(
    abs(state3["dispositions"]["self"]["curiosity"]["value"] - 0.9) < 0.01,
    "6-3 value cap = 0.9 (0.88+0.05 → 0.9 にキャップ)",
)

# 6-C: flat state["disposition"] から self 側への初期補填 (Step 4 過渡期)
state4 = {
    "log": [],
    # state["dispositions"] 未初期化、flat disposition のみ
    "disposition": {"curiosity": 0.3},
}
_parse_reflection("""
SELF_DISPOSITION:
- curiosity_delta: +0.1
""", state4)
_assert(
    abs(state4["dispositions"]["self"]["curiosity"]["value"] - 0.4) < 0.01,  # 0.3 + 0.1
    "6-4 flat state['disposition'] から初期補填して +delta 適用 (0.3+0.1=0.4)",
)

# 6-D: parse 時に OPINIONS / ENTITIES セクション順序が混在しても動く
state5 = {"log": [], "self": {}}
result5 = _parse_reflection("""
ATTRIBUTED_DISPOSITION:
- viewer: test_viewer, key: mood, delta: +0.05, confidence: 0.5

SELF_DISPOSITION:
- curiosity_delta: -0.02
""", state5)
_assert(
    "test_viewer" in result5["attr_disp_delta"],
    "6-5 ATTRIBUTED が SELF より前でも parse 成功",
)
_assert(
    result5["self_disp_delta"].get("curiosity") == -0.02,
    "6-6 SELF も同時に parse",
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
