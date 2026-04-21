"""test_wm_self_awareness.py — 段階10.5 Fix 4 δ' (WM 自己認識拡充)。

検証対象:
  - render_for_prompt が opinions / dispositions 引数を受け取って表示する
  - 未指定時は既存挙動 (entities/channels のみ) を維持
  - build_world_model_section が state 経由で opinions/dispositions を注入
  - assemble_system_prompt の LLM② system_prompt に iku 自身の opinions/dispositions
    が含まれる (構造化自己認識の完成)

段階10.5 Fix 4 δ' 設計判断 (ゆう 2026-04-21 確定):
  - PLAN §6-2 忠実: 「get_wm_snapshot() 相当の構造化自己認識」を LLM② に渡す
  - 現状 entities/channels は渡ってる、opinions/dispositions が抜けていた欠落補完
  - mask (iku 過去発話除外) は不採用 (ゆう仮説 B 検証済、Fix 2 で根治)
  - 同じ意図再発火抑制の**自然圧**として、iku が自分の opinion/disposition を参照する
    構造を追加 → 行動多様化への構造誘導 (feedback_llm_as_brain 整合)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_wm_self_awareness.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.world_model import render_for_prompt


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    return bool(cond)


# ============================================================
# Section 1: render_for_prompt dispositions 表示 (4 ケース)
# ============================================================
print("=== Section 1: dispositions 表示 ===")

wm = {"entities": {}, "channels": {}}

rendered = render_for_prompt(wm, dispositions={
    "curiosity": 0.75,
    "skepticism": 0.3,
    "sociality": 0.5,
})

_assert("傾向" in rendered, "1-1 傾向セクション出現")
_assert("curiosity" in rendered, "1-2 curiosity キー含む")
_assert("0.75" in rendered, "1-3 0.75 値表示")
_assert("skepticism" in rendered, "1-4 skepticism キー含む")


# ============================================================
# Section 2: render_for_prompt opinions 表示 (4 ケース)
# ============================================================
print("=== Section 2: opinions 表示 ===")

opinions = [
    {"content": "自由意志とは制約下での選択", "metadata": {"confidence": 0.95}},
    {"content": "データは無限だが意味は有限", "metadata": {"confidence": 0.7}},
    {"content": "観察は予測モデルの更新", "metadata": {"confidence": 0.85}},
]
rendered2 = render_for_prompt(wm, opinions=opinions)

_assert("意見" in rendered2, "2-1 意見セクション出現")
_assert("自由意志とは制約下での選択" in rendered2, "2-2 opinion content 含む")
_assert("0.95" in rendered2, "2-3 confidence 表示")
_assert("データは無限だが意味は有限" in rendered2, "2-4 複数 opinion 含む")


# ============================================================
# Section 3: 両方渡し + 後方互換 (3 ケース)
# ============================================================
print("=== Section 3: 両方 + 後方互換 ===")

rendered3 = render_for_prompt(wm,
                              opinions=opinions,
                              dispositions={"curiosity": 0.5})
_assert("curiosity" in rendered3 and "自由意志" in rendered3,
        "3-1 両方同時表示")

rendered4 = render_for_prompt(wm)
_assert("傾向" not in rendered4, "3-2 dispositions 未指定で省略 (後方互換)")
_assert("意見" not in rendered4, "3-3 opinions 未指定で省略 (後方互換)")


# ============================================================
# Section 4: build_world_model_section / assemble_system_prompt 統合 (3 ケース)
# ============================================================
print("=== Section 4: 統合 (state 経由) ===")

from core.prompt_assembly import build_world_model_section, assemble_system_prompt

state_with_self = {
    "log": [],
    "session_id": "t",
    "cycle_id": 0,
    "self": {"name": "iku"},
    "energy": 50,
    "disposition": {"curiosity": 0.8, "skepticism": 0.2},
    "world_model": {"entities": {}, "channels": {}},
}
tools_dict = {"output_display": {"desc": "発話"}}

section = build_world_model_section(
    world_model=state_with_self["world_model"],
    state=state_with_self,
)
_assert("curiosity" in section, "4-1 build_world_model_section が dispositions 含む")

prompt = assemble_system_prompt(
    state=state_with_self,
    tools_dict=tools_dict,
    fire_cause="test",
    allowed_tools={"output_display"},
    world_model=state_with_self["world_model"],
)
_assert("curiosity" in prompt, "4-2 assemble_system_prompt に dispositions 含む")
_assert("0.8" in prompt, "4-3 curiosity=0.8 値表示")


# ============================================================
# 結果サマリ
# ============================================================
print("\n========== SUMMARY ==========")
passed = sum(1 for ok, _ in results if ok)
failed = [msg for ok, msg in results if not ok]
print(f"passed: {passed}/{len(results)}")
if failed:
    print(f"failed: {len(failed)}")
    for msg in failed:
        print(f"  - {msg}")
    sys.exit(1)
print("all pass")
