"""test_step67_perspective_propagation.py — 段階11-A Step 6+7: 認知 unit 拡張。

正典 PLAN: WORLD_MODEL_DESIGN/STAGE11A_PERSPECTIVE_FOUNDATION_PLAN.md §7 Step 6/7

検証対象:
  Section 1: pending_add perspective 付与
    - kwarg 未指定 → default_self_perspective (self/actual、iku 自己 action の
      observation 待ち)
    - kwarg 指定 → その perspective が entry に入る (imagined etc.)
    - PendingEntry TypedDict の契約: entry に "perspective" キー
  Section 2: pending_add_response_intent (wrapper) 経由でも perspective 付与
  Section 3: memory_network_search view_filter
    - view_filter=None → 全視点 (既存挙動、デフォルト)
    - view_filter={"viewer": "self"} → self 視点 entry のみ
    - view_filter={"viewer": "device"} → attributed:device 視点のみ
    - perspective 欠落 entry (旧形式) は default self で判定 (backward compat)
  Section 4: output_display to_perspective (PLAN §5 表の仕様)
    - main.py の分岐ロジックを再現、first_args.channel を viewer にした
      perspective が entry に入ることを期待形で検証

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_step67_perspective_propagation.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_step67_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

from core.memory import memory_network_search, memory_store
from core.pending_unified import pending_add, pending_add_response_intent
from core.perspective import (
    default_self_perspective,
    is_self_view,
    make_perspective,
)
from core.tag_registry import register_standard_tags

register_standard_tags()


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    return bool(cond)


# =========================================================================
# Section 1: pending_add perspective 付与
# =========================================================================
print("=== Section 1: pending_add perspective ===")

# 1-A: kwarg 未指定 → default self/actual
state = {"pending": []}
entry_default = pending_add(
    state=state,
    source_action="output_display",
    expected_observation="応答の届き確認",
    lag_kind="cycles",
    content_intent="test default",
    cycle_id=1,
    channel="device",
)
_assert(
    "perspective" in entry_default,
    "1-1 pending entry に perspective キー",
)
_assert(
    is_self_view(entry_default["perspective"]),
    "1-2 kwarg 未指定 → self/actual (iku 自己 action の observation 待ち)",
)

# 1-B: kwarg 指定 (想像視点 pending — 他者の想定反応を待つような仮想 case)
imag_persp = make_perspective(
    viewer="imagined_yuu_reaction", viewer_type="imagined", confidence=0.5,
)
entry_imag = pending_add(
    state=state,
    source_action="elyth_post",
    expected_observation="想像上の反応 (仮想)",
    lag_kind="hours",
    content_intent="test imagined",
    cycle_id=2,
    perspective=imag_persp,
)
_assert(
    entry_imag["perspective"]["viewer_type"] == "imagined",
    "1-3 kwarg 指定 imagined → viewer_type 保持",
)
_assert(
    entry_imag["perspective"]["viewer"] == "imagined_yuu_reaction",
    "1-4 kwarg 指定 viewer 保持",
)

# 1-C: state['pending'] リストに perspective 付きで格納
_assert(
    len(state["pending"]) == 2
    and all("perspective" in p for p in state["pending"]),
    "1-5 state['pending'] 内 entry 全てに perspective",
)


# =========================================================================
# Section 2: pending_add_response_intent (wrapper 経由)
# =========================================================================
print("\n=== Section 2: pending_add_response_intent ===")

state2 = {"pending": []}
entry_resp = pending_add_response_intent(
    state=state2,
    channel="claude",
    text="こんにちは",
    cycle_id=3,
)
_assert(
    "perspective" in entry_resp and is_self_view(entry_resp["perspective"]),
    "2-1 response_intent wrapper 経由でも self/actual 付与 (iku 内部応答意図)",
)


# =========================================================================
# Section 3: memory_network_search view_filter
# =========================================================================
print("\n=== Section 3: memory_network_search view_filter ===")

# テスト用 opinion entry を 4 種用意 — 全 entry に共通キーワード "opinion" を
# 含めて embedding 非依存のキーワード fallback search でも全件ヒットさせる
memory_store(
    "opinion", "opinion self タグの意見内容",
    origin="test_section3",
    perspective=make_perspective(),
)
memory_store(
    "opinion", "opinion device タグの他者帰属",
    origin="test_section3",
    perspective=make_perspective(viewer="device", viewer_type="actual", confidence=0.5),
)
memory_store(
    "opinion", "opinion imagined タグの仮想",
    origin="test_section3",
    perspective=make_perspective(viewer="future_self", viewer_type="imagined", confidence=0.4),
)
# 旧形式 (perspective 欠落) も手書きで追加
_op_file = _tmp_memory / "opinion.jsonl"
legacy_line = {
    "id": "mem_legacy_old",
    "network": "opinion",
    "content": "opinion legacy (perspective 欠落)",
    "origin": "legacy",
    "source_context": "",
    "metadata": {},
    "created_at": "2026-04-01 10:00:00",
    "updated_at": "2026-04-01 10:00:00",
}
with open(_op_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(legacy_line, ensure_ascii=False) + "\n")

# 3-A: view_filter=None → 全視点 (4 件全部)
res_all = memory_network_search("opinion", networks=["opinion"], limit=10)
_assert(
    len(res_all) == 4,
    f"3-1 view_filter=None: 全 4 件ヒット (got {len(res_all)})",
)

# 3-B: view_filter={"viewer": "self"} → self 視点 + 欠落 (default self 補完) 2 件
res_self = memory_network_search(
    "opinion", networks=["opinion"], limit=10,
    view_filter={"viewer": "self"},
)
contents_self = [r.get("content", "") for r in res_self]
_assert(
    len(res_self) == 2,
    f"3-2 view_filter=self: 2 件 (self 明示 + legacy 欠落、got {len(res_self)})",
)
_assert(
    any("self タグ" in c for c in contents_self),
    "3-3 view_filter=self: 明示 self entry 含む",
)
_assert(
    any("legacy" in c for c in contents_self),
    "3-4 view_filter=self: 欠落 legacy 含む (default self 補完)",
)

# 3-C: view_filter={"viewer": "device"} → device 視点のみ 1 件
res_device = memory_network_search(
    "opinion", networks=["opinion"], limit=10,
    view_filter={"viewer": "device"},
)
_assert(len(res_device) == 1, f"3-5 view_filter=device: 1 件 (got {len(res_device)})")
_assert(
    "device タグ" in res_device[0].get("content", ""),
    "3-6 view_filter=device: device entry 内容一致",
)

# 3-D: view_filter={"viewer_type": "actual"} → imagined 以外 3 件
res_actual = memory_network_search(
    "opinion", networks=["opinion"], limit=10,
    view_filter={"viewer_type": "actual"},
)
contents_actual = [r.get("content", "") for r in res_actual]
_assert(
    len(res_actual) == 3,
    f"3-7 view_filter=actual: 3 件 (imagined 除外、got {len(res_actual)})",
)
_assert(
    not any("imagined" in c for c in contents_actual),
    "3-8 view_filter=actual: imagined entry 除外確認",
)

# 3-E: マッチ 0 件の時は空 list
res_zero = memory_network_search(
    "opinion", networks=["opinion"], limit=10,
    view_filter={"viewer": "nonexistent_viewer"},
)
_assert(res_zero == [], "3-9 filter でマッチ 0 件 → 空 list")


# =========================================================================
# Section 4: output_display to_perspective (期待形の再現テスト)
# =========================================================================
# main.py 本体の分岐は tests から直接呼ぶには重いので、条件ロジックを再現して
# 期待される entry 形を検証する。Section の意図は PLAN §5 表の仕様契約が
# 動作する ことを観察可能に残すこと (smoke で最終確認)。
print("\n=== Section 4: output_display to_perspective 契約 ===")

def _simulate_entry(tool_name: str, first_args: dict) -> dict:
    """main.py L854 付近の分岐を再現。"""
    entry = {
        "tool": tool_name,
        "perspective": make_perspective(),  # from: self/actual (Step 2 で付与)
    }
    if first_args:
        entry["args"] = first_args
    # 段階11-A Step 7: output_display の to_perspective
    if tool_name == "output_display" and isinstance(first_args, dict):
        _ch = first_args.get("channel")
        if isinstance(_ch, str) and _ch.strip():
            entry["to_perspective"] = make_perspective(
                viewer=_ch.strip(), viewer_type="actual",
            )
    return entry


# 4-A: output_display + channel → to_perspective 入る
e1 = _simulate_entry("output_display", {"content": "hi", "channel": "device"})
_assert(
    "to_perspective" in e1,
    "4-1 output_display + channel → entry に to_perspective キー",
)
_assert(
    e1["to_perspective"]["viewer"] == "device",
    "4-2 to_perspective.viewer = args.channel (device)",
)
_assert(
    e1["to_perspective"]["viewer_type"] == "actual",
    "4-3 to_perspective.viewer_type = actual",
)
_assert(
    is_self_view(e1["perspective"]),
    "4-4 from perspective は既存通り self (Step 2 で付与済)",
)

# 4-B: output_display + channel なし → to_perspective 入らない
e2 = _simulate_entry("output_display", {"content": "hi"})
_assert(
    "to_perspective" not in e2,
    "4-5 output_display で channel なし → to_perspective 入らない",
)

# 4-C: 他 tool → to_perspective 入らない
e3 = _simulate_entry("read_file", {"path": "x.py"})
_assert(
    "to_perspective" not in e3,
    "4-6 他 tool → to_perspective 入らない",
)

# 4-D: channel 空文字 → to_perspective 入らない
e4 = _simulate_entry("output_display", {"content": "hi", "channel": "   "})
_assert(
    "to_perspective" not in e4,
    "4-7 channel 空白のみ → to_perspective 入らない",
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
