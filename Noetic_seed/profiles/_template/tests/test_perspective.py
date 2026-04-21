"""test_perspective.py — 段階11-A Step 1: Perspective schema + 8 helper の検証。

正典 PLAN: WORLD_MODEL_DESIGN/STAGE11A_PERSPECTIVE_FOUNDATION_PLAN.md §3, §7 Step 1

検証対象:
  1. make_perspective (defaults, view_time 自動付与, kwargs 全指定, sparse)
  2. default_self_perspective (viewer=self, viewer_type=actual)
  3. is_self_view (self+actual のみ True)
  4. is_actual_view (viewer_type=actual で True、仮想/過去/未来で False)
  5. is_nested / perspective_depth (0 / 1 / 3 / 10 段)
  6. perspective_tag_str (self/imagined/past_self/future_self/other + ネスト)
  7. perspective_key_str (self / attributed: / imagined: / past_self: / future_self:)
  8. backward compat (欠損 field の default 挙動)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_perspective.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.perspective import (
    Perspective,
    default_self_perspective,
    is_actual_view,
    is_nested,
    is_self_view,
    make_perspective,
    perspective_depth,
    perspective_key_str,
    perspective_tag_str,
)


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    return bool(cond)


# =========================================================================
# Section 1: make_perspective 基本
# =========================================================================
print("=== Section 1: make_perspective ===")

p1 = make_perspective()
_assert(p1["viewer"] == "self", "1-1 defaults viewer=self")
_assert(p1["viewer_type"] == "actual", "1-2 defaults viewer_type=actual")
_assert(
    re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", p1.get("view_time", "")),
    "1-3 view_time ISO8601 UTC 秒精度自動付与",
)
_assert("confidence" not in p1, "1-4 confidence 未指定で sparse (キー省略)")
_assert("nested" not in p1, "1-5 nested 未指定で sparse (キー省略)")

p2 = make_perspective(
    viewer="ent_yuu",
    viewer_type="imagined",
    view_time="2026-04-21T18:00:00Z",
    confidence=0.6,
    nested={"viewer": "self", "viewer_type": "actual"},
)
_assert(p2["viewer"] == "ent_yuu", "1-6 kwargs viewer")
_assert(p2["viewer_type"] == "imagined", "1-7 kwargs viewer_type")
_assert(p2["view_time"] == "2026-04-21T18:00:00Z", "1-8 kwargs view_time 上書き")
_assert(p2["confidence"] == 0.6, "1-9 kwargs confidence")
_assert(p2["nested"]["viewer"] == "self", "1-10 kwargs nested")

# confidence=0.0 (falsy だが None ではない) は set される
p_conf0 = make_perspective(confidence=0.0)
_assert("confidence" in p_conf0, "1-11 confidence=0.0 (falsy) でも set される")
_assert(p_conf0["confidence"] == 0.0, "1-12 confidence=0.0 の値保持")

# =========================================================================
# Section 2: default_self_perspective
# =========================================================================
print("\n=== Section 2: default_self_perspective ===")

d = default_self_perspective()
_assert(d["viewer"] == "self", "2-1 default viewer=self")
_assert(d["viewer_type"] == "actual", "2-2 default viewer_type=actual")
_assert("view_time" in d, "2-3 default view_time 自動付与")
_assert("confidence" not in d, "2-4 default confidence なし (self/actual)")

# =========================================================================
# Section 3: is_self_view / is_actual_view
# =========================================================================
print("\n=== Section 3: is_self_view / is_actual_view ===")

p_self = make_perspective(viewer="self", viewer_type="actual")
p_self_past = make_perspective(viewer="self", viewer_type="past_self", view_time="2026-01-01T00:00:00Z")
p_other = make_perspective(viewer="ent_yuu", viewer_type="actual", confidence=0.6)
p_imag = make_perspective(viewer="fear_future", viewer_type="imagined", confidence=0.4)

_assert(is_self_view(p_self) is True, "3-1 self+actual → is_self_view True")
_assert(is_self_view(p_self_past) is False, "3-2 self+past_self → is_self_view False")
_assert(is_self_view(p_other) is False, "3-3 other+actual → is_self_view False")
_assert(is_self_view(p_imag) is False, "3-4 imagined → is_self_view False")

_assert(is_actual_view(p_self) is True, "3-5 self+actual → is_actual_view True")
_assert(is_actual_view(p_other) is True, "3-6 other+actual → is_actual_view True")
_assert(is_actual_view(p_self_past) is False, "3-7 past_self → is_actual_view False")
_assert(is_actual_view(p_imag) is False, "3-8 imagined → is_actual_view False")

# backward compat: 欠損 field 時の挙動
_assert(is_self_view({}) is False, "3-9 空 dict → is_self_view False (欠損)")
_assert(is_actual_view({}) is False, "3-10 空 dict → is_actual_view False (欠損)")

# =========================================================================
# Section 4: is_nested / perspective_depth
# =========================================================================
print("\n=== Section 4: is_nested / perspective_depth ===")

p_flat = make_perspective()
_assert(is_nested(p_flat) is False, "4-1 フラット perspective → is_nested False")
_assert(perspective_depth(p_flat) == 0, "4-2 フラット → depth=0")

p_1 = make_perspective(nested=make_perspective(viewer="ent_yuu", viewer_type="actual"))
_assert(is_nested(p_1) is True, "4-3 1 段ネスト → is_nested True")
_assert(perspective_depth(p_1) == 1, "4-4 1 段ネスト → depth=1")

# 3 段ネスト: self ← yuu ← claude ← unknown
p_3 = make_perspective(
    nested=make_perspective(
        viewer="ent_yuu",
        viewer_type="actual",
        nested=make_perspective(
            viewer="claude",
            viewer_type="actual",
            nested=make_perspective(viewer="unknown", viewer_type="imagined"),
        ),
    ),
)
_assert(perspective_depth(p_3) == 3, "4-5 3 段ネスト → depth=3")

# 10 段ネスト (PLAN §7 Step 1 指定)
current: Perspective = make_perspective(viewer="layer_10", viewer_type="actual")
for i in range(9, 0, -1):
    current = make_perspective(viewer=f"layer_{i}", viewer_type="actual", nested=current)
_assert(perspective_depth(current) == 9, "4-6 10 段ネスト (外側9段+最内) → depth=9")
# ↑ 呼び方注意: "10 段ネスト" は外側から内側まで 10 層だが、depth は nested カウントなので 9
# 完全 10 depth を作るには 11 層必要
current11: Perspective = make_perspective(viewer="layer_11", viewer_type="actual")
for i in range(10, 0, -1):
    current11 = make_perspective(viewer=f"layer_{i}", viewer_type="actual", nested=current11)
_assert(perspective_depth(current11) == 10, "4-7 depth=10 構築可能 (構造上無制限確認)")

# =========================================================================
# Section 5: perspective_tag_str
# =========================================================================
print("\n=== Section 5: perspective_tag_str ===")

_assert(perspective_tag_str(p_self) == "[self]", "5-1 self/actual → [self]")
_assert(
    perspective_tag_str(p_other) == "[ent_yuu view]",
    "5-2 other/actual → [<viewer> view]",
)
_assert(
    perspective_tag_str(p_imag) == "[imagined:fear_future]",
    "5-3 imagined → [imagined:<viewer>]",
)

p_past = make_perspective(viewer="self", viewer_type="past_self", view_time="2026-01-01T00:00:00Z")
_assert(
    perspective_tag_str(p_past) == "[past_self@2026-01-01T00:00:00Z]",
    "5-4 past_self → [past_self@<time>]",
)

p_future = make_perspective(viewer="self", viewer_type="future_self", view_time="2027-01-01T00:00:00Z")
_assert(
    perspective_tag_str(p_future) == "[future_self@2027-01-01T00:00:00Z]",
    "5-5 future_self → [future_self@<time>]",
)

# ネスト表記 "A←B"
p_nest = make_perspective(nested=make_perspective(viewer="ent_yuu", viewer_type="actual"))
_assert(
    perspective_tag_str(p_nest) == "[self]←[ent_yuu view]",
    "5-6 ネスト → A←B 形式",
)

# 深ネスト
p_deep = make_perspective(
    nested=make_perspective(
        viewer="ent_yuu",
        viewer_type="actual",
        nested=make_perspective(viewer="future_me", viewer_type="future_self", view_time="2030-01-01T00:00:00Z"),
    ),
)
_assert(
    perspective_tag_str(p_deep)
    == "[self]←[ent_yuu view]←[future_self@2030-01-01T00:00:00Z]",
    "5-7 3 段ネスト → A←B←C",
)

# =========================================================================
# Section 6: perspective_key_str
# =========================================================================
print("\n=== Section 6: perspective_key_str ===")

_assert(perspective_key_str(p_self) == "self", "6-1 self/actual → 'self'")
_assert(
    perspective_key_str(p_other) == "attributed:ent_yuu",
    "6-2 other/actual → 'attributed:<viewer>'",
)
_assert(
    perspective_key_str(p_imag) == "imagined:fear_future",
    "6-3 imagined → 'imagined:<viewer>'",
)
_assert(
    perspective_key_str(p_past) == "past_self:self",
    "6-4 past_self → 'past_self:<viewer>'",
)
_assert(
    perspective_key_str(p_future) == "future_self:self",
    "6-5 future_self → 'future_self:<viewer>'",
)

# ID stable: view_time 違っても同 key (safe for state["dispositions"])
p_self_t1 = make_perspective(view_time="2026-04-21T18:00:00Z")
p_self_t2 = make_perspective(view_time="2026-04-21T19:00:00Z")
_assert(
    perspective_key_str(p_self_t1) == perspective_key_str(p_self_t2),
    "6-6 key_str は view_time 依存しない (ID stable)",
)

# 欠損 field 時の安全 default
_assert(
    perspective_key_str({}) == "attributed:?",
    "6-7 空 dict → 'attributed:?' (default viewer=?, viewer_type=actual)",
)

# =========================================================================
# Section 7: backward compat (既存 memory 欠損への耐性)
# =========================================================================
print("\n=== Section 7: backward compat ===")

# 既存 memory entry で perspective 欠落 = default_self_perspective() と解釈
rec_old = {"id": "mem_xxx", "content": "old", "network": "wm"}  # perspective なし
persp = rec_old.get("perspective") or default_self_perspective()
_assert(is_self_view(persp), "7-1 perspective 欠落 → default_self_perspective 補完で self view")

# メタデータ付き既存 entry (perspective 欠落でも metadata は無影響)
rec_old_meta = {"id": "mem_yyy", "content": "old", "metadata": {"k": "v"}}
persp2 = rec_old_meta.get("perspective") or default_self_perspective()
_assert(is_actual_view(persp2), "7-2 metadata ありの古い entry でも default 補完で actual")

# =========================================================================
# Summary
# =========================================================================
print("\n=== Summary ===")
passed = sum(1 for r, _ in results if r)
failed = sum(1 for r, _ in results if not r)
for r, m in results:
    if not r:
        print(f"  FAIL: {m}")
print(f"\nPASSED: {passed} / {passed + failed}")
if failed:
    sys.exit(1)
