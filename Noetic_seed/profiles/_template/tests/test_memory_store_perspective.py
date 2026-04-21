"""test_memory_store_perspective.py — 段階11-A Step 2: memory_store + tag_registry
の perspective / reflect_section 拡張検証。

正典 PLAN: WORLD_MODEL_DESIGN/STAGE11A_PERSPECTIVE_FOUNDATION_PLAN.md §4-2, §7 Step 2

検証対象:
  Section 1: memory_store perspective kwarg
    - kwarg 未指定 → default_self_perspective 補完
    - kwarg 指定 → その値が entry に入る (self/other/imagined)
    - 専用キー昇格: metadata と並列 (metadata 内に潜らせない)
    - jsonl 永続化 + list_records 読み戻しで perspective 含む
  Section 2: backward compat
    - 既存 jsonl entry (perspective 欠落) が list_records で壊れずに読める
    - ↑ 欠落 entry と新 entry 混在でも問題なし
  Section 3: tag_registry reflect_section (G1 抽象化受け皿)
    - STANDARD_TAGS の opinion / entity に reflect_section が定義されてる
    - register_standard_tags() 経由で opinion / entity に reflect_section 付く
    - wm / experience には reflect_section 無し (opt-out)
    - register_tag(..., reflect_section=...) で dynamic tag も付けられる (11-B 伏線)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_memory_store_perspective.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# tempdir に MEMORY_DIR を向けてテスト isolation
_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_persp_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

# core.config.MEMORY_DIR を差し替える (他モジュール import 前に)
import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

# memory.py と tag_registry.py は MEMORY_DIR を module-level で既に bind している
# 可能性があるので、強制的に再代入で上書き
import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

from core.memory import memory_store, list_records
from core.perspective import (
    default_self_perspective,
    is_self_view,
    make_perspective,
    perspective_key_str,
)
from core.tag_registry import (
    STANDARD_TAGS,
    get_tag_rules,
    register_standard_tags,
    register_tag,
)


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    return bool(cond)


# 標準タグを register (テスト前提)
register_standard_tags()


# =========================================================================
# Section 1: memory_store perspective kwarg
# =========================================================================
print("=== Section 1: memory_store perspective kwarg ===")

# 1-A: kwarg 未指定 → default_self_perspective
entry_default = memory_store(
    network="opinion",
    content="未指定テスト",
    origin="test_default",
)
_assert("perspective" in entry_default, "1-1 entry に perspective キーが入る")
_assert(
    entry_default["perspective"].get("viewer") == "self",
    "1-2 kwarg 未指定 → viewer=self (default 補完)",
)
_assert(
    entry_default["perspective"].get("viewer_type") == "actual",
    "1-3 kwarg 未指定 → viewer_type=actual",
)
_assert(is_self_view(entry_default["perspective"]), "1-4 is_self_view 判定 True")
_assert(
    "perspective" not in entry_default.get("metadata", {}),
    "1-5 perspective は metadata 内ではなく entry 専用キー (metadata と並列)",
)

# 1-B: kwarg 指定 (他者視点)
p_yuu = make_perspective(viewer="device", viewer_type="actual", confidence=0.7)
entry_other = memory_store(
    network="entity",
    content="ゆう観察 (他者視点例)",
    origin="test_other",
    perspective=p_yuu,
)
_assert(
    entry_other["perspective"]["viewer"] == "device",
    "1-6 kwarg 指定 → viewer=device",
)
_assert(
    entry_other["perspective"]["confidence"] == 0.7,
    "1-7 kwarg 指定 → confidence 保存",
)
_assert(
    perspective_key_str(entry_other["perspective"]) == "attributed:device",
    "1-8 perspective_key_str は 'attributed:device'",
)

# 1-C: kwarg 指定 (仮想視点)
p_imag = make_perspective(
    viewer="fear_future",
    viewer_type="imagined",
    confidence=0.4,
)
entry_imag = memory_store(
    network="opinion",
    content="想像視点テスト",
    origin="test_imagined",
    perspective=p_imag,
)
_assert(
    entry_imag["perspective"]["viewer_type"] == "imagined",
    "1-9 kwarg imagined → viewer_type=imagined",
)
_assert(
    perspective_key_str(entry_imag["perspective"]) == "imagined:fear_future",
    "1-10 imagined key_str",
)


# =========================================================================
# Section 1-D: jsonl 永続化 + list_records 読み戻し
# =========================================================================
print("\n=== Section 1-D: jsonl 永続化確認 ===")

# 直前 Section 1 で 3 件 entry を store した、うち opinion は 2 件、entity は 1 件
op_records = list_records("opinion", limit=10)
ent_records = list_records("entity", limit=10)

# opinion は新しい順に (entry_imag, entry_default) で返ってくる想定
_assert(len(op_records) == 2, f"1-11 opinion 2 件読み戻せる (got {len(op_records)})")
_assert(
    all("perspective" in r for r in op_records),
    "1-12 読み戻し後も perspective キー保持",
)
# 最新 entry_imag
_assert(
    op_records[0]["perspective"]["viewer_type"] == "imagined",
    "1-13 最新 entry は imagined (LIFO 読み順)",
)
_assert(
    op_records[1]["perspective"]["viewer"] == "self",
    "1-14 2 件目は default self",
)

# entity の確認
_assert(len(ent_records) == 1, "1-15 entity 1 件読み戻せる")
_assert(
    ent_records[0]["perspective"]["viewer"] == "device",
    "1-16 entity は viewer=device",
)


# =========================================================================
# Section 2: backward compat (既存 jsonl perspective 欠落)
# =========================================================================
print("\n=== Section 2: backward compat ===")

# experience.jsonl を手書きで「perspective 欠落 entry」を作成
legacy_line = {
    "id": "mem_legacy_001",
    "network": "experience",
    "content": "段階11-A 以前の entry",
    "origin": "legacy",
    "source_context": "",
    "metadata": {},
    "created_at": "2026-04-01 10:00:00",
    "updated_at": "2026-04-01 10:00:00",
    # ← perspective 欠落
}
_exp_file = _tmp_memory / "experience.jsonl"
with open(_exp_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(legacy_line, ensure_ascii=False) + "\n")

# 新 entry (perspective 付き) を混ぜる
memory_store(
    network="experience",
    content="段階11-A 以降の entry",
    origin="test_new",
)

recs = list_records("experience", limit=10)
_assert(len(recs) == 2, f"2-1 2 件 (legacy + new) 読み戻せる (got {len(recs)})")

# 新 entry (LIFO 順で先頭) は perspective 付き
_assert("perspective" in recs[0], "2-2 新 entry は perspective 付き")
# legacy entry は perspective 欠落のまま (破壊的改変なし)
_assert("perspective" not in recs[1], "2-3 legacy entry は perspective 欠落のまま保持 (破壊的改変なし)")

# default 補完は view 層の責務 — 読み取り側で default_self_perspective() or default_self_perspective
persp_for_legacy = recs[1].get("perspective") or default_self_perspective()
_assert(
    is_self_view(persp_for_legacy),
    "2-4 legacy entry を view 側で default 補完すると self/actual",
)


# =========================================================================
# Section 3: tag_registry reflect_section 抽象化
# =========================================================================
print("\n=== Section 3: tag_registry reflect_section ===")

# 3-A: STANDARD_TAGS に opinion / entity の reflect_section が定義されてる
_assert(
    "reflect_section" in STANDARD_TAGS["opinion"],
    "3-1 STANDARD_TAGS['opinion'] に reflect_section 定義あり",
)
_assert(
    STANDARD_TAGS["opinion"]["reflect_section"]["header"] == "OPINIONS",
    "3-2 opinion reflect_section.header = OPINIONS",
)
_assert(
    STANDARD_TAGS["opinion"]["reflect_section"]["enabled_in_reflect"] is True,
    "3-3 opinion enabled_in_reflect = True",
)
_assert(
    "reflect_section" in STANDARD_TAGS["entity"],
    "3-4 STANDARD_TAGS['entity'] に reflect_section 定義あり",
)
_assert(
    STANDARD_TAGS["entity"]["reflect_section"]["header"] == "ENTITIES",
    "3-5 entity reflect_section.header = ENTITIES",
)

# 3-B: wm / experience には定義なし (opt-out)
_assert(
    "reflect_section" not in STANDARD_TAGS["wm"],
    "3-6 wm に reflect_section なし (opt-out)",
)
_assert(
    "reflect_section" not in STANDARD_TAGS["experience"],
    "3-7 experience に reflect_section なし (opt-out)",
)

# 3-C: register_standard_tags() 経由で reflect_section が登録済 tag に入ってる
op_rules = get_tag_rules("opinion")
_assert(
    op_rules is not None and "reflect_section" in op_rules,
    "3-8 get_tag_rules('opinion') に reflect_section 入ってる",
)
_assert(
    op_rules["reflect_section"]["header"] == "OPINIONS",
    "3-9 registered opinion reflect_section 内容保持",
)

ent_rules = get_tag_rules("entity")
_assert(
    ent_rules is not None and "reflect_section" in ent_rules,
    "3-10 get_tag_rules('entity') にも reflect_section",
)

# wm は無し
wm_rules = get_tag_rules("wm")
_assert(
    wm_rules is not None and "reflect_section" not in wm_rules,
    "3-11 registered wm に reflect_section なし",
)


# =========================================================================
# Section 3-D: dynamic tag に reflect_section 付けられる (11-B 伏線)
# =========================================================================
print("\n=== Section 3-D: dynamic tag with reflect_section ===")

# 新規 dynamic tag を register (11-B で AI が自由発明する想定)
custom_section = {
    "header": "CUSTOM_REFLECT",
    "template": "- 自由発明した形式",
    "enabled_in_reflect": True,
}
dyn_entry = register_tag(
    "dynamic_custom_tag",
    learning_rules={"beta_plus": False, "bitemporal": False},
    origin="dynamic",
    intent="11-B 風の AI 自由発明タグ test",
    reflect_section=custom_section,
)
_assert(
    "reflect_section" in dyn_entry,
    "3-12 dynamic tag 登録時に reflect_section kwarg で付けられる",
)
_assert(
    dyn_entry["reflect_section"]["header"] == "CUSTOM_REFLECT",
    "3-13 dynamic reflect_section 内容保持",
)

# get_tag_rules で読み戻し
dyn_rules = get_tag_rules("dynamic_custom_tag")
_assert(
    dyn_rules is not None and dyn_rules.get("reflect_section") == custom_section,
    "3-14 dynamic tag get_tag_rules で reflect_section 読み戻し",
)

# reflect_section 省略 dynamic tag (opt-in 確認)
plain_entry = register_tag(
    "dynamic_plain_tag",
    learning_rules={"beta_plus": False, "bitemporal": False},
    origin="dynamic",
)
_assert(
    "reflect_section" not in plain_entry,
    "3-15 reflect_section 省略 dynamic tag には付かない (opt-in)",
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

# 一時ディレクトリ掃除
shutil.rmtree(_tmp_root, ignore_errors=True)

if failed:
    sys.exit(1)
