"""test_reflect_tag_consideration.py — 段階11-B Phase 2 Step 2.6。

検証対象:
  - enabled_in_reflect=False (default) で TAG CONSIDERATION が
    _build_reflect_sections() prompt に現れない (opt-in 保証、P2 affordance)
  - enabled_in_reflect=True に切り替えた時 prompt に出現、
    header と template が期待通り
  - write_protected=True の pseudo-tag は memory_store で ValueError (誤 write 防止)
  - 既存 OPINIONS / ENTITIES 組立が tag_consideration 存在下でも破綻しない

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_reflect_tag_consideration.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_tagconsid_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

from core.memory import memory_store
from core.perspective import default_self_perspective
from core.reflection import _build_reflect_sections
from core.tag_registry import (
    get_tag_rules,
    is_tag_registered,
    list_registered_tags,
    register_standard_tags,
    register_tag,
)


register_standard_tags()


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


# =========================================================================
# Section A: tag_consideration pseudo-tag の存在と default opt-in off
# =========================================================================
print("=== Section A: tag_consideration pseudo-tag default state ===")

_assert(
    is_tag_registered("tag_consideration"),
    "A-1 register_standard_tags() で tag_consideration 登録済",
)

_rules = get_tag_rules("tag_consideration") or {}
_learning = _rules.get("learning_rules", {})
_reflect_sec = _rules.get("reflect_section", {})

_assert(
    _learning.get("write_protected") is True,
    "A-2 learning_rules.write_protected = True",
)
_assert(
    _learning.get("beta_plus") is False and _learning.get("bitemporal") is False
    and _learning.get("c_gradual_source") is False,
    "A-3 他 learning_rules は全て False (pseudo-tag、学習対象ではない)",
)
_assert(
    _reflect_sec.get("enabled_in_reflect") is False,
    "A-4 reflect_section.enabled_in_reflect = False (opt-in default)",
)


# =========================================================================
# Section B: default 状態で _build_reflect_sections() に TAG CONSIDERATION
#            が含まれないこと (opt-in 保証)
# =========================================================================
print("=== Section B: default で prompt に含まれない (opt-in 保証) ===")

_sections = _build_reflect_sections()
_assert(
    "TAG CONSIDERATION" not in _sections,
    "B-1 default で 'TAG CONSIDERATION' が prompt に出現しない",
)
_assert(
    "OPINIONS" in _sections,
    "B-2 既存 OPINIONS セクションは存在 (回帰ゼロ)",
)
_assert(
    "ENTITIES" in _sections,
    "B-3 既存 ENTITIES セクションは存在 (回帰ゼロ)",
)


# =========================================================================
# Section C: enabled_in_reflect=True 切替で prompt に出現
# =========================================================================
print("=== Section C: enabled_in_reflect=True で prompt に出現 ===")

# pseudo-tag の reflect_section を直接切替 (iku の update_selfmodel 相当の内部操作)
_entry = _tr._REGISTERED["tag_consideration"]
_entry["reflect_section"]["enabled_in_reflect"] = True

_sections_on = _build_reflect_sections()
_assert(
    "TAG CONSIDERATION" in _sections_on,
    "C-1 enabled_in_reflect=True で 'TAG CONSIDERATION' が prompt に含まれる",
)
_assert(
    "新 tag を発明" in _sections_on,
    "C-2 template 本文が prompt に展開される",
)
_assert(
    "tag_name: 理由" in _sections_on,
    "C-3 既存 tag 記録の書式ガイドが展開される",
)

# 元に戻す (opt-in default 維持)
_entry["reflect_section"]["enabled_in_reflect"] = False


# =========================================================================
# Section D: write_protected=True の pseudo-tag は memory_store で ValueError
# =========================================================================
print("=== Section D: write_protected で memory_store reject ===")

try:
    memory_store(
        "tag_consideration", "誤って書き込もうとした",
        origin="test", perspective=default_self_perspective(),
    )
    _assert(False, "D-1 write_protected tag への memory_store が通ってしまった (異常)")
except ValueError as e:
    _assert("write_protected" in str(e), f"D-1 write_protected ValueError 発火 (msg: {e})")
except Exception as e:
    _assert(False, f"D-1 想定外 exception ({type(e).__name__}: {e})")

# 通常 tag (entity) は従来通り書き込める
try:
    entry = memory_store(
        "entity", "書けるはず", {"entity_name": "test_ent"},
        origin="test", perspective=default_self_perspective(),
    )
    _assert(entry.get("network") == "entity", "D-2 通常 tag (entity) は従来通り書ける")
except Exception as e:
    _assert(False, f"D-2 通常 tag の書込に失敗 ({type(e).__name__}: {e})")


# =========================================================================
# Section E: dynamic tag を write_protected=True で register → 同様に reject
# =========================================================================
print("=== Section E: dynamic tag の write_protected 契約 ===")

register_tag(
    "my_meta",
    learning_rules={"write_protected": True},
    origin="dynamic",
    intent="meta-section 実験",
)
_assert(is_tag_registered("my_meta"), "E-1 dynamic write_protected tag 登録成功")

try:
    memory_store(
        "my_meta", "meta 書けないはず",
        origin="test", perspective=default_self_perspective(),
    )
    _assert(False, "E-2 dynamic write_protected tag への memory_store が通った (異常)")
except ValueError as e:
    _assert("write_protected" in str(e), f"E-2 dynamic write_protected も reject (msg: {e})")


# =========================================================================
# Section F: 既存 get_tags_with_rule との整合 (Phase 1 rule 機構への影響ゼロ)
# =========================================================================
print("=== Section F: Phase 1 get_tags_with_rule への影響ゼロ ===")

from core.tag_registry import get_tags_with_rule

_c_grad_tags = set(get_tags_with_rule("c_gradual_source"))
_assert(
    _c_grad_tags == {"entity"},
    f"F-1 c_gradual_source は entity のみ (tag_consideration や my_meta は含まれない、got {_c_grad_tags})",
)

_wp_tags = set(get_tags_with_rule("write_protected"))
_assert(
    _wp_tags == {"tag_consideration", "my_meta"},
    f"F-2 write_protected が rule 駆動で取得可能 ({_wp_tags})",
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
