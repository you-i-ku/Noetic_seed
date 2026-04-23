"""test_write_protected.py — 段階11-B Phase 2' (partial scope-down)。

検証対象 (汎用 pseudo-tag / write_protected schema):
  - register_tag の learning_rules に write_protected key が明示列挙 (4 key schema)
  - write_protected=True の dynamic tag は memory_store で ValueError (誤 write 防止)
  - 通常 tag (entity) の write path は影響ゼロ (回帰)
  - get_tags_with_rule("write_protected") で rule 駆動取得可能 (Phase 1 API 汎用)
  - enabled_in_reflect=False の dynamic pseudo-tag は _build_reflect_sections() に
    出現しない (meta-section opt-in 機構の動作確認)

Note: Phase 2' で tag_consideration pseudo-tag は撤去 (リマインダーは動的タグ生成
  の tool spec で既に可視、冗長)。write_protected schema と reflect_section opt-in
  機構は汎用の足場として保持、Phase 5 白紙 onboarding 後に再利用判断。

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_write_protected.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_wp_"))
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
    get_tags_with_rule,
    is_tag_registered,
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
# Section A: schema 明示列挙 — write_protected が rules_norm の 4 key 目
# =========================================================================
print("=== Section A: learning_rules schema 明示列挙 ===")

register_tag(
    "schema_check",
    learning_rules={"beta_plus": True, "bitemporal": False},  # write_protected 省略
    origin="dynamic",
)
_rules = get_tag_rules("schema_check") or {}
_learning = _rules.get("learning_rules", {})

_assert(
    set(_learning.keys()) == {"beta_plus", "bitemporal", "c_gradual_source", "write_protected"},
    f"A-1 rules_norm 4 key (beta_plus/bitemporal/c_gradual_source/write_protected)",
)
_assert(
    _learning["write_protected"] is False,
    "A-2 省略時 default False で格納 (明示列挙、サイレント受理なし)",
)


# =========================================================================
# Section B: write_protected=True の dynamic tag は memory_store で reject
# =========================================================================
print("=== Section B: write_protected で memory_store reject ===")

register_tag(
    "my_meta",
    learning_rules={"write_protected": True},
    origin="dynamic",
    intent="pseudo-tag 実験",
)
_assert(is_tag_registered("my_meta"), "B-1 dynamic write_protected tag 登録成功")

try:
    memory_store(
        "my_meta", "書けないはず",
        origin="test", perspective=default_self_perspective(),
    )
    _assert(False, "B-2 write_protected tag への memory_store が通ってしまった (異常)")
except ValueError as e:
    _assert("write_protected" in str(e), f"B-2 write_protected ValueError 発火 (msg: {e})")
except Exception as e:
    _assert(False, f"B-2 想定外 exception ({type(e).__name__}: {e})")


# =========================================================================
# Section C: 通常 tag (entity) の write path は影響ゼロ
# =========================================================================
print("=== Section C: 通常 tag は従来通り write 可能 ===")

try:
    entry = memory_store(
        "entity", "書けるはず", {"entity_name": "test_ent"},
        origin="test", perspective=default_self_perspective(),
    )
    _assert(entry.get("network") == "entity", "C-1 通常 tag (entity) は従来通り書ける")
except Exception as e:
    _assert(False, f"C-1 通常 tag の書込に失敗 ({type(e).__name__}: {e})")


# =========================================================================
# Section D: get_tags_with_rule("write_protected") 汎用取得
# =========================================================================
print("=== Section D: Phase 1 rule 駆動 API の write_protected 対応 ===")

_wp_tags = set(get_tags_with_rule("write_protected"))
_assert(
    _wp_tags == {"my_meta"},
    f"D-1 write_protected tag は my_meta のみ (got {_wp_tags})",
)

_c_grad_tags = set(get_tags_with_rule("c_gradual_source"))
_assert(
    _c_grad_tags == {"entity"},
    f"D-2 c_gradual_source は entity のみ (Phase 1 挙動に影響なし、got {_c_grad_tags})",
)


# =========================================================================
# Section E: enabled_in_reflect=False の dynamic pseudo-tag は prompt に出ない
# =========================================================================
print("=== Section E: meta-section opt-in (enabled_in_reflect=False) ===")

register_tag(
    "future_meta",
    learning_rules={"write_protected": True},
    origin="dynamic",
    reflect_section={
        "header": "FUTURE META (opt-in)",
        "template": "- (未来の affordance)",
        "enabled_in_reflect": False,
    },
)
_sections = _build_reflect_sections()
_assert(
    "FUTURE META" not in _sections,
    "E-1 enabled_in_reflect=False の reflect_section は prompt に出現しない",
)
_assert(
    "OPINIONS" in _sections and "ENTITIES" in _sections,
    "E-2 既存 OPINIONS / ENTITIES は従来通り出現 (回帰ゼロ)",
)


# =========================================================================
# Section F: enabled_in_reflect=True の dynamic tag は prompt に出現
#            (将来の pseudo-tag affordance を自発 on にする機構の足場確認)
# =========================================================================
print("=== Section F: dynamic reflect_section ON で prompt 出現 ===")

# future_meta の enabled を直接 True に切替 (iku 将来 tool 相当の内部操作)
_entry = _tr._REGISTERED["future_meta"]
_entry["reflect_section"]["enabled_in_reflect"] = True

_sections_on = _build_reflect_sections()
_assert(
    "FUTURE META" in _sections_on,
    "F-1 enabled_in_reflect=True で dynamic reflect_section が prompt に出現",
)
_assert(
    "(未来の affordance)" in _sections_on,
    "F-2 template 本文が展開される",
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
