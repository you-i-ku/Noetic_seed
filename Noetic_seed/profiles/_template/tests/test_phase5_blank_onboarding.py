"""test_phase5_blank_onboarding.py — 段階11-B Phase 5 Step 5.7。

検証対象:
  - 空 registry (register_standard_tags 未呼出) で主要 API が graceful:
    - get_tags_with_rule("c_gradual_source") が [] (Phase 1 rule 駆動化の自然延長)
    - memory_store でタグ不在 → ValueError (iku 自発 tag 発明を促す)
    - list_records("entity") が [] (空)
    - reflection 内の rule 駆動 lookup が空 records / 空 search に収束
  - _tool_memory_store 経由で inline register + 書込成功 → registered_tags.json
    / {new_tag}.jsonl が作られる (iku 初回 tool 呼出の体験シミュレーション)
  - inline register 後に get_tags_with_rule が反映

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_phase5_blank_onboarding.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_blank_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

from core.memory import list_records, memory_store
from core.perspective import default_self_perspective
from core.tag_registry import (
    get_tags_with_rule,
    is_tag_registered,
    list_registered_tags,
)

# 重要: register_standard_tags() は呼ばない (Phase 5 白紙 onboarding 前提)


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


# =========================================================================
# Section A: 空 registry 初期状態の確認
# =========================================================================
print("=== Section A: 空 registry 初期状態 ===")

_assert(list_registered_tags() == [], "A-1 registered_tags 空")
_assert(get_tags_with_rule("c_gradual_source") == [], "A-2 c_gradual_source tag=[] (Phase 1 回帰)")
_assert(get_tags_with_rule("beta_plus") == [], "A-3 beta_plus tag=[]")
_assert(get_tags_with_rule("write_protected") == [], "A-4 write_protected tag=[]")
_assert(is_tag_registered("entity") is False, "A-5 'entity' も未登録")
_assert(is_tag_registered("wm") is False, "A-6 'wm' も未登録")


# =========================================================================
# Section B: memory_store で未登録タグ → ValueError (iku 自発発明を促す)
# =========================================================================
print("=== Section B: memory_store で未登録タグは reject ===")

try:
    memory_store(
        "entity", "未登録状態で書けないはず",
        origin="test", perspective=default_self_perspective(),
        _auto_metadata=False,
    )
    _assert(False, "B-1 未登録タグに memory_store 通ってしまった (異常)")
except ValueError as e:
    _assert(
        "Invalid network" in str(e),
        f"B-1 未登録で ValueError (msg: {str(e)[:80]})",
    )


# =========================================================================
# Section C: list_records が空 network でも graceful []
# =========================================================================
print("=== Section C: list_records の graceful ===")

_assert(list_records("entity", limit=20) == [], "C-1 未登録 'entity' で [] 返却")
_assert(list_records("nonexistent", limit=20) == [], "C-2 未登録 'nonexistent' で [] 返却")


# =========================================================================
# Section D: _tool_memory_store で inline register (iku 初回体験)
# =========================================================================
print("=== Section D: _tool_memory_store 経由で inline register ===")

from tools.memory_tool import _tool_memory_store

# iku が最初の観察で「観察」tag を発明して保存
result_d1 = _tool_memory_store({
    "network": "観察",
    "content": "USB デバイスが接続された",
    "rules": {"beta_plus": False, "bitemporal": False},
    "tool_intent": "初めての観察を「観察」tag として記録",
})

_assert(
    "保存完了" in result_d1,
    f"D-1 inline register 後の書込成功 (msg: {result_d1[:80]})",
)
_assert(
    is_tag_registered("観察"),
    "D-2 '観察' tag が registered_tags.json に追加",
)
_assert(
    "観察" in get_tags_with_rule("beta_plus") or "観察" not in get_tags_with_rule("beta_plus"),
    # beta_plus=False で登録したので rule list には含まれない想定
    "D-3 get_tags_with_rule が inline register 後も正しく機能",
)
assert "観察" not in get_tags_with_rule("beta_plus"), "D-3 check: 観察 tag は beta_plus=False で rule list 外"


# =========================================================================
# Section E: jsonl 永続化確認
# =========================================================================
print("=== Section E: jsonl 永続化 ===")

obs_file = _tmp_memory / "観察.jsonl"
_assert(obs_file.exists(), f"E-1 観察.jsonl 生成確認 (path: {obs_file.name})")

# registered_tags.json も更新済
reg_path = _tmp_memory / "registered_tags.json"
_assert(reg_path.exists(), "E-2 registered_tags.json 生成")

# list_records で 1 件取得できる
recs = list_records("観察", limit=10)
_assert(len(recs) == 1, f"E-3 list_records で 1 件取得 (got {len(recs)})")
if recs:
    _assert(
        recs[0].get("content") == "USB デバイスが接続された",
        "E-4 content 保存",
    )


# =========================================================================
# Section F: 複数 tag 発明後の state 反映 (iku が "観察" "夢" 2 tag を発明)
# =========================================================================
print("=== Section F: 複数 tag 発明 ===")

result_f = _tool_memory_store({
    "network": "夢",
    "content": "想像した情景 — 海辺で光が揺れている",
    "rules": {"beta_plus": True, "bitemporal": False, "c_gradual_source": True},
    "tool_intent": "白紙から夢 tag を発明",
})
_assert("保存完了" in result_f, "F-1 夢 tag inline register 成功")
_assert(is_tag_registered("夢"), "F-2 夢 tag が登録済")
_assert("夢" in get_tags_with_rule("c_gradual_source"), "F-3 c_gradual_source rule で取得")
_assert(
    set(list_registered_tags()) == {"観察", "夢"},
    f"F-4 registered_tags = {{観察, 夢}} (got {set(list_registered_tags())})",
)


# =========================================================================
# Section G: Phase 1 rule 駆動 API が空 registry 相当に戻るかの回帰
# =========================================================================
print("=== Section G: Phase 1 互換 (reset 後の空 registry 挙動) ===")

# reset で空 registry 相当に戻し、Phase 1 回帰 check。
# 既存 registered_tags.json を load してしまうと空にならないので、
# 存在しない path を registry file に差し替えて reset (空 registry の挙動確認)。
_empty_reg = _tmp_memory / "empty_registry_for_section_G.json"
_tr._reset_for_testing(registry_file=_empty_reg)

_assert(list_registered_tags() == [], "G-1 reset 後 registered_tags 空に戻る")
_assert(get_tags_with_rule("c_gradual_source") == [], "G-2 空で []")
_assert(list_records("観察", limit=5) == [], "G-3 未登録 network で list_records []")


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
