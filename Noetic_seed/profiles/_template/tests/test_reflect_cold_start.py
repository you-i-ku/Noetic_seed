"""test_reflect_cold_start.py — 段階11-C G-lite Phase 3 Step 3.4。

正典 PLAN: WORLD_MODEL_DESIGN/STAGE11C_G_LITE_PLAN.md §5 Phase 3

検証対象:
  Section A: _build_reflect_sections(visibility_mode) 単体
    - デフォルト引数省略 = "visible" と同値 (backward compat)
    - "visible" で opinion/entity reflect_section が組み立てられる
    - "cold_start" で空文字列 (tag prior 完全除外)
  Section B: settings.reflection.reflect_cold_start_mode 配線
    - False / キー無 → reflect prompt に OPINIONS / ENTITIES header 出現
    - True → reflect prompt に header 出現しない、dynamic_sections が空
    - どちらの場合も SELF_DISPOSITION / ATTRIBUTED_DISPOSITION は残る
      (cold_start は reflect_section ハードコードのみ影響、本体指示文は不変)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" \\
      tests/test_reflect_cold_start.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_cold_start_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

from core.reflection import _build_reflect_sections, reflect
from core.tag_registry import register_standard_tags


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


# =========================================================================
# Section A: _build_reflect_sections(visibility_mode) 単体
# =========================================================================
print("=== Section A: _build_reflect_sections 単体 ===")

# 標準タグ (opinion/entity) 登録して reflect_section を用意
register_standard_tags()

# A-1: デフォルト省略 = "visible" と同値
sec_default = _build_reflect_sections()
sec_visible = _build_reflect_sections("visible")
_assert(
    sec_default == sec_visible,
    "A-1 引数省略 = 'visible' (backward compat)",
)

# A-2: "visible" で opinion/entity header が出現
_assert(
    "OPINIONS" in sec_visible,
    f"A-2a 'visible' に OPINIONS header 含む (got len={len(sec_visible)})",
)
_assert(
    "ENTITIES" in sec_visible,
    "A-2b 'visible' に ENTITIES header 含む",
)
_assert(
    len(sec_visible) > 0,
    "A-2c 'visible' で非空文字列",
)

# A-3: "cold_start" で空文字列
sec_cold = _build_reflect_sections("cold_start")
_assert(
    sec_cold == "",
    f"A-3 'cold_start' で空文字列 (got repr={sec_cold!r})",
)

# A-4: 未知 mode は visible 相当 (graceful)
sec_unknown = _build_reflect_sections("unknown_mode_xyz")
_assert(
    sec_unknown == sec_visible,
    "A-4 未知 mode は 'visible' fallback (graceful)",
)


# =========================================================================
# Section B: settings 配線で reflect prompt 切替
# =========================================================================
print("=== Section B: settings.reflection.reflect_cold_start_mode 配線 ===")

# reflect() 内部の prompt を capture するための mock llm
_captured_prompt = {"last": None}


def _mock_llm(prompt, **kwargs):
    _captured_prompt["last"] = prompt
    # reflect() は call_llm_fn の戻り値を _parse_reflection に渡す、
    # 最小限の空 reflection 応答で OK (exception 回避)
    return "SELF_DISPOSITION:\n\nATTRIBUTED_DISPOSITION:\n"


def _minimal_state():
    """reflect に渡す最小 state。"""
    return {
        "self": {"name": "test"},
        "dispositions": {"self": {}},
        "log": [],
        "pending": [],
    }


# settings 操作: llm_cfg は core.config グローバル、test 中だけ書き換える
_original_reflection_cfg = _cfg.llm_cfg.get("reflection", None)


def _set_cold_start(value: bool):
    _cfg.llm_cfg["reflection"] = {"reflect_cold_start_mode": value}


def _restore_reflection_cfg():
    if _original_reflection_cfg is None:
        _cfg.llm_cfg.pop("reflection", None)
    else:
        _cfg.llm_cfg["reflection"] = _original_reflection_cfg


# B-1: cold_start=False → OPINIONS/ENTITIES prompt に含まれる
_set_cold_start(False)
_captured_prompt["last"] = None
reflect(_minimal_state(), _mock_llm)
prompt_visible = _captured_prompt["last"]
_assert(
    prompt_visible is not None,
    "B-1a prompt capture 成功",
)
_assert(
    "OPINIONS" in (prompt_visible or ""),
    "B-1b cold_start=False で OPINIONS header prompt に含まれる",
)
_assert(
    "ENTITIES" in (prompt_visible or ""),
    "B-1c cold_start=False で ENTITIES header prompt に含まれる",
)
_assert(
    "SELF_DISPOSITION" in (prompt_visible or ""),
    "B-1d SELF_DISPOSITION は cold_start に関係なく残る (本体指示文)",
)

# B-2: cold_start=True → OPINIONS/ENTITIES prompt から消える、SELF_DISPOSITION は残る
_set_cold_start(True)
_captured_prompt["last"] = None
reflect(_minimal_state(), _mock_llm)
prompt_cold = _captured_prompt["last"]
_assert(
    prompt_cold is not None,
    "B-2a prompt capture 成功",
)
_assert(
    "OPINIONS" not in (prompt_cold or ""),
    "B-2b cold_start=True で OPINIONS header prompt から消える",
)
_assert(
    "ENTITIES" not in (prompt_cold or ""),
    "B-2c cold_start=True で ENTITIES header prompt から消える",
)
_assert(
    "SELF_DISPOSITION" in (prompt_cold or ""),
    "B-2d cold_start=True でも SELF_DISPOSITION は残る (本体指示文不変)",
)
_assert(
    "ATTRIBUTED_DISPOSITION" in (prompt_cold or ""),
    "B-2e cold_start=True でも ATTRIBUTED_DISPOSITION は残る",
)

# B-3: reflection section 無で挙動 = False と同値 (backward compat)
_cfg.llm_cfg.pop("reflection", None)
_captured_prompt["last"] = None
reflect(_minimal_state(), _mock_llm)
prompt_no_setting = _captured_prompt["last"]
_assert(
    "OPINIONS" in (prompt_no_setting or ""),
    "B-3 reflection section 無 → False と同値 (OPINIONS 残、backward compat)",
)

# cleanup
_restore_reflection_cfg()


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
