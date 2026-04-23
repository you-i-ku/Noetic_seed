"""test_memory_keywords.py — 段階11-B Phase 3 Step 3.1。

検証対象:
  - memory_store の signature 拡張 (keywords / contextual_description keyword-only)
  - 明示 pass で override (test / migration 用途)
  - _auto_metadata=False で LLM skip (空値格納、graceful fallback と同挙動)
  - LLM mock 経由で自動生成 (keywords 3-7 個、description 文字列)
  - LLM error (raise) 時の graceful fallback (keywords=[] / description="")
  - jsonl 永続化で 2 field が保存される
  - 旧 entry (keywords/description 欠落) 読取で KeyError 無し (backward compat)

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_memory_keywords.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp_root = Path(tempfile.mkdtemp(prefix="noetic_test_keywords_"))
_tmp_memory = _tmp_root / "memory"
_tmp_memory.mkdir()

import core.config as _cfg
_cfg.MEMORY_DIR = _tmp_memory

import core.memory as _mem
_mem.MEMORY_DIR = _tmp_memory
import core.tag_registry as _tr
_tr._REGISTRY_FILE = _tmp_memory / "registered_tags.json"
_tr._reset_for_testing(registry_file=_tr._REGISTRY_FILE)

from core.memory import (
    _parse_metadata_response,
    list_records,
    memory_store,
)
from core.perspective import default_self_perspective
from core.tag_registry import register_standard_tags


register_standard_tags()


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


# =========================================================================
# Section A: 明示 pass で override (_auto_metadata=True でも明示優先)
# =========================================================================
print("=== Section A: 明示 pass override ===")

entry_a = memory_store(
    "entity", "明示テスト", {"entity_name": "test_a"},
    origin="test", perspective=default_self_perspective(),
    keywords=["a", "b", "c"],
    contextual_description="明示 pass デスクリプション",
    _auto_metadata=False,  # LLM skip 確認
)
_assert(entry_a.get("keywords") == ["a", "b", "c"], "A-1 keywords 明示値保存")
_assert(
    entry_a.get("contextual_description") == "明示 pass デスクリプション",
    "A-2 contextual_description 明示値保存",
)


# =========================================================================
# Section B: _auto_metadata=False で LLM skip (空値格納)
# =========================================================================
print("=== Section B: _auto_metadata=False で skip ===")

entry_b = memory_store(
    "entity", "skip テスト", {"entity_name": "test_b"},
    origin="test", perspective=default_self_perspective(),
    _auto_metadata=False,
)
_assert(entry_b.get("keywords") == [], "B-1 keywords=[] (skip で空)")
_assert(entry_b.get("contextual_description") == "", "B-2 contextual_description='' (skip で空)")


# =========================================================================
# Section C: LLM mock 経由で自動生成 (_auto_metadata=True, monkey patch)
# =========================================================================
print("=== Section C: LLM mock で自動生成 ===")


def _mock_generate(content, network):
    return {
        "keywords": [f"kw_{network}_1", f"kw_{network}_2", f"kw_{network}_3"],
        "contextual_description": f"mock desc for {network}: {content[:30]}",
    }


_orig_gen = _mem._generate_memory_metadata
_mem._generate_memory_metadata = _mock_generate
try:
    entry_c = memory_store(
        "entity", "mock テスト", {"entity_name": "test_c"},
        origin="test", perspective=default_self_perspective(),
    )  # _auto_metadata 省略 = default True
    _assert(
        entry_c.get("keywords") == ["kw_entity_1", "kw_entity_2", "kw_entity_3"],
        f"C-1 keywords mock 値取込 (got {entry_c.get('keywords')})",
    )
    _assert(
        "mock desc for entity" in (entry_c.get("contextual_description") or ""),
        f"C-2 contextual_description mock 値取込",
    )
finally:
    _mem._generate_memory_metadata = _orig_gen


# =========================================================================
# Section D: LLM error (raise) 時の graceful fallback
# =========================================================================
print("=== Section D: LLM error で graceful fallback ===")


def _mock_raise(content, network):
    raise RuntimeError("LLM connection failed (test)")


_mem._generate_memory_metadata = _mock_raise
try:
    entry_d = memory_store(
        "entity", "error テスト", {"entity_name": "test_d"},
        origin="test", perspective=default_self_perspective(),
    )
    _assert(entry_d.get("keywords") == [], "D-1 LLM error 時 keywords=[] fallback")
    _assert(
        entry_d.get("contextual_description") == "",
        "D-2 LLM error 時 contextual_description='' fallback",
    )
    _assert(entry_d.get("id", "").startswith("mem_"), "D-3 memory 書込自体は継続 (id 生成)")
finally:
    _mem._generate_memory_metadata = _orig_gen


# =========================================================================
# Section E: jsonl 永続化 — 2 field が正しく保存される
# =========================================================================
print("=== Section E: jsonl 永続化 ===")

fpath = _tmp_memory / "entity.jsonl"
_assert(fpath.exists(), "E-1 entity.jsonl 作成済")
lines = fpath.read_text(encoding="utf-8").splitlines()
latest = [json.loads(l) for l in lines if l.strip()]

for e in latest:
    _assert("keywords" in e, f"E-2 entry {e.get('id')} に keywords field あり")
    _assert(
        "contextual_description" in e,
        f"E-3 entry {e.get('id')} に contextual_description field あり",
    )

# 明示 pass entry が jsonl でも保存されているか
_found_a = [e for e in latest if e.get("id") == entry_a.get("id")]
_assert(len(_found_a) == 1 and _found_a[0].get("keywords") == ["a", "b", "c"],
        "E-4 明示 pass した keywords が jsonl でも保存")


# =========================================================================
# Section F: 旧 entry (keywords/description 欠落) 読取で KeyError 無し
#            (list_records / memory 系読取側の後方互換)
# =========================================================================
print("=== Section F: backward compat (旧 entry 読取) ===")

# 手動で「旧 entry」を jsonl に混入 (keywords / contextual_description 欠落)
legacy_entry = {
    "id": "mem_legacy_test1",
    "network": "entity",
    "content": "旧世代 entry",
    "origin": "legacy",
    "source_context": "",
    "metadata": {"entity_name": "legacy_ent"},
    "perspective": default_self_perspective(),
    "created_at": "2026-04-01 00:00:00",
    "updated_at": "2026-04-01 00:00:00",
    # keywords / contextual_description なし
}
with open(fpath, "a", encoding="utf-8") as f:
    f.write(json.dumps(legacy_entry, ensure_ascii=False) + "\n")

recs = list_records("entity", limit=100)
_legacy = [r for r in recs if r.get("id") == "mem_legacy_test1"]
_assert(len(_legacy) == 1, "F-1 旧 entry が list_records で読める")
_assert(
    _legacy[0].get("keywords", []) == [],
    "F-2 旧 entry の keywords を entry.get で空 list 扱い (KeyError 無し)",
)
_assert(
    _legacy[0].get("contextual_description", "") == "",
    "F-3 旧 entry の contextual_description を entry.get で空文字扱い",
)


# =========================================================================
# Section G: _parse_metadata_response の robustness 単体確認
# =========================================================================
print("=== Section G: _parse_metadata_response robustness ===")

_p1 = _parse_metadata_response('{"keywords": ["a", "b"], "contextual_description": "desc"}')
_assert(_p1["keywords"] == ["a", "b"] and _p1["contextual_description"] == "desc",
        "G-1 純粋 JSON をパース")

_p2 = _parse_metadata_response('前置き\n{"keywords": ["x"], "contextual_description": "y"}\n後置き')
_assert(_p2["keywords"] == ["x"] and _p2["contextual_description"] == "y",
        "G-2 前後文付きの JSON を抽出")

_p3 = _parse_metadata_response('not a json at all')
_assert(_p3["keywords"] == [] and _p3["contextual_description"] == "",
        "G-3 非 JSON で graceful 空返却")

_p4 = _parse_metadata_response('{"keywords": "not a list", "contextual_description": 42}')
_assert(_p4["keywords"] == [] and _p4["contextual_description"] == "42",
        f"G-4 型不整合で定型化 (got {_p4})")

_p5 = _parse_metadata_response(
    '{"keywords": ["' + "a," * 20 + '"], "contextual_description": "' + "x" * 600 + '"}'
)
_assert(len(_p5["keywords"]) <= 7, f"G-5 keywords 7 個まで切詰 (got {len(_p5['keywords'])})")
_assert(len(_p5["contextual_description"]) <= 500, f"G-6 description 500 文字まで切詰")


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
