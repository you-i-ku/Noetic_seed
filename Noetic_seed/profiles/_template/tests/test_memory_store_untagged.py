"""memory_store network 引数 optional 化テスト (段階11-D Phase 1 Step 1.1-1.5).

成功条件:
  - network 未指定 (空文字 or 省略) で保存できる
  - 保存先は `_untagged.jsonl` 専用ファイル
  - entry["network"] は UNTAGGED_NETWORK マーカー
  - rules / inline register 不要 (tag_registry を汚さない)
  - 既存 tag 付き保存は挙動不変 (回帰確認)
  - memory_network_search が untagged memory も hit
  - format_memories_for_prompt で `[untagged] {content}` 表示
  - list_records(UNTAGGED_NETWORK) で untagged memory 取得
  - memory_update / memory_forget が untagged memory も対象

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_memory_store_untagged.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.memory as memory_mod
import core.tag_registry as tr
from core.memory import (
    memory_store, memory_update, memory_forget,
    memory_network_search, list_records,
    format_memories_for_prompt, UNTAGGED_NETWORK,
)
from tools.memory_tool import _tool_memory_store


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup(tmp_path: Path):
    memory_mod.MEMORY_DIR = tmp_path
    reg_file = tmp_path / "registered_tags.json"
    tr._reset_for_testing(registry_file=reg_file)
    tr.register_standard_tags()
    return reg_file


def test_store_with_network_none(tmp_path: Path):
    print("== memory_store(network=None, ...) で untagged 保存 ==")
    _setup(tmp_path)
    entry = memory_store(
        network=None,
        content="タグなしの記憶テスト",
        _auto_metadata=False,
    )
    untagged_jsonl = tmp_path / f"{UNTAGGED_NETWORK}.jsonl"
    return all([
        _assert(entry["network"] == UNTAGGED_NETWORK,
                f"entry network = {UNTAGGED_NETWORK} (got: {entry['network']})"),
        _assert(untagged_jsonl.exists(), f"{UNTAGGED_NETWORK}.jsonl 生成"),
        _assert(entry["content"] == "タグなしの記憶テスト", "content 保存"),
        _assert(entry["id"].startswith("mem_"), f"id 採番 (got: {entry['id']})"),
    ])


def test_store_via_tool_empty_network(tmp_path: Path):
    print("== _tool_memory_store({network='', content=...}) で untagged 保存 ==")
    _setup(tmp_path)
    result = _tool_memory_store({
        "network": "",
        "content": "tool 経由で network なし保存",
    })
    untagged_jsonl = tmp_path / f"{UNTAGGED_NETWORK}.jsonl"
    return all([
        _assert("記憶保存完了" in result, f"保存完了メッセージ: {result[:80]}"),
        _assert(f"[{UNTAGGED_NETWORK}]" in result, f"[{UNTAGGED_NETWORK}] 表示: {result[:80]}"),
        _assert(untagged_jsonl.exists(), f"{UNTAGGED_NETWORK}.jsonl 生成"),
    ])


def test_store_via_tool_no_network_key(tmp_path: Path):
    print("== _tool_memory_store({content=...}) network key 省略でも untagged 保存 ==")
    _setup(tmp_path)
    result = _tool_memory_store({
        "content": "network key 省略",
    })
    untagged_jsonl = tmp_path / f"{UNTAGGED_NETWORK}.jsonl"
    return all([
        _assert("記憶保存完了" in result, f"保存完了メッセージ: {result[:80]}"),
        _assert(untagged_jsonl.exists(), f"{UNTAGGED_NETWORK}.jsonl 生成"),
    ])


def test_untagged_does_not_register_tag(tmp_path: Path):
    print("== untagged 保存で tag_registry に '_untagged' 登録されない ==")
    _setup(tmp_path)
    memory_store(network=None, content="x", _auto_metadata=False)
    return all([
        _assert(not tr.is_tag_registered(UNTAGGED_NETWORK),
                f"{UNTAGGED_NETWORK} は tag_registry 非登録"),
        _assert(UNTAGGED_NETWORK not in list(tr.list_registered_tags()),
                f"list_registered_tags に {UNTAGGED_NETWORK} 含まれない"),
    ])


def test_existing_tag_unchanged(tmp_path: Path):
    print("== 既存 tag 付き保存は挙動不変 (回帰確認) ==")
    _setup(tmp_path)
    entry = memory_store(
        network="opinion",
        content="意見テスト",
        metadata={"confidence": 0.7},
        _auto_metadata=False,
    )
    opinion_jsonl = tmp_path / "opinion.jsonl"
    untagged_jsonl = tmp_path / f"{UNTAGGED_NETWORK}.jsonl"
    return all([
        _assert(entry["network"] == "opinion", "entry network = opinion"),
        _assert(opinion_jsonl.exists(), "opinion.jsonl 生成"),
        _assert(not untagged_jsonl.exists(), f"{UNTAGGED_NETWORK}.jsonl は作られない"),
    ])


def test_search_includes_untagged(tmp_path: Path):
    print("== memory_network_search(networks=None) で untagged も hit ==")
    # 注: 既存 keyword fallback (\w+ tokenize) は Unicode greedy で
    # 日本語と英字混在を 1 token 化するため、ASCII で test。
    # vector embedding 経路は LM Studio 起動時のみ hit、test 環境では fallback。
    _setup(tmp_path)
    memory_store(network=None, content="hello untagged search test", _auto_metadata=False)
    results = memory_network_search(query="hello", limit=5)
    found = any(r.get("network") == UNTAGGED_NETWORK for r in results)
    return _assert(found, f"untagged memory が search 結果に含まれる (n={len(results)})")


def test_format_untagged_display(tmp_path: Path):
    print("== format_memories_for_prompt で [untagged] 表示 ==")
    _setup(tmp_path)
    memory_store(network=None, content="hello untagged display test", _auto_metadata=False)
    results = memory_network_search(query="hello", limit=5)
    text = format_memories_for_prompt(results)
    return _assert("[untagged]" in text, f"text に [untagged] 含む: {text[:100]}")


def test_list_records_untagged(tmp_path: Path):
    print("== list_records(UNTAGGED_NETWORK) で untagged memory 取得 ==")
    _setup(tmp_path)
    memory_store(network=None, content="A", _auto_metadata=False)
    memory_store(network=None, content="B", _auto_metadata=False)
    records = list_records(UNTAGGED_NETWORK, limit=10)
    return all([
        _assert(len(records) == 2, f"2 件取得 (got: {len(records)})"),
        _assert(all(r.get("network") == UNTAGGED_NETWORK for r in records),
                "全 record が UNTAGGED_NETWORK"),
    ])


def test_update_untagged(tmp_path: Path):
    print("== memory_update が untagged memory も対象 ==")
    _setup(tmp_path)
    entry = memory_store(network=None, content="旧 content", _auto_metadata=False)
    result = memory_update(entry["id"], content="新 content")
    records = list_records(UNTAGGED_NETWORK, limit=5)
    updated = [r for r in records if r["id"] == entry["id"]]
    return all([
        _assert("更新完了" in result, f"更新完了メッセージ: {result[:60]}"),
        _assert(len(updated) == 1, "entry 1 件存在"),
        _assert(updated[0]["content"] == "新 content", "content 更新済"),
    ])


def test_forget_untagged(tmp_path: Path):
    print("== memory_forget が untagged memory も対象 ==")
    _setup(tmp_path)
    entry = memory_store(network=None, content="削除対象", _auto_metadata=False)
    result = memory_forget(entry["id"])
    records = list_records(UNTAGGED_NETWORK, limit=5)
    return all([
        _assert("削除完了" in result, f"削除完了メッセージ: {result[:60]}"),
        _assert(len(records) == 0, f"entry 削除済 (残: {len(records)})"),
    ])


def run_all():
    print("=" * 60)
    print("test_memory_store_untagged.py (段階11-D Phase 1 Step 1.1-1.5)")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        results = []
        for test_fn in [
            test_store_with_network_none,
            test_store_via_tool_empty_network,
            test_store_via_tool_no_network_key,
            test_untagged_does_not_register_tag,
            test_existing_tag_unchanged,
            test_search_includes_untagged,
            test_format_untagged_display,
            test_list_records_untagged,
            test_update_untagged,
            test_forget_untagged,
        ]:
            sub = Path(td) / test_fn.__name__
            sub.mkdir(exist_ok=True)
            results.append(test_fn(sub))
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"結果: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
