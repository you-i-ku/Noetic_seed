"""M1b: 実ファイルのID順位・全文/出典・公開schemaから同じhandlerへの経路。"""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import memory, memory_links
from core.config import cap_tool_result, TOOL_RESULT_MAX_CHARS
from core.providers.base import ToolUseBlock
from core.runtime.conversation import ConversationRuntime
from core.runtime.permissions import PermissionEnforcer, PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tools.noetic_ext import NOETIC_TOOL_NAMES, register_noetic_tools
from core.runtime.tools.noetic_ext.memory import register
from tools import TOOLS, memory_tool
from test_outward_metrics import SeqProvider, _fire_harness, _run, fixed_io  # noqa: F401


APPROVAL = {"tool_intent": "本文を確認", "tool_expected_outcome": "保存内容がわかる", "note": "確認中"}


def _write(directory, name, rows):
    (directory / name).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(memory_tool, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(memory, "list_registered_tags", lambda: ["alpha", "beta"])
    monkeypatch.setattr(memory, "is_tag_registered", lambda tag: tag in {"alpha", "beta"})
    return tmp_path


def _entry(mid="mem_target", network="alpha", content=None):
    return {"id": mid, "network": network,
            "content": content if content is not None else "日本語の本文\n" + "詳細" * 110 + "\n末尾の証拠",
            "created_at": "2020-01-02 03:04:05", "updated_at": "2021-02-03 04:05:06",
            "origin": "tool:memory_store", "source_context": "保存当時の資料",
            "perspective": {"subject": "self"}, "metadata": {"confidence": 0.37},
            "embedding": [0.1, 0.2]}


@pytest.mark.parametrize("network", ["alpha", "beta", "_untagged"])
def test_full_content_provenance_without_archive_or_learning(store, monkeypatch, network):
    entry = _entry(network=network)
    _write(store, network + ".jsonl", [entry])
    _write(store, "memory_links.jsonl", [{"id": "link", "strength": 0.31}])
    spies = []
    for module, name in [(memory, "_embed_sync"), (memory_tool, "_embed_sync"),
                         (memory, "memory_network_search"), (memory_tool, "memory_network_search"),
                         (memory, "get_relevant_memories"),
                         (memory_links, "update_link_strength_used"),
                         (memory_links, "generate_co_activation_links")]:
        spy = Mock(side_effect=AssertionError("ID取得では呼ばない"))
        monkeypatch.setattr(module, name, spy)
        spies.append(spy)
    before = {p.name: p.read_bytes() for p in store.iterdir()}
    result = memory_tool._search_memory({"id": entry["id"], "query": "無関係な検索語"})
    source, content = result.split("\n", 1)
    assert content == entry["content"]
    assert json.loads(source) == {k: v for k, v in entry.items() if k not in {"content", "embedding"}}
    assert {p.name: p.read_bytes() for p in store.iterdir()} == before
    for spy in spies:
        spy.assert_not_called()


@pytest.mark.parametrize("stage,expected", [(1, "archive-exact"), (2, "memory-exact"),
                                            (3, "archive-partial"), (4, "memory-partial")])
def test_four_ranks_and_scan_order(store, stage, expected):
    # 部分一致を先に置き、各段を消すと次の段が勝つ。prefixによる特別扱いも検出。
    archive = [dict(id="prefix_mem_target_z", intent="archive-partial"),
               dict(id="prefix_mem_target_a", intent="wrong-archive-tie")]
    if stage == 1:
        archive.append(dict(id="mem_target", intent="archive-exact"))
    if stage >= 3:
        memories = []
    else:
        memories = [_entry("mem_target", content="memory-exact")]
    # load_all_memories は各networkの新しい行を先に読む。
    memories += [_entry("prefix_mem_target_a", content="wrong-memory-tie"),
                 _entry("prefix_mem_target_z", content="memory-partial")]
    _write(store, "archive_z.jsonl", archive if stage < 4 else [])
    _write(store, "archive_a.jsonl", [dict(id="older_mem_target", intent="wrong-older-file")] if stage < 4 else [])
    _write(store, "alpha.jsonl", memories)
    _write(store, "beta.jsonl", [_entry("beta_mem_target", "beta", "wrong-network-order")])
    result = memory_tool._search_memory({"id": "mem_target"})
    assert expected in result and "wrong-" not in result
    actual_id = "mem_target" if stage <= 2 else "prefix_mem_target_z"
    if stage in (1, 3):
        assert result == f"id={actual_id} time= tool= intent={expected} result="
    else:
        assert json.loads(result.split("\n", 1)[0])["id"] == actual_id


@pytest.mark.parametrize("archive", [False, True])
def test_empty_missing_and_unregistered_network(store, archive):
    if archive:
        _write(store, "archive_test.jsonl", [])
    assert memory_tool._search_memory({"id": "missing"}) == "ID 'missing' に一致するエントリなし"
    for args in ({}, {"query": "", "id": ""}):
        assert memory_tool._search_memory(args) == "エラー: queryまたはidを指定してください"
    expected = "記憶ファイルが空です" if archive else "記憶ファイルがまだありません"
    assert memory_tool._search_memory({"query": "anything", "id": ""}) == expected
    _write(store, "unregistered.jsonl", [_entry("hidden", "unregistered")])
    assert memory_tool._search_memory({"id": "hidden"}) == "ID 'hidden' に一致するエントリなし"


def test_missing_provenance_is_not_fabricated(store):
    _write(store, "alpha.jsonl", [{"id": "legacy", "network": "alpha", "content": "本文"}])
    source, content = memory_tool._search_memory({"id": "legacy"}).split("\n", 1)
    assert json.loads(source) == {"id": "legacy", "network": "alpha"}
    assert content == "本文"


def test_query_stays_archive_only_and_empty_id_falls_back(store, monkeypatch):
    _write(store, "archive_test.jsonl", [{"id": "archive", "intent": "needle"}])
    _write(store, "alpha.jsonl", [_entry(content="needle")])
    monkeypatch.setattr(memory_tool, "is_vector_ready", lambda: False)
    loader = Mock(side_effect=AssertionError("query検索では記憶を追加しない"))
    monkeypatch.setattr(memory_tool, "load_all_memories", loader)
    result = memory_tool._search_memory({"query": "needle", "id": ""})
    assert result == "[100%] id=archive time= tool= intent=needle"
    loader.assert_not_called()


@pytest.mark.parametrize("provider_name", ["anthropic", "openai", "claude_code"])
def test_provider_schema_id_only_reaches_registered_handler(store, provider_name):
    entry = _entry()
    _write(store, "alpha.jsonl", [entry])
    args = {"id": entry["id"], **APPROVAL}
    original = copy.deepcopy(args)
    requests = []
    provider = SeqProvider([[ToolUseBlock(id="call", name="search_memory", input=args)]], requests)
    provider.name = provider_name
    registry = ToolRegistry()
    register(registry, TOOLS)
    spec = registry.get("search_memory")
    assert spec.handler is TOOLS["search_memory"]["func"] is memory_tool._search_memory
    runtime = ConversationRuntime(provider, registry,
                                  permission_enforcer=PermissionEnforcer(PermissionMode.ALLOW))
    # fake provider のstream境界に実際に渡った定義を検証する。
    response = runtime._call_llm()
    published = next(s for s in requests[0].tools
                     if s.get("name", s.get("function", {}).get("name")) == "search_memory")
    schema = published["input_schema"] if "input_schema" in published else published["function"]["parameters"]
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for selectors in ({"id": "mem_target"}, {"query": "語"},
                      {"id": "mem_target", "query": "語"}, {"id": "", "query": "語"},
                      {"id": "mem_target", "query": ""}):
        assert validator.is_valid({**selectors, **APPROVAL})
    for selectors in ({}, {"id": "", "query": ""}):
        empty_args = {**selectors, **APPROVAL}
        validator.validate(empty_args)
        rec = runtime._execute_tool_use("empty", "search_memory", empty_args, push_session=False)
        assert rec.output == "エラー: queryまたはidを指定してください"
    for selectors in ({"id": 1}, {"id": "mem_target", "extra": 1}):
        assert not validator.is_valid({**selectors, **APPROVAL})
    for field in ("tool_intent", "tool_expected_outcome"):
        assert not validator.is_valid({k: v for k, v in args.items() if k != field})
    assert validator.is_valid({k: v for k, v in args.items() if k != "note"})
    call = response.tool_uses[0]
    validator.validate(call.input)
    if provider_name == "claude_code":
        output, is_error = requests[0].tool_executor(call.id, call.name, call.input)
    else:
        rec = runtime._execute_tool_use(call.id, call.name, call.input, push_session=False)
        output, is_error = rec.output, rec.is_error
    assert not is_error and output.split("\n", 1)[1] == entry["content"]
    assert args == original


def test_all_noetic_schemas_have_plain_object_root():
    """providerへ渡す全Noetic定義で最上位の合成schemaを検出する。"""
    registry = ToolRegistry()
    register_noetic_tools(registry, TOOLS)
    assert set(registry.all_names()) == NOETIC_TOOL_NAMES
    for spec in registry.list():
        schema = spec.to_anthropic_format()["input_schema"]
        assert schema.get("type") == "object", spec.name
        assert not {"anyOf", "oneOf", "allOf"}.intersection(schema), spec.name


def test_full_result_gets_existing_marker_in_fire(monkeypatch, fixed_io, store):
    """取得時切詰めと、fire表示時marker欠落を別々に検出する。"""
    entry = _entry(content="長" * (TOOL_RESULT_MAX_CHARS + 10) + "本文の末尾")
    _write(store, "alpha.jsonl", [entry])
    args = {"id": entry["id"], **APPROVAL}
    raw = memory_tool._search_memory(args)
    assert raw.endswith(entry["content"])
    harness = _fire_harness(
        monkeypatch, fixed_io,
        stages=[[ToolUseBlock(id="read", name="search_memory", input=args)]],
        proposal="1. [考える] → search_memory (pe2=40, pec=0.2)",
        env_overrides={"cap_tool_result": cap_tool_result, "TOOLS": TOOLS,
                       "controller": lambda *a: {"allowed_tools": {"search_memory"}, "tool_rank": {}}})
    runtime = harness[0]["plain"].__globals__["_runtime"]
    register(runtime.tool_registry, TOOLS)
    _run(harness[0]["plain"])
    saved = harness[1]["raw_events"][-1]
    assert saved["result"] == "[search_memory]\n" + cap_tool_result(raw)
    assert f"表示上 {TOOL_RESULT_MAX_CHARS}/{len(raw)}字に切詰" in saved["result"]
    assert "本文の末尾" not in saved["result"]
