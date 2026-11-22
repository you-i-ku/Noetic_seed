"""C-2b memory tools: real JSONL reads and saved/secondary failure boundaries."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from core import memory, memory_links, state
from core.runtime.registry import ToolError, ToolResult
from tools import memory_tool, memory_graph_tool, world_fact_view_tool


@pytest.fixture
def store(tmp_path, monkeypatch):
    for module in (memory, memory_links, state, memory_tool):
        monkeypatch.setattr(module, "MEMORY_DIR", tmp_path)
    for module in (memory, memory_tool, memory_graph_tool):
        monkeypatch.setattr(module, "list_registered_tags", lambda: ["test"])
    for module in (memory, memory_tool):
        monkeypatch.setattr(module, "is_tag_registered", lambda tag: tag == "test")
    monkeypatch.setattr(memory, "get_tag_rules", lambda tag: {})
    for module in (memory, memory_tool, memory_graph_tool):
        monkeypatch.setattr(module, "is_vector_ready", lambda: False)
    monkeypatch.setattr(memory_tool, "load_state", lambda **kw: {})
    monkeypatch.setattr(memory_graph_tool, "load_state", lambda **kw: {})
    monkeypatch.setattr(memory, "_generate_memory_metadata", lambda *a: {})
    import core.reconciliation
    monkeypatch.setattr(core.reconciliation, "check_on_write", lambda *a, **kw: None)

    monkeypatch.setattr(world_fact_view_tool, "MEMORY_DIR", tmp_path)
    return tmp_path


def write(store, name, rows):
    (store / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) if isinstance(r, dict) else r for r in rows) + "\n", encoding="utf-8")


READERS = ["archive", "id", "network", "graph_memory", "graph_links", "world"]


def filename(reader):
    return {"archive": "archive_test.jsonl", "id": "_untagged.jsonl", "network": "_untagged.jsonl", "graph_memory": "_untagged.jsonl", "graph_links": "memory_links.jsonl", "world": "raw_events.jsonl"}[reader]


def invoke(reader, query="エラー"):
    if reader == "archive":
        return memory_tool._search_memory({"query": query})
    if reader == "id":
        return memory_tool._search_memory({"id": "mem_target" if query == "エラー" else "missing"})
    if reader == "network":
        return memory_tool._tool_search_memory({"query": query})
    if reader.startswith("graph"):
        return memory_graph_tool._memory_graph({"view": "ego"})
    return world_fact_view_tool._world_fact_view({"filter_tool": "missing"} if query == "missing" else {})


def row():
    return {"id": "mem_target", "content": "エラー", "intent": "エラー", "result": "エラー", "network": "_untagged", "from_id": "m1", "to_id": "m2", "link_type": "similar", "confidence": 0.8}


@pytest.mark.parametrize("reader", READERS)
def test_missing_empty_and_nonmatching_are_success(store, reader):
    assert isinstance(invoke(reader), str)
    write(store, filename(reader), ["", " "])
    assert isinstance(invoke(reader), str)
    write(store, filename(reader), [row()])
    assert isinstance(invoke(reader, "missing"), str)


@pytest.mark.parametrize("reader", READERS)
def test_partial_rows_and_error_word_are_success(store, reader):
    write(store, filename(reader), ["{broken", row(), "null"])
    result = invoke(reader)
    assert isinstance(result, ToolResult)
    assert result.detail == {"records_read": 1, "rows_unreadable": 2, "files_unreadable": 0, "empty_files": 0}
    if not reader.startswith("graph"):
        assert "エラー" in result.message
    # Reading succeeded even when the valid row does not match.
    assert isinstance(invoke(reader, "missing"), ToolResult)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("failure", ["rows", "encoding", "permission"])
def test_all_unreadable_is_failure(store, monkeypatch, reader, failure):
    path = store / filename(reader)
    if failure == "rows":
        write(store, path.name, ["{broken", "[]"])
    elif failure == "encoding":
        path.write_bytes(b"\xff")
    else:
        write(store, path.name, [row()])
        original = Path.read_text
        def deny(self, *args, **kwargs):
            if self == path:
                raise PermissionError("denied")
            return original(self, *args, **kwargs)
        monkeypatch.setattr(Path, "read_text", deny)
    with pytest.raises(ToolError) as caught:
        invoke(reader)
    assert caught.value.detail["records_read"] == 0
    assert caught.value.detail["rows_unreadable" if failure == "rows" else "files_unreadable"] == (2 if failure == "rows" else 1)


@pytest.mark.parametrize("reader", ["archive", "id", "network", "graph_memory"])
def test_one_unreadable_file_is_partial(store, reader):
    write(store, filename(reader), [row()])
    bad = "archive_zz.jsonl" if reader == "archive" else "test.jsonl"
    (store / bad).write_bytes(b"\xff")
    result = invoke(reader)
    assert isinstance(result, ToolResult)
    assert result.detail["files_unreadable"] == 1
    assert result.detail["records_read"] == 1


@pytest.mark.parametrize("reader", ["id", "network", "graph_memory"])
def test_unregistered_file_is_intentionally_skipped(store, reader):
    (store / "not_registered.jsonl").write_bytes(b"\xff")
    assert isinstance(invoke(reader), str)


def test_limit_does_not_count_unvisited_rows(store):
    write(store, "archive_test.jsonl", [row()] * 1000 + ["{broken"])
    assert isinstance(invoke("archive"), str)


@pytest.mark.parametrize("handler,args,message", [
    (memory_tool._search_memory, {}, "エラー: queryまたはidを指定してください"),
    (memory_tool._tool_search_memory, {}, "エラー: queryを指定してください"),
    (memory_tool._tool_memory_store, {}, "エラー: contentを指定してください"),
    (memory_tool._tool_memory_update, {}, "エラー: memory_idを指定してください"),
    (memory_tool._tool_memory_forget, {}, "エラー: memory_idを指定してください"),
])
def test_missing_arguments_keep_message(store, handler, args, message):
    with pytest.raises(ToolError) as caught:
        handler(args)
    assert str(caught.value) == message


@pytest.mark.parametrize("handler,args", [
    (memory_graph_tool._memory_graph, {"view": "bad"}),
    (world_fact_view_tool._world_fact_view, {"mode": "bad"}),
    (world_fact_view_tool._world_fact_view, {"mode": "by_tool"}),
    (world_fact_view_tool._world_fact_view, {"mode": "by_channel"}),
])
def test_invalid_modes_raise_with_original_json(store, handler, args):
    with pytest.raises(ToolError) as caught:
        handler(args)
    assert "error" in json.loads(str(caught.value))


def test_toolerror_not_swallowed_by_archive_or_graph_read(store, monkeypatch):
    sentinel = ToolError("sentinel")
    write(store, "archive_test.jsonl", [row()])
    monkeypatch.setattr(Path, "read_text", Mock(side_effect=sentinel))
    for reader in ("archive", "graph_memory"):
        if reader == "graph_memory":
            write(store, "_untagged.jsonl", [row()])
        with pytest.raises(ToolError) as caught:
            invoke(reader)
        assert caught.value is sentinel


def test_saved_memory_survives_link_failure(store, monkeypatch):
    monkeypatch.setattr(memory_links, "generate_links_for", Mock(side_effect=ToolError("link failed")))
    result = memory_tool._tool_memory_store({"content": "エラー"})
    assert isinstance(result, ToolResult)
    saved = json.loads((store / "_untagged.jsonl").read_text(encoding="utf-8"))
    assert saved["content"] == "エラー"
    assert result.message == f"記憶保存完了: [_untagged] エラー (id={saved['id']})"
    assert result.detail == {"saved": True, "memory_id": saved["id"], "post_save_failures": [{"phase": "links", "exception_type": "ToolError", "message": "link failed"}]}


def test_save_write_failure_is_failure(store, monkeypatch):
    monkeypatch.setattr(memory_tool, "memory_store", Mock(side_effect=PermissionError("denied")))
    with pytest.raises(ToolError, match="denied"):
        memory_tool._tool_memory_store({"content": "value"})
    assert not (store / "_untagged.jsonl").exists()


@pytest.mark.parametrize("response,failed", [("bad json", True), ('{"link_type":"none","confidence":0}', False)])
def test_link_parse_failure_separate_from_no_relation(store, response, failed):
    write(store, "_untagged.jsonl", [{"id": "old", "content": "old"}])
    failures = []
    entry = memory.memory_store(content="new", _auto_metadata=False, _state={},
        _reconcile_embed_fn=lambda texts: [[1]] * len(texts),
        _reconcile_cosine_fn=lambda a, b: 1,
        _reconcile_llm_fn=lambda *a, **kw: response,
        _post_save_failures=failures)
    assert entry["content"] == "new"
    assert bool(failures) is failed
    if failed:
        assert failures[0]["phase"] == "link_judge"
    assert len(memory.list_records("_untagged")) == 2


def test_link_llm_exception_is_recorded(store):
    failures = []
    memory_links._llm_judge_link(row(), row(), llm_call_fn=Mock(side_effect=RuntimeError("offline")), failures=failures)
    assert failures == [{"phase": "link_judge", "exception_type": "RuntimeError", "message": "offline"}]


def test_link_embedding_exception_is_recorded(store):
    write(store, "_untagged.jsonl", [{"id": "old", "content": "old"}])
    failures = []
    result = memory_links.generate_links_for(row(), embed_fn=Mock(side_effect=RuntimeError("offline")), cosine_fn=lambda a,b: 1, failures=failures)
    assert result == []
    assert failures == [{"phase": "link_embedding", "exception_type": "RuntimeError", "message": "offline"}]


@pytest.mark.parametrize("handler", [memory_tool._tool_memory_update, memory_tool._tool_memory_forget])
def test_mutation_partial_and_all_unreadable(store, handler):
    write(store, "_untagged.jsonl", ["{broken", row()])
    result = handler({"memory_id": "mem_target", "content": "updated"})
    assert isinstance(result, ToolResult)
    assert result.detail["rows_unreadable"] == 1
    saved = (store / "_untagged.jsonl").read_text(encoding="utf-8")
    assert "{broken" in saved
    if handler is memory_tool._tool_memory_update:
        assert json.loads(saved.splitlines()[1])["content"] == "updated"
    else:
        assert "mem_target" not in saved
    write(store, "_untagged.jsonl", ["{broken"])
    with pytest.raises(ToolError) as caught:
        handler({"memory_id": "missing"})
    assert caught.value.detail["records_read"] == 0


@pytest.mark.parametrize("handler", [memory_tool._tool_memory_update, memory_tool._tool_memory_forget])
def test_mutation_write_failure(store, monkeypatch, handler):
    write(store, "_untagged.jsonl", [row()])
    monkeypatch.setattr(Path, "write_text", Mock(side_effect=PermissionError("denied")))
    with pytest.raises(ToolError, match="denied"):
        handler({"memory_id": "mem_target", "content": "new"})


def test_forget_does_not_delete_content_match(store):
    survivor = {"id": "mem_other", "content": "mem_target"}
    write(store, "_untagged.jsonl", [survivor, row(), ""])
    memory_tool._tool_memory_forget({"id": "mem_target"})
    assert memory.list_records("_untagged") == [survivor]


@pytest.mark.parametrize("reader", ["archive", "world", "graph_memory"])
def test_partial_and_full_failure_reach_runtime_hooks(store, monkeypatch, reader):
    from test_failure_contract_foundation import runtime
    from core.runtime.hooks import HookRunner, HookRunResult
    monkeypatch.setattr("core.auth._load_secrets", lambda: {})
    success = Mock(return_value=HookRunResult.allow())
    failure = Mock(return_value=HookRunResult.allow())
    hooks = HookRunner()
    hooks.register_post(success)
    hooks.register_failure(failure)
    rt = runtime(lambda args: invoke(reader), hooks=hooks)
    write(store, filename(reader), [row(), "{broken"])
    result = rt._execute_tool_use("one", "probe", {})
    assert not result.is_error and result.error_kind is None
    assert result.detail["rows_unreadable"] == 1
    assert success.call_count == 1 and failure.call_count == 0
    write(store, filename(reader), ["{broken"])
    result = rt._execute_tool_use("two", "probe", {})
    assert result.is_error and result.error_kind == "tool_failure"
    assert result.detail["records_read"] == 0
    assert success.call_count == 1 and failure.call_count == 1


@pytest.mark.parametrize("handler", [memory_tool._tool_memory_update, memory_tool._tool_memory_forget])
def test_missing_mutation_id_is_annotated_failure(store, handler):
    write(store, "_untagged.jsonl", [row()])
    before = (store / "_untagged.jsonl").read_bytes()
    with pytest.raises(ToolError) as caught:
        handler({"id": "missing", "content": "new"})
    assert str(caught.value) == "エラー: missing が見つかりません"
    assert caught.value.detail["memory_id"] == "missing"
    assert caught.value.detail["networks_searched"] == 2
    assert caught.value.detail["records_read"] == 1
    assert (store / "_untagged.jsonl").read_bytes() == before
