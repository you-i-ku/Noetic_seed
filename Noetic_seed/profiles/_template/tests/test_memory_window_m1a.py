"""M1a: 空抜粋・expect欠落・slice先行・入力書換えを区別する。"""
import copy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import config, memory, subjective_view


def _retrieve(monkeypatch, entries, networks=None, externals=None):
    state = {"subjective_entries": entries}
    networks, externals = networks or [], externals or []
    before = copy.deepcopy((state, networks, externals))
    search = Mock(return_value=networks)
    monkeypatch.setattr(memory, "memory_network_search", search)
    monkeypatch.setattr(memory, "_recent_externals_from_archive", lambda **kw: externals)
    result = memory.get_relevant_memories(state)
    assert (state, networks, externals) == before
    return result, search


def test_filter_before_last_three_and_intent_priority(monkeypatch):
    entries = [
        {"id": "old", "intent": "対象外"},
        {"id": "both", "intent": "理由", "expect": "別の予想", "content": "誤った抜粋元"},
        {"id": "expect", "expect": "予想だけ"},
        {"id": "intent", "intent": "最新の理由"},
        {"id": "blank1"}, {"id": "blank2", "intent": "", "expect": ""},
        {"id": "blank3", "intent": None, "expect": None},
    ]
    result, _ = _retrieve(monkeypatch, entries)
    assert [(m["id"], m["kind"], m["excerpt_80chars"]) for m in result] == [
        ("both", "subjective", "理由"), ("expect", "subjective", "予想だけ"),
        ("intent", "subjective", "最新の理由"),
    ]


@pytest.mark.parametrize("field", ["intent", "expect", "content"])
@pytest.mark.parametrize("length", [79, 80, 81])
def test_excerpt_boundary_and_original_fields(monkeypatch, field, length):
    text = "あ" * length
    entry = {"id": "target", field: text}
    entries = [entry] if field != "content" else [{"id": "query", "intent": "検索"}]
    result, search = _retrieve(monkeypatch, entries, [entry] if field == "content" else [])
    target = next(m for m in result if m["id"] == "target")
    assert target[field] == text
    assert target["excerpt_80chars"] == text[:80] + ("…" if length > 80 else "")
    if field == "expect":
        search.assert_not_called()  # intentなしの早期returnでも予想は表示できる。


@pytest.mark.parametrize("entries", [[], [{"id": "blank", "intent": "", "expect": ""}]])
def test_no_subjective_text(monkeypatch, entries):
    result, search = _retrieve(monkeypatch, entries)
    assert result == []
    search.assert_not_called()


def test_external_and_memory_content_stays_the_excerpt_source(monkeypatch):
    external = {"id": "external", "content": "外部入力", "intent": "別の理由"}
    network = {"id": "memory", "content": "本" * 81, "expect": "別の予想"}
    result, _ = _retrieve(monkeypatch, [], [network], [external])
    assert [(m["id"], m["excerpt_80chars"]) for m in result] == [
        ("external", "外部入力"), ("memory", "本" * 80 + "…")]


def test_subjective_id_required_and_duplicate_not_added(monkeypatch):
    result, _ = _retrieve(monkeypatch, [
        {"intent": "IDなし"}, {"id": "same", "intent": "重複"},
        {"id": "new", "intent": "採用"}], [{"id": "same", "content": "本文"}])
    assert [(m["id"], m["excerpt_80chars"]) for m in result] == [
        ("same", "本文"), ("new", "採用")]


def test_related_memory_displays_subjective_excerpts_without_empty_slots(monkeypatch):
    """実取得→表示変換を通し、contentへの逆戻り・expect欠落・空枠を検出する。"""
    state = {"subjective_entries": [
        {"id": "both", "intent": "表示する理由", "expect": "優先しない予想"},
        {"id": "expect_only", "expect": "表示する予想"},
        {"id": "blank", "intent": "", "expect": ""},
    ]}
    monkeypatch.setattr(config, "llm_cfg", {"retrieval": {"use_links": False}})
    monkeypatch.setattr(memory, "memory_network_search", lambda *a, **kw: [])
    monkeypatch.setattr(memory, "_recent_externals_from_archive", lambda **kw: [])
    displayed = subjective_view._related_memory(state)
    assert [(row["id"], row["excerpt_80chars"]) for row in displayed] == [
        ("both", "表示する理由"), ("expect_only", "表示する予想")]
    assert all(row["excerpt_80chars"] for row in displayed)
