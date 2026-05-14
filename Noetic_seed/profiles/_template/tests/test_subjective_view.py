"""test_subjective_view.py — V07 Phase 1 commit 2 識別力 test (CLAUDE.md §5/§6 厳守)

V07_PROMPT_PARADIGM_SHIFT_PLAN §6-2 / §6-2-A literal:
- fixture 4 種 (cycle 1 空 / cycle 50 育 / related_memory 空 / related_memory overflow)
- 想定誤実装 5 fail パターン (ego 抜け / cap 抜け / limit 超過 / self 欠損 / LLM 再帰)
- bootstrap empty literal assert (Codex audit P3-03 fix)
- ★ LLM 再帰呼出阻止 verify (Codex audit P2-03 fix の identification literal)
"""
import pytest

from core import subjective_view


# ============================================================
# fixture 4 種 + 補助 fixture
# ============================================================

@pytest.fixture
def state_cycle1_empty_graph():
    """cycle 1、memory_graph 空、state.self に name のみ (seed.txt 由来)。"""
    return {
        "self": {"name": "iku"},
        "cycle_id": 1,
        "phase6_metrics": {},
        "subjective_entries": [],
    }


@pytest.fixture
def state_cycle50_full_graph():
    """cycle 50、graph 育 (cluster_mi / frontier 等)、state.self 多 field。"""
    return {
        "self": {
            "name": "iku",
            "context_integrity_check": "verified",
            "status": "active",
        },
        "cycle_id": 50,
        "phase6_metrics": {
            "cluster_mi": 0.333,
            "cluster_inter_ratio": 0.61,
            "last_cluster_link_pairs": 226,
            "last_reflect_cycle": 47,
        },
        "subjective_entries": [
            {"intent": "test intent"} for _ in range(5)
        ],
    }


@pytest.fixture
def memory_5_items():
    """get_relevant_memories 戻り値 mock 5 件 (正常)。"""
    return [
        {"id": f"m_{i}", "content": f"content_{i} text", "cycle_id": 45 - i}
        for i in range(5)
    ]


@pytest.fixture
def memory_10_items():
    """overflow: 10 件 (limit 5 超過、識別力 c 用)。"""
    return [
        {"id": f"m_{i}", "content": f"content_{i}", "cycle_id": 50 - i}
        for i in range(10)
    ]


@pytest.fixture
def memory_long_content():
    """100 字超 content 1 件 (識別力 b: cap 抜け marker 検出用)。"""
    return [{
        "id": "m_long",
        "content": "X" * 100,
        "cycle_id": 40,
    }]


# ============================================================
# bootstrap empty literal verify (Codex audit P3-03 fix)
# ============================================================

def test_bootstrap_empty_literal(state_cycle1_empty_graph, monkeypatch):
    """cycle 1 で <subjective_state> body に explicit empty literal が表示される。"""
    monkeypatch.setattr(subjective_view, "_related_memory", lambda s, **k: [])

    body = subjective_view.build_subjective_state(state_cycle1_empty_graph)
    assert body.startswith("<subjective_state>"), "open tag missing"
    assert body.endswith("</subjective_state>"), "close tag missing"
    # 主要 section が body 内に存在 (空でも block 構造維持)
    assert "current_self:" in body
    assert "graph_summary:" in body
    assert "related_memory:" in body
    # 空 list でも explicit literal が表示される (bootstrap engine)
    assert "(no related memories)" in body
    # cycle 1 でも seed.txt 由来 name は表示
    assert "iku" in body


# ============================================================
# ★ LLM 再帰呼出阻止 verify (Codex audit P2-03 fix、最重要)
# ============================================================

def test_graph_summary_no_llm_recursion(state_cycle50_full_graph, monkeypatch):
    """_graph_summary は estimate_clusters / call_llm を絶対呼ばない (P2-03 fix verify)。

    cheap metadata only 設計の literal 識別力 test。誤実装 e (memory_graph_tool 全パス流用) で fail。
    """
    # call_llm が呼ばれたら即 fail で trigger detection
    call_llm_invoked = []
    estimate_clusters_invoked = []

    def fake_call_llm(*args, **kwargs):
        call_llm_invoked.append((args, kwargs))
        raise AssertionError("call_llm invoked from _graph_summary (P2-03 fix violation)")

    def fake_estimate_clusters(*args, **kwargs):
        estimate_clusters_invoked.append((args, kwargs))
        raise AssertionError("estimate_clusters invoked from _graph_summary (P2-03 fix violation)")

    monkeypatch.setattr("core.llm.call_llm", fake_call_llm)
    try:
        monkeypatch.setattr(
            "core.cluster_estimation.estimate_clusters", fake_estimate_clusters
        )
    except (AttributeError, ImportError):
        pass  # cluster_estimation が optional な環境では skip

    # 実行 (assert 内で AssertionError raise なら test fail)
    summary = subjective_view._graph_summary(state_cycle50_full_graph)

    # post-check: invocation 記録ゼロ
    assert call_llm_invoked == [], (
        f"call_llm was invoked {len(call_llm_invoked)} times (P2-03 fix violation)"
    )
    assert estimate_clusters_invoked == [], (
        f"estimate_clusters was invoked {len(estimate_clusters_invoked)} times (P2-03 fix violation)"
    )

    # summary structure verify
    assert "ego" in summary
    assert "global" in summary
    assert "frontier_node_count" in summary["global"]
    assert "graph_maturity" in summary["global"]


# ============================================================
# 誤実装 a: _graph_summary ego mode 抜け 検出
# ============================================================

def test_graph_summary_both_modes(state_cycle50_full_graph):
    """誤実装 a: ego mode 抜けで global のみ返す実装で fail。"""
    summary = subjective_view._graph_summary(state_cycle50_full_graph)
    # V7-A1 確定 (both): ego + global 両方必須
    assert "ego" in summary, "ego mode missing (V7-A1 violation)"
    assert "global" in summary, "global mode missing"
    # ego content
    assert "self_edges" in summary["ego"]
    assert "memory_1hop_count" in summary["ego"]
    # global content (cheap metadata only fields)
    assert "frontier_node_count" in summary["global"]
    assert "graph_maturity" in summary["global"]
    assert "cluster_mi" in summary["global"]


def test_global_phase6_metrics_propagation(state_cycle50_full_graph):
    """phase6_metrics の cluster_mi が global summary に伝播 (CLAUDE.md §5 content 強化)。"""
    summary = subjective_view._graph_summary(state_cycle50_full_graph)
    # 誤実装 (phase6_metrics 無視) で fail
    assert summary["global"]["cluster_mi"] == 0.333


# ============================================================
# 誤実装 b: _related_memory 80 字 cap 抜け marker 検出
# ============================================================

def test_excerpt_80_long_content(state_cycle50_full_graph, monkeypatch, memory_long_content):
    """誤実装 b: 80 字超で marker "…" 抜けで fail。

    100 字 fixture (80 字超) で marker 必須。
    """
    monkeypatch.setattr(
        "core.subjective_view.get_relevant_memories",
        lambda *a, **k: memory_long_content,
        raising=False,
    )
    # get_relevant_memories は module 内 import なので直接 patch
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: memory_long_content)

    result = subjective_view._related_memory(state_cycle50_full_graph)
    assert len(result) == 1
    # 誤実装 b (full 100 字) で fail
    excerpt = result[0]["excerpt_80chars"]
    assert len(excerpt) <= 82, f"excerpt cap 80 not enforced, got {len(excerpt)}"
    assert excerpt.endswith("…"), f"truncation marker missing, excerpt ends: {excerpt[-5:]}"


def test_excerpt_80_short_content(state_cycle50_full_graph, monkeypatch):
    """80 字未満 content は marker なし (識別力 強化)。"""
    short_memories = [{"id": "m_s", "content": "short", "cycle_id": 40}]
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: short_memories)

    result = subjective_view._related_memory(state_cycle50_full_graph)
    assert len(result) == 1
    # 誤実装 (常に marker 付与) で fail
    assert result[0]["excerpt_80chars"] == "short"
    assert not result[0]["excerpt_80chars"].endswith("…")


def test_excerpt_80_exact_boundary(state_cycle50_full_graph, monkeypatch):
    """80 字ちょうどで marker なし (off-by-one 検出)。"""
    exact_memories = [{"id": "m_e", "content": "X" * 80, "cycle_id": 40}]
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: exact_memories)

    result = subjective_view._related_memory(state_cycle50_full_graph)
    # 誤実装 (off-by-one で 80 字に marker) で fail
    assert not result[0]["excerpt_80chars"].endswith("…"), "marker wrongly attached at exact 80"
    assert len(result[0]["excerpt_80chars"]) == 80


# ============================================================
# 誤実装 c: _related_memory limit 超過受容 検出
# ============================================================

def test_related_memory_limit_5(state_cycle50_full_graph, monkeypatch, memory_10_items):
    """誤実装 c: 10 件 fixture で limit を無視し全件返す実装で fail。"""
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: memory_10_items)

    result = subjective_view._related_memory(state_cycle50_full_graph)
    # 誤実装 c (10 件) で fail
    assert len(result) == 5, f"limit not enforced, got {len(result)}"
    # 最初の 5 件であることを verify (content 強化)
    assert result[0]["id"] == "m_0"
    assert result[4]["id"] == "m_4"


# ============================================================
# 誤実装 d: _current_self で state.self 欠損 検出
# ============================================================

def test_current_self_with_name(state_cycle1_empty_graph):
    """誤実装 d: state.self.name 抜けで空 dict 返すで fail。"""
    cs = subjective_view._current_self(state_cycle1_empty_graph)
    # JSON 文字列 内に "iku" 含む
    assert "iku" in cs
    assert "name" in cs


def test_current_self_none():
    """state.self=None で "(なし)" literal (graceful、prompt.py:234 同パターン)。"""
    cs = subjective_view._current_self({"self": None})
    # 誤実装 (None で crash / 空文字) で fail
    assert cs == "(なし)"


def test_current_self_empty_dict():
    """state.self={} で "(なし)" literal (falsy 整合)。"""
    cs = subjective_view._current_self({"self": {}})
    assert cs == "(なし)"


# ============================================================
# bootstrap 期 graph_summary の空 graph graceful skip
# ============================================================

def test_graph_summary_bootstrap_zero(state_cycle1_empty_graph):
    """cycle 1 (memory_graph 空) で全 metadata 0 / None graceful (P3-03 fix 補強)。"""
    summary = subjective_view._graph_summary(state_cycle1_empty_graph)
    # 誤実装 (空 graph で crash) で fail
    assert summary["ego"]["self_edges"] == 0
    assert summary["ego"]["memory_1hop_count"] == 0
    assert summary["global"]["frontier_node_count"] == 0
    # cluster_mi は phase6_metrics 空 → None
    assert summary["global"]["cluster_mi"] is None


# ============================================================
# build_subjective_state 全体 integration
# ============================================================

def test_build_subjective_state_structure(state_cycle50_full_graph, monkeypatch, memory_5_items):
    """build_subjective_state は <subjective_state> XML block + 3 section を返す。"""
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: memory_5_items)

    body = subjective_view.build_subjective_state(state_cycle50_full_graph)
    assert body.startswith("<subjective_state>")
    assert body.endswith("</subjective_state>")
    assert "current_self:" in body
    assert "graph_summary:" in body
    assert "related_memory:" in body
    # cluster_mi=0.333 が body に出る (phase6_metrics propagation)
    assert "0.333" in body
    # memory id m_0 が body に出る
    assert "m_0" in body


def test_build_subjective_state_no_memories(state_cycle1_empty_graph, monkeypatch):
    """memory 空でも crash しない、explicit empty literal (P3-03 fix)。"""
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: [])

    body = subjective_view.build_subjective_state(state_cycle1_empty_graph)
    assert "<subjective_state>" in body
    assert "(no related memories)" in body
