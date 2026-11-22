"""test_world_state_view.py — V07 Phase 1 commit 1 識別力 test (CLAUDE.md §5/§6 厳守)

V07_PROMPT_PARADIGM_SHIFT_PLAN §6-2 / §6-2-A literal:
- fixture 4 種 (cycle 1 空 / cycle 50 正常 / orphan / 0 byte)
- 想定誤実装 4 fail パターン (orphan 無視 / cap 無視 / limit 無視 / 世界視点違反)
- bootstrap empty literal assert (Codex audit P3-03 fix)
- semantic loss 検出 assert (Codex audit P2-06 fix)
"""
import json

import pytest

from core import world_state_view


# ============================================================
# fixture 4 種 (PLAN §6-2 commit 1 識別力 test)
# ============================================================

@pytest.fixture
def state_cycle1_empty():
    """cycle 1、raw_events 空、bootstrap 期 fixture。"""
    return {"raw_events": []}


@pytest.fixture
def state_cycle50_normal():
    """cycle 50、raw_events 50 件、orphan なし、正常運行 fixture。"""
    return {
        "raw_events": [
            {"tool": f"tool_{i}", "args": {"a": i}, "result": f"result_{i}"}
            for i in range(50)
        ]
    }


@pytest.fixture
def state_with_long_result():
    """result が 200 字超 fixture (誤実装 b: cap 無視 検出用)。"""
    return {
        "raw_events": [
            {"tool": "read_file", "args": {"path": "a"}, "result": "X" * 300}
        ]
    }


@pytest.fixture
def state_with_subjective_fields():
    """intent / e1-e4 評価が混入してる fixture (誤実装 d: 世界視点違反 検出用)。"""
    return {
        "raw_events": [
            {
                "tool": "write_file",
                "args": {"path": "x"},
                "result": "ok",
                "intent": "test 意図",  # subjective field、混入禁止
                "e1": 0.8,
                "e2": 70,
                "e3": 0.5,
                "e4": 0.3,
            }
        ]
    }


@pytest.fixture
def state_10_events():
    """semantic loss 検出 fixture: raw_events 10 件 (5 件 limit より多い、P2-06 fix)。"""
    return {
        "raw_events": [
            {"tool": f"tool_{i}", "args": {"i": i}, "result": f"r_{i}"}
            for i in range(10)
        ]
    }


# ============================================================
# bootstrap empty literal verify (Codex audit P3-03 fix)
# ============================================================

def test_bootstrap_empty_literal(state_cycle1_empty):
    """cycle 1 で <world_state> body に explicit empty literal が表示される。

    誤実装 (bootstrap 期に section ごと省略 / 空 dict で collapse) で fail する。
    """
    body = world_state_view.build_world_state(state_cycle1_empty)
    # XML tag pair 成立
    assert body.startswith("<world_state>"), "open tag missing"
    assert body.endswith("</world_state>"), "close tag missing"
    # 主要 section 名が body 内に存在 (空でも block 構造維持)
    assert "file_snapshot:" in body, "file_snapshot section missing"
    assert "recent_events:" in body, "recent_events section missing"
    # 空 list でも explicit literal が表示される (bootstrap engine)
    assert "(no recent events)" in body, "empty recent_events not rendered explicitly"


# ============================================================
# 誤実装 a: orphan flag 無視 検出 (PLAN §6-2 commit 1)
# ============================================================

def test_orphan_detection(monkeypatch, tmp_path):
    """誤実装 a: orphan を常に [] で返す実装で fail。

    fixture: 一時 profile root 直下に正規 state.json、別 dir に orphan state.json 配置。
    expected: orphan list に orphan が含まれる。
    """
    canonical = tmp_path / "state.json"
    canonical.write_text("{}", encoding="utf-8")
    orphan_dir = tmp_path / "profile"
    orphan_dir.mkdir()
    orphan_path = orphan_dir / "state.json"
    orphan_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(world_state_view, "PROFILE_ROOT", tmp_path)
    monkeypatch.setattr(world_state_view, "STATE_PATH", canonical)
    monkeypatch.setattr(
        world_state_view, "RAW_EVENTS_PATH", tmp_path / "memory" / "raw_events.jsonl"
    )

    snapshot = world_state_view._file_snapshot()
    # 誤実装 a (常に []) で fail
    assert snapshot["orphan"] != [], "orphan detection silently dropped"
    # 具体的 path verify (content assertion、CLAUDE.md §5 強化)
    assert any("profile" in p and "state.json" in p for p in snapshot["orphan"]), (
        f"expected orphan profile/state.json, got {snapshot['orphan']}"
    )


def test_orphan_none_when_canonical_only(monkeypatch, tmp_path):
    """正規 state.json のみで orphan 列なし (識別力強化: 全一致 fixture)。"""
    canonical = tmp_path / "state.json"
    canonical.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(world_state_view, "PROFILE_ROOT", tmp_path)
    monkeypatch.setattr(world_state_view, "STATE_PATH", canonical)
    monkeypatch.setattr(
        world_state_view, "RAW_EVENTS_PATH", tmp_path / "memory" / "raw_events.jsonl"
    )

    snapshot = world_state_view._file_snapshot()
    # 誤実装 (正規も orphan 扱い) で fail
    assert snapshot["orphan"] == [], (
        f"canonical state.json wrongly flagged as orphan: {snapshot['orphan']}"
    )


# ============================================================
# 誤実装 b: result cap 200 字無視 検出 (PLAN §6-2 commit 1)
# ============================================================

def test_result_cap_200(state_with_long_result):
    """誤実装 b: result cap を守らず full result 返す実装で fail。

    300 字 result fixture で 200 字超なら literal "...[300字]" marker 必須。
    """
    events = world_state_view._recent_events(state_with_long_result["raw_events"])
    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    # 誤実装 b (300 字そのまま) で fail: 200 + "...[300字]" marker ≈ 209 字
    assert len(events[0]["result"]) <= 220, (
        f"result cap not enforced, len={len(events[0]['result'])}"
    )
    # marker 付与 verify (CLAUDE.md §5 強化: count + content)
    assert events[0]["result"].endswith("[300字]"), (
        f"truncation marker missing, result={events[0]['result'][-20:]}"
    )


def test_result_under_cap_no_marker():
    """200 字未満 result は marker なし (識別力強化: 全一致 fixture)。"""
    raw = [{"tool": "t", "args": None, "result": "short"}]
    events = world_state_view._recent_events(raw)
    # 誤実装 (常に marker 付与) で fail
    assert events[0]["result"] == "short", (
        f"unexpected truncation marker: {events[0]['result']}"
    )


# ============================================================
# 誤実装 c: limit 守らず全件返す 検出 (PLAN §6-2 commit 1)
# ============================================================

def test_limit_5(state_cycle50_normal):
    """誤実装 c: limit を無視し全件返す実装で fail。"""
    events = world_state_view._recent_events(state_cycle50_normal["raw_events"], limit=5)
    # 誤実装 c (50 件) で fail
    assert len(events) == 5, f"expected 5, got {len(events)}"
    # 最新 5 件であることを verify (時系列降順、CLAUDE.md §5 content 強化)
    assert events[0]["tool"] == "tool_49", f"head should be newest, got {events[0]['tool']}"
    assert events[4]["tool"] == "tool_45", f"tail should be 5th newest, got {events[4]['tool']}"


def test_limit_under_total():
    """raw_events 件数 < limit でも error なく動く (識別力強化: 部分一致 fixture)。"""
    raw = [{"tool": "t1", "args": None, "result": "r1"},
           {"tool": "t2", "args": None, "result": "r2"}]
    events = world_state_view._recent_events(raw, limit=5)
    assert len(events) == 2, f"expected 2, got {len(events)}"
    assert events[0]["tool"] == "t2"


# ============================================================
# 誤実装 d: intent / e1-e4 混入 検出 (PLAN §6-2 commit 1)
# ============================================================

def test_world_view_only(state_with_subjective_fields):
    """誤実装 d: intent / e1-e4 を含めて返す実装で fail (世界視点違反検出、5 論点 ①)。"""
    events = world_state_view._recent_events(state_with_subjective_fields["raw_events"])
    assert len(events) == 1
    e = events[0]
    # tool/args/result 3 field のみ (CLAUDE.md §5 content assertion 強化)
    assert "tool" in e
    assert "args" in e
    assert "result" in e
    # 誤実装 d (subjective field 混入) で fail
    assert "intent" not in e, "subjective field 'intent' leaked"
    assert "e1" not in e, "subjective field 'e1' leaked"
    assert "e2" not in e, "subjective field 'e2' leaked"
    assert "e3" not in e, "subjective field 'e3' leaked"
    assert "e4" not in e, "subjective field 'e4' leaked"
    # field 数も exact check
    assert set(e.keys()) == {"tool", "args", "result"}, f"unexpected fields: {set(e.keys())}"


# ============================================================
# semantic loss 検出 (Codex audit P2-06 fix、PLAN §6-2-A)
# ============================================================

def test_semantic_loss_5_of_10_in_world_state(state_10_events):
    """raw_events 10 件 fixture で <world_state.recent_events> は最新 5 件のみ。

    6 件目以降の id/tool は silent loss、world_fact_view tool 経由 affordance で取得する設計。
    本 test は 5 件 limit 性のみ verify (silent loss が attractor 化するかは Phase 4 smoke で観察)。
    """
    body = world_state_view.build_world_state(state_10_events)
    # 最新 5 件 (tool_5 〜 tool_9) は body に含まれる
    for i in [5, 6, 7, 8, 9]:
        assert f"tool_{i}" in body, f"newest event tool_{i} missing from body"
    # 古い 5 件 (tool_0 〜 tool_4) は body に含まれない (silent loss、affordance 経由)
    for i in [0, 1, 2, 3, 4]:
        assert f"tool_{i}" not in body, (
            f"old event tool_{i} leaked into body (should be silent-lost)"
        )


# ============================================================
# file_snapshot 構造 verify
# ============================================================

def test_file_snapshot_structure(monkeypatch, tmp_path):
    """file_snapshot dict が state_json + raw_events_jsonl + orphan の 3 keys を返す。"""
    canonical = tmp_path / "state.json"
    canonical.write_text('{"x": 1}', encoding="utf-8")
    raw_events_jsonl = tmp_path / "memory" / "raw_events.jsonl"
    raw_events_jsonl.parent.mkdir()
    raw_events_jsonl.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(world_state_view, "PROFILE_ROOT", tmp_path)
    monkeypatch.setattr(world_state_view, "STATE_PATH", canonical)
    monkeypatch.setattr(world_state_view, "RAW_EVENTS_PATH", raw_events_jsonl)

    snapshot = world_state_view._file_snapshot()
    assert set(snapshot.keys()) == {"state_json", "raw_events_jsonl", "orphan"}
    assert snapshot["state_json"]["size"] > 0
    assert snapshot["state_json"]["mtime"] is not None
    assert snapshot["raw_events_jsonl"]["size"] > 0
    # 誤実装 (size 文字列で返す等) で fail
    assert isinstance(snapshot["state_json"]["size"], int)


def test_file_snapshot_missing_files(monkeypatch, tmp_path):
    """state.json / raw_events.jsonl 不在で size=0 / mtime=None (完全不一致 fixture)。"""
    monkeypatch.setattr(world_state_view, "PROFILE_ROOT", tmp_path)
    monkeypatch.setattr(world_state_view, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(
        world_state_view, "RAW_EVENTS_PATH", tmp_path / "memory" / "raw_events.jsonl"
    )

    snapshot = world_state_view._file_snapshot()
    # 誤実装 (file 不在で error / size=None 等) で fail
    assert snapshot["state_json"]["size"] == 0
    assert snapshot["state_json"]["mtime"] is None
    assert snapshot["raw_events_jsonl"]["size"] == 0


# ============================================================
# build_world_state 全体 integration
# ============================================================

def test_build_world_state_structure(state_cycle50_normal):
    """build_world_state は <world_state> XML block + body を返す。"""
    body = world_state_view.build_world_state(state_cycle50_normal)
    assert body.startswith("<world_state>")
    assert body.endswith("</world_state>")
    assert "file_snapshot:" in body
    assert "recent_events:" in body
    assert "state.json:" in body
    assert "raw_events.jsonl:" in body
    # 最新 5 件の tool 名が body に出る
    assert "tool_49" in body
    assert "tool_45" in body


def test_build_world_state_zero_byte_raw_events(monkeypatch, tmp_path):
    """raw_events.jsonl が 0 byte でも crash しない (識別力 fixture: edge case)。"""
    canonical = tmp_path / "state.json"
    canonical.write_text("{}", encoding="utf-8")
    raw_events_jsonl = tmp_path / "memory" / "raw_events.jsonl"
    raw_events_jsonl.parent.mkdir()
    raw_events_jsonl.write_text("", encoding="utf-8")

    monkeypatch.setattr(world_state_view, "PROFILE_ROOT", tmp_path)
    monkeypatch.setattr(world_state_view, "STATE_PATH", canonical)
    monkeypatch.setattr(world_state_view, "RAW_EVENTS_PATH", raw_events_jsonl)

    body = world_state_view.build_world_state({"raw_events": []})
    assert "<world_state>" in body
    # 0 byte raw_events.jsonl も size=0 で表示 (file 存在は表示)
    assert "raw_events.jsonl: size=0B" in body
    assert "(no recent events)" in body
