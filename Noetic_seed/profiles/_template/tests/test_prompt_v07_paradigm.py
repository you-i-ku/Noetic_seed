"""test_prompt_v07_paradigm.py — V07 Phase 1 commit 3 + hotfix 識別力 test

build_prompt_propose (LLM①) + assemble_system_prompt (LLM②) の XML tag 化 +
subjective_state / world_state 相同並置検証。

PLAN §6-2 commit 3 literal + Codex audit 2026-05-14 hotfix:
- fixture 3 種 (cycle 50 normal / cycle 1 bootstrap / orphan)
- 誤実装 fail パターン: XML tag pair 不成立 / subjective 内 world 混入 / 順序違反 /
  bootstrap 全省略 / pending 欠落 (AUD-P1-01) / tool block 旧 heading (AUD-P1-01)
"""
import pytest

from core import prompt as prompt_module
from core import prompt_assembly


# ============================================================
# fixture
# ============================================================

@pytest.fixture
def state_normal():
    """cycle 50 正常 state。"""
    return {
        "self": {"name": "iku", "context_integrity_check": "ok"},
        "cycle_id": 50,
        "subjective_entries": [
            {"intent": "test 意図", "e1": 0.8, "e2": 70}
        ],
        "raw_events": [
            {"tool": "read_file", "args": {"p": "a"}, "result": "ok"}
        ],
        "pending": [],
        "phase6_metrics": {"cluster_mi": 0.5},
        "summaries": [],
    }


@pytest.fixture
def state_bootstrap():
    """cycle 1、空 state (bootstrap 期)。"""
    return {
        "self": {"name": "iku"},
        "cycle_id": 1,
        "subjective_entries": [],
        "raw_events": [],
        "pending": [],
        "phase6_metrics": {},
        "summaries": [],
    }


@pytest.fixture
def ctrl_minimal():
    """controller 最小 dict (tool_level / allowed_tools)。"""
    return {"tool_level": 2, "allowed_tools": set()}


@pytest.fixture
def tools_minimal():
    """tools_dict 最小。"""
    return {}


# ============================================================
# XML tag pair 成立 verify (誤実装 a: open/close 抜け 検出)
# ============================================================

def test_xml_tag_pairs_open_close(state_normal, ctrl_minimal, tools_minimal, monkeypatch):
    """誤実装 a: <subjective_state> open しかなく close なし、で fail。"""
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: [])

    body = prompt_module.build_prompt_propose(state_normal, ctrl_minimal, tools_minimal)
    # 主要 XML tag が pair で存在
    for tag in ("subjective_state", "world_state", "pending", "recent_history",
                "available_tools", "task"):
        assert f"<{tag}>" in body, f"open tag <{tag}> missing"
        assert f"</{tag}>" in body, f"close tag </{tag}> missing"


# ============================================================
# subjective 内 world 混入禁止 (誤実装 b: 重複 inject 検出)
# ============================================================

def test_subjective_state_no_world_leak(state_normal, ctrl_minimal, tools_minimal, monkeypatch):
    """誤実装 b: subjective_state body 内に file_snapshot / recent_events 混入で fail。"""
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: [])

    body = prompt_module.build_prompt_propose(state_normal, ctrl_minimal, tools_minimal)
    # subjective_state block の内容を抽出
    start = body.index("<subjective_state>")
    end = body.index("</subjective_state>")
    subj_body = body[start:end]
    # 誤実装 b (world fields が subj に混入) で fail
    assert "file_snapshot" not in subj_body, "world_state field leaked into subjective_state"
    # ただし recent_events は別 section (world_state) で出るので subj block には不在
    # subj body には current_self / graph_summary / related_memory のみ
    assert "current_self" in subj_body
    assert "graph_summary" in subj_body
    assert "related_memory" in subj_body


# ============================================================
# section 順序違反検出 (誤実装 c: subj が world より後 検出)
# ============================================================

def test_section_order(state_normal, ctrl_minimal, tools_minimal, monkeypatch):
    """誤実装 c: section 順序 (subj → world → pending → history → tools → task) 違反で fail。

    PLAN §3-4: 認知 ground (subj/world) を pending より前、task は recency bias で末尾。
    """
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: [])

    body = prompt_module.build_prompt_propose(state_normal, ctrl_minimal, tools_minimal)
    # 各 section の出現位置を取得
    positions = {tag: body.index(f"<{tag}>") for tag in (
        "subjective_state", "world_state", "pending", "recent_history",
        "available_tools", "task"
    )}
    # 期待順序
    expected_order = ["subjective_state", "world_state", "pending", "recent_history",
                      "available_tools", "task"]
    sorted_by_pos = sorted(positions, key=lambda k: positions[k])
    # 誤実装 c (順序違反) で fail
    assert sorted_by_pos == expected_order, (
        f"section order violation: expected {expected_order}, got {sorted_by_pos}"
    )


# ============================================================
# bootstrap 期に全 section 省略しない (誤実装 d 検出)
# ============================================================

def test_bootstrap_all_sections_present(state_bootstrap, ctrl_minimal, tools_minimal, monkeypatch):
    """誤実装 d: bootstrap 期 (空 state) で section ごと省略する実装で fail。

    P3-03 fix: empty literal を表示するのは bootstrap engine。
    """
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: [])

    body = prompt_module.build_prompt_propose(state_bootstrap, ctrl_minimal, tools_minimal)
    # 誤実装 d (bootstrap で省略) で fail: 全 section 存在
    for tag in ("subjective_state", "world_state", "pending", "recent_history"):
        assert f"<{tag}>" in body, f"section <{tag}> wrongly omitted in bootstrap"
    # empty literal も表示される (空 list の literal 表示)
    assert "(no subjective entries)" in body or "(no related memories)" in body


# ============================================================
# task section末尾配置 (recency bias 利用)
# ============================================================

def test_task_at_end(state_normal, ctrl_minimal, tools_minimal, monkeypatch):
    """task section は最末尾 (recency bias、PLAN §3-4)。"""
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: [])

    body = prompt_module.build_prompt_propose(state_normal, ctrl_minimal, tools_minimal)
    # 誤実装 (task が中盤) で fail
    task_end = body.rindex("</task>")
    # </task> 以降に他の section が出ないこと verify
    for tag in ("subjective_state", "world_state", "pending", "recent_history",
                "available_tools"):
        last_pos = body.rfind(f"<{tag}>")
        assert last_pos < task_end, f"<{tag}> appears after </task> (recency bias violated)"


# ============================================================
# Tool affordance guidance 確認 (silent loss marker、Codex audit P1-02 整合)
# ============================================================

def test_world_fact_view_affordance_guidance(state_normal, ctrl_minimal, tools_minimal, monkeypatch):
    """task block 内に「5 cycle より古い tool 事実は world_fact_view で取得可能」guidance 存在。

    Codex audit P1-02 (silent loss) ゆう確定 議論 A: affordance 経由取得 guidance を prompt に明示。
    """
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: [])

    body = prompt_module.build_prompt_propose(state_normal, ctrl_minimal, tools_minimal)
    # task block 内
    task_start = body.index("<task>")
    task_end = body.index("</task>")
    task_body = body[task_start:task_end]
    # 誤実装 (guidance 抜け) で fail
    assert "world_fact_view" in task_body, "affordance guidance missing in task block"
    assert "5 cycle より古い" in task_body, "silent loss notice missing"


# ============================================================
# LLM② (assemble_system_prompt) V07 検証 (hotfix 追加、AUD-P1-01 fix)
# ============================================================

def test_assemble_system_prompt_pending_included(state_normal, monkeypatch):
    """AUD-P1-01 fix: assemble_system_prompt sections に <pending> builder 接続必須。

    誤実装 (commit 4 ミス、pending builder 欠落) で fail する識別力 test。
    """
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: [])

    body = prompt_assembly.assemble_system_prompt(state_normal, tools_dict={})
    assert "<pending>" in body, "AUD-P1-01: <pending> open tag missing in assemble_system_prompt"
    assert "</pending>" in body, "AUD-P1-01: </pending> close tag missing"


def test_assemble_system_prompt_tool_block_xml(state_normal, monkeypatch):
    """AUD-P1-01 fix: tool block が <available_tools> XML tag に統一 (旧 [利用可能なツール] 撤去)。"""
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: [])

    body = prompt_assembly.assemble_system_prompt(state_normal, tools_dict={})
    # XML tag 統一
    assert "<available_tools>" in body, "AUD-P1-01: <available_tools> tag missing"
    assert "</available_tools>" in body, "AUD-P1-01: </available_tools> close missing"
    # 旧 heading 撤去
    assert "[利用可能なツール]" not in body, (
        "AUD-P1-01: legacy heading [利用可能なツール] still present"
    )


def test_assemble_system_prompt_v07_section_order(state_normal, monkeypatch):
    """LLM② 側 section 順序検証 (PLAN §3-4 案)。

    誤実装 (順序違反 / section 欠落) で fail。
    """
    import core.memory as mm
    monkeypatch.setattr(mm, "get_relevant_memories", lambda *a, **k: [])

    body = prompt_assembly.assemble_system_prompt(state_normal, tools_dict={})
    # V07 主要 section 全部存在
    expected_tags = ("subjective_state", "world_state", "pending",
                     "recent_history", "available_tools")
    positions = {}
    for tag in expected_tags:
        idx = body.find(f"<{tag}>")
        assert idx >= 0, f"section <{tag}> missing in assemble_system_prompt"
        positions[tag] = idx
    # 順序: subjective → world → pending → history → tools
    expected_order = list(expected_tags)
    sorted_by_pos = sorted(positions, key=lambda k: positions[k])
    assert sorted_by_pos == expected_order, (
        f"V07 section order violation in LLM② path: expected {expected_order}, got {sorted_by_pos}"
    )
