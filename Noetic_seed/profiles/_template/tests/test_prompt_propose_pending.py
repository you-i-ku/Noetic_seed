"""段階9 fix 1 — build_prompt_propose の pending 表示テスト

prompt.py:250- (段階9 fix 1) が以下を守ることを検証:
  - 未消化 pending だけが "未対応事項" に表示される
  - 消化済 pending (observed_content 有り or gap=0) は除外される
  - 消化済は「最近完了した応答 (参考)」セクションで直近 3 件まで表示される
  - pending 全件消化 + 全件未消化ゼロ → 「なし」

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_prompt_propose_pending.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 重い memory 依存 (archive 読込 + embedding) を外すためのスタブ
# 段階11-C G-lite Phase 1: get_relevant_memories に use_links/link_depth/link_top_n
# の keyword-only 引数追加に伴い、mock 側も **kwargs で受け流す。
import core.memory as _memory_mod
_memory_mod.get_relevant_memories = lambda state, limit=8, **kwargs: []
_memory_mod.format_memories_for_prompt = lambda mems, max_chars=2000: ""

from core.prompt import build_prompt_propose


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _fresh_state(pending=None):
    return {
        "cycle_id": 20,
        "log": [],
        "pending": pending or [],
        "self": {"name": "iku"},
        "energy": 50,
        "summaries": [],
    }


def _ctrl():
    return {"allowed_tools": {"read_file", "wait"}, "tool_level": 1}


def _tools():
    return {"read_file": {"desc": "ファイル読込"}, "wait": {"desc": "待機"}}


def _pending_unresolved(**overrides):
    """未消化の UPS v2 pending を生成 (observed_content=None, gap>0)。"""
    base = {
        "id": "p_test_0010_src_01",
        "type": "pending",
        "content": "未消化タスク",
        "priority": 0.7,
        "gap": 0.5,
        "attempts": 1,
        "source_action": "output_display",
        "observation_lag_kind": "reply",
        "expected_channel": "claude",
        "observed_channel": None,
        "observed_content": None,
        "observed_time": None,
        "origin_cycle": 10,
    }
    base.update(overrides)
    return base


def _pending_resolved(**overrides):
    """消化済の UPS v2 pending を生成 (observed_content 有、gap=0)。"""
    base = {
        "id": "p_test_0021_src_99",
        "type": "pending",
        "content": "消化済タスク",
        "priority": 0.0,
        "gap": 0.0,
        "attempts": 2,
        "source_action": "output_display",
        "observation_lag_kind": "reply",
        "expected_channel": "claude",
        "observed_channel": "claude",
        "observed_content": "応答届いた",
        "observed_time": "2026-04-20 07:28:18",
        "origin_cycle": 10,
    }
    base.update(overrides)
    return base


# ============================================================
# Fix 1 検証
# ============================================================

def test_resolved_excluded_from_unresolved_section():
    print("== 消化済 pending が '未対応事項' に出ない ==")
    state = _fresh_state(pending=[_pending_resolved()])
    prompt = build_prompt_propose(state, _ctrl(), _tools())
    # 消化済 content は未対応セクションに出ない
    return all([
        _assert("[pending " not in prompt, "未対応 pending マーカー '[pending ' が無い"),
        _assert("消化済タスク" in prompt, "content 自体は参考セクションに出る"),
        _assert("最近完了した応答" in prompt, "完了参考 heading 出現"),
    ])


def test_unresolved_displayed_in_unresolved_section():
    print("== 未消化 pending が '未対応事項' に表示される ==")
    state = _fresh_state(pending=[_pending_unresolved()])
    prompt = build_prompt_propose(state, _ctrl(), _tools())
    return all([
        _assert("[pending " in prompt, "未対応マーカー表示"),
        _assert("未消化タスク" in prompt, "content 表示"),
        _assert("最近完了した応答" not in prompt, "完了参考は消化済ゼロなので非表示"),
    ])


def test_mixed_pending_split_correctly():
    print("== 未消化 + 消化済 混在で正しく分離 ==")
    state = _fresh_state(pending=[
        _pending_unresolved(id="p_u1", content="未消化A"),
        _pending_resolved(id="p_r1", content="消化A",
                          observed_time="2026-04-20 07:00:00"),
        _pending_resolved(id="p_r2", content="消化B",
                          observed_time="2026-04-20 07:30:00"),
    ])
    prompt = build_prompt_propose(state, _ctrl(), _tools())
    # 未対応セクション
    u_start = prompt.find("[未対応事項]")
    done_start = prompt.find("最近完了した応答")
    return all([
        _assert(u_start >= 0 and done_start > u_start, "未対応 → 完了参考 順序"),
        _assert("未消化A" in prompt, "未消化content表示"),
        _assert("消化A" in prompt, "消化A content表示 (参考)"),
        _assert("消化B" in prompt, "消化B content表示 (参考)"),
        _assert("[完了 " in prompt, "完了マーカー '[完了 ' 表示"),
    ])


def test_resolved_limited_to_3_most_recent():
    print("== 消化済は直近 3 件までに制限 ==")
    pending = [
        _pending_resolved(id=f"p_r{i}", content=f"消化{i}",
                          observed_time=f"2026-04-20 07:{i:02d}:00")
        for i in range(5)
    ]
    state = _fresh_state(pending=pending)
    prompt = build_prompt_propose(state, _ctrl(), _tools())
    # observed_time 降順で直近 3 件 = 消化4, 消化3, 消化2
    return all([
        _assert("消化4" in prompt, "最新含む"),
        _assert("消化3" in prompt, "2番目含む"),
        _assert("消化2" in prompt, "3番目含む"),
        _assert("消化1" not in prompt, "4番目は除外"),
        _assert("消化0" not in prompt, "5番目は除外"),
    ])


def test_gap_zero_treated_as_resolved():
    print("== gap=0.0 (observed_content None でも) は消化済扱い ==")
    # エッジケース: match_pattern が gap=0 にしたが observed_content 未設定
    p = _pending_unresolved(gap=0.0)
    state = _fresh_state(pending=[p])
    prompt = build_prompt_propose(state, _ctrl(), _tools())
    # 未対応セクションに出ないことを確認
    u_section_end = prompt.find("[STM")
    u_section = prompt[:u_section_end]
    return _assert("[pending " not in u_section, "gap=0.0 は未対応に出ない")


def test_all_pending_empty():
    print("== pending 全件なし → 'なし' 表示 ==")
    state = _fresh_state(pending=[])
    prompt = build_prompt_propose(state, _ctrl(), _tools())
    return _assert("なし" in prompt, "'なし' 表示")


def test_all_resolved_no_unresolved_section():
    print("== 全 pending が消化済 → 未対応は 'なし' だが完了参考は出る ==")
    state = _fresh_state(pending=[
        _pending_resolved(id="p_r1", content="済A"),
        _pending_resolved(id="p_r2", content="済B"),
    ])
    prompt = build_prompt_propose(state, _ctrl(), _tools())
    # 未消化 0 件 → pending_lines は完了参考 section + entries のみ
    return all([
        _assert("最近完了した応答" in prompt, "完了参考 heading"),
        _assert("済A" in prompt, "済A 表示"),
        _assert("済B" in prompt, "済B 表示"),
        _assert("[pending " not in prompt, "未対応 pending マーカーなし"),
    ])


def test_legacy_pending_type_still_works():
    print("== 旧形式 pending (type != 'pending') も未消化判定で表示 ==")
    legacy = {
        "id": "old_001",
        "type": "response_to_external",
        "content": "旧形式",
        "priority": 0.5,
        "gap": 0.8,
        "channel": "device",
        "timestamp": "2026-04-20 06:00:00",
        # UPS v2 フィールド欠如 (observed_content key 自体ない)
    }
    state = _fresh_state(pending=[legacy])
    prompt = build_prompt_propose(state, _ctrl(), _tools())
    return all([
        _assert("[response_to_external " in prompt, "旧形式マーカー表示"),
        _assert("旧形式" in prompt, "content 表示"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("消化済が未対応に出ない", test_resolved_excluded_from_unresolved_section),
        ("未消化が未対応に表示", test_unresolved_displayed_in_unresolved_section),
        ("混在で正しく分離", test_mixed_pending_split_correctly),
        ("消化済は直近 3 件まで", test_resolved_limited_to_3_most_recent),
        ("gap=0.0 は消化済扱い", test_gap_zero_treated_as_resolved),
        ("pending 全件なしで 'なし'", test_all_pending_empty),
        ("全消化済で未対応に出ない", test_all_resolved_no_unresolved_section),
        ("旧形式 pending も表示", test_legacy_pending_type_still_works),
    ]
    results = []
    for _label, fn in groups:
        print()
        ok = fn()
        results.append((_label, ok))
    print()
    print("=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for _label, ok in results:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {_label}")
    print(f"\n  {passed}/{total} groups passed")
    sys.exit(0 if passed == total else 1)
