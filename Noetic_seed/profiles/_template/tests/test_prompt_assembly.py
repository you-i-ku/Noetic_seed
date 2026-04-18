"""prompt_assembly.py テスト。

成功条件:
  - 5 要素が順序良く含まれる
  - 承認プロトコル指示 / 世界モデル stub が LLM に届く形
  - 発火原因メタ注入が動的
  - prompt 予算超過で警告 or raise
  - 既存 _pack_log_block / _build_tool_lines が流用されている

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_prompt_assembly.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.prompt_assembly import (
    SYSTEM_PROMPT_SOFT_LIMIT,
    assemble_system_prompt,
    build_approval_protocol,
    build_fire_cause_section,
    build_log_block,
    build_tool_block,
    build_world_model_section,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _fresh_state():
    return {
        "cycle_id": 10,
        "log": [],
        "pending": [],
        "self": {"name": "iku"},
        "energy": 50,
    }


def _sample_tools():
    return {
        "read_file": {"desc": "ファイル読込"},
        "write_file": {"desc": "ファイル書込"},
        "wait": {"desc": "待機"},
    }


# ============================================================
# 個別 builder
# ============================================================

def test_approval_protocol_has_3_fields():
    print("== 承認プロトコル: 3 フィールド + 対等協力者 文言 ==")
    s = build_approval_protocol()
    return all([
        _assert("tool_intent" in s, "tool_intent 明記"),
        _assert("tool_expected_outcome" in s, "tool_expected_outcome 明記"),
        _assert("message" in s, "message 明記"),
        _assert("対等" in s, "対等な協力者"),
        _assert("許可してください" in s, "上下関係語彙の禁止例あり"),
    ])


def test_fire_cause_section_empty():
    print("== fire_cause='' → 空文字 (省略される) ==")
    return _assert(build_fire_cause_section("") == "", "空文字")


def test_fire_cause_section_with_value():
    print("== fire_cause='X' → [発火原因: X] ==")
    s = build_fire_cause_section("threshold breach")
    return all([
        _assert(s.startswith("[発火原因:"), "prefix"),
        _assert("threshold breach" in s, "fire_cause 文字列埋込"),
    ])


def test_world_model_stub():
    print("== 世界モデル: Phase 4 は stub ==")
    s = build_world_model_section()
    return all([
        _assert("stub" in s.lower() or "世界モデル" in s, "section あり"),
        _assert("Phase 5" in s, "Phase 5 言及"),
    ])


def test_log_block_empty():
    print("== log block: 空 log でも落ちない ==")
    s = build_log_block(_fresh_state(), budget_tok=1000)
    return _assert(isinstance(s, str), "文字列返却")


def test_log_block_with_entries():
    print("== log block: entry あり → 1 行ずつレンダ ==")
    state = _fresh_state()
    state["log"] = [
        {"id": "e1", "time": "09:00", "tool": "read_file",
         "intent": "設定確認", "result": "OK"},
        {"id": "e2", "time": "09:05", "tool": "write_file",
         "intent": "更新", "result": "done"},
    ]
    s = build_log_block(state, budget_tok=1000)
    return all([
        _assert("read_file" in s, "tool 1 含む"),
        _assert("write_file" in s, "tool 2 含む"),
        _assert("設定確認" in s, "intent 含む"),
    ])


def test_tool_block_filters_by_allowed():
    print("== tool block: allowed_tools で絞り込み ==")
    tools = _sample_tools()
    s_all = build_tool_block(None, tools)
    s_subset = build_tool_block({"read_file"}, tools)
    return all([
        _assert("read_file" in s_all and "write_file" in s_all,
                "全 tool 含む"),
        _assert("read_file" in s_subset, "subset に read_file"),
        _assert("write_file" not in s_subset, "subset に write_file 無し"),
    ])


def test_tool_block_registry_fallback():
    print("== tool block: tools_dict に無い tool を registry から補完 (B 案) ==")
    from types import SimpleNamespace

    class _MockReg:
        def get(self, name):
            specs = {
                "glob_search": SimpleNamespace(
                    description="ファイルパターン検索 (claw)"),
                "WebSearch": SimpleNamespace(
                    description="Web 検索 (claw)"),
            }
            return specs.get(name)

    tools = _sample_tools()  # read_file, write_file, wait のみ
    allowed = {"read_file", "glob_search", "WebSearch"}
    s = build_tool_block(allowed, tools, registry=_MockReg())
    return all([
        _assert("read_file" in s, "TOOLS にある tool 表示継続"),
        _assert("glob_search" in s, "registry から glob_search 補完"),
        _assert("ファイルパターン検索 (claw)" in s, "claw description 取得"),
        _assert("WebSearch" in s, "registry から WebSearch 補完"),
    ])


def test_tool_block_no_registry_still_works():
    print("== tool block: registry=None で従来動作 (後方互換) ==")
    tools = _sample_tools()
    allowed = {"read_file", "glob_search"}
    s = build_tool_block(allowed, tools, registry=None)
    return all([
        _assert("read_file" in s, "TOOLS 経由で read_file"),
        _assert("glob_search" not in s, "registry なしなら glob_search 非表示"),
    ])


# ============================================================
# 全体 assembly
# ============================================================

def test_assemble_contains_all_five_sections():
    print("== assemble: 5 要素全部含む ==")
    state = _fresh_state()
    state["log"] = [{"id": "e1", "time": "09:00", "tool": "read_file",
                     "intent": "x", "result": "y"}]
    tools = _sample_tools()
    prompt = assemble_system_prompt(
        state=state, tools_dict=tools,
        fire_cause="threshold breach",
    )
    return all([
        _assert("Approval Protocol" in prompt, "① 承認プロトコル"),
        _assert("発火原因: threshold breach" in prompt, "② 発火原因"),
        _assert("世界モデル" in prompt, "③ 世界モデル stub"),
        _assert("STM — log" in prompt, "④ log block heading"),
        _assert("read_file" in prompt, "④ log 中身"),
        _assert("利用可能なツール" in prompt, "⑤ tool 一覧 heading"),
    ])


def test_assemble_section_order():
    print("== assemble: 5 要素の順序が正しい ==")
    state = _fresh_state()
    state["log"] = [{"id": "e1", "time": "09:00", "tool": "read_file",
                     "intent": "x", "result": "y"}]
    prompt = assemble_system_prompt(
        state=state, tools_dict=_sample_tools(),
        fire_cause="test cause",
    )
    i_approval = prompt.find("Approval Protocol")
    i_fire = prompt.find("発火原因")
    i_wm = prompt.find("世界モデル")
    i_log = prompt.find("STM — log")
    i_tools = prompt.find("利用可能なツール")
    return all([
        _assert(i_approval < i_fire, "Approval < 発火原因"),
        _assert(i_fire < i_wm, "発火原因 < 世界モデル"),
        _assert(i_wm < i_log, "世界モデル < log"),
        _assert(i_log < i_tools, "log < tools"),
    ])


def test_assemble_fire_cause_omitted():
    print("== assemble: fire_cause 空 → 発火原因セクション省略 ==")
    prompt = assemble_system_prompt(
        state=_fresh_state(), tools_dict=_sample_tools(),
        fire_cause="",
    )
    return all([
        _assert("発火原因" not in prompt, "発火原因なし"),
        _assert("Approval Protocol" in prompt, "他 4 要素は残る"),
        _assert("利用可能なツール" in prompt, "tool 一覧残る"),
    ])


def test_assemble_no_magic_if():
    print("== assemble: Magic-If 語彙が残っていない ==")
    prompt = assemble_system_prompt(
        state=_fresh_state(), tools_dict=_sample_tools(),
        fire_cause="",
    )
    return all([
        _assert("Magic-If" not in prompt, "Magic-If 言及なし"),
        _assert("意味的同一性" not in prompt, "意味的同一性 言及なし"),
        _assert("given circumstances" not in prompt, "given circumstances 言及なし"),
    ])


def test_assemble_within_budget():
    print("== assemble: 通常条件で SOFT_LIMIT 内に収まる ==")
    state = _fresh_state()
    state["log"] = [{"id": f"e{i}", "time": "09:00",
                     "tool": "read_file", "intent": f"intent {i}",
                     "result": "x"} for i in range(30)]
    prompt = assemble_system_prompt(
        state=state, tools_dict=_sample_tools(),
        fire_cause="",
        log_budget_tok=2000,
    )
    from core.config import estimate_tokens
    return _assert(estimate_tokens(prompt) <= SYSTEM_PROMPT_SOFT_LIMIT,
                   f"token={estimate_tokens(prompt)} <= {SYSTEM_PROMPT_SOFT_LIMIT}")


def test_assemble_overbudget_raises():
    print("== assemble: raise_on_overbudget=True で超過時 ValueError ==")
    state = _fresh_state()
    # 超長 log を強引に作る
    state["log"] = [{"id": f"e{i}", "time": "09:00", "tool": "read_file",
                     "intent": "x", "result": "Y" * 1000} for i in range(100)]
    try:
        assemble_system_prompt(
            state=state, tools_dict=_sample_tools(),
            fire_cause="", log_budget_tok=100_000,
            raise_on_overbudget=True,
        )
        return _assert(False, "ValueError 期待")
    except ValueError as e:
        return all([
            _assert(True, "ValueError 発生"),
            _assert("超過" in str(e), "msg に '超過' 含む"),
        ])


def test_assemble_overbudget_warns_no_raise():
    print("== assemble: raise_on_overbudget=False → stderr 警告のみ ==")
    state = _fresh_state()
    state["log"] = [{"id": f"e{i}", "time": "09:00", "tool": "read_file",
                     "intent": "x", "result": "Y" * 1000} for i in range(100)]
    try:
        prompt = assemble_system_prompt(
            state=state, tools_dict=_sample_tools(),
            fire_cause="", log_budget_tok=100_000,
            raise_on_overbudget=False,
        )
        return all([
            _assert(isinstance(prompt, str), "文字列が返る (raise せず)"),
            _assert(len(prompt) > 0, "空ではない"),
        ])
    except ValueError:
        return _assert(False, "raise すべきでない")


def test_assemble_allowed_tools_filter():
    print("== assemble: allowed_tools で tool 一覧が絞られる ==")
    prompt = assemble_system_prompt(
        state=_fresh_state(), tools_dict=_sample_tools(),
        allowed_tools={"read_file"},
    )
    return all([
        _assert("read_file" in prompt, "read_file 含む"),
        _assert("write_file" not in prompt, "write_file 除外"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("承認プロトコル: 3 層 + 対等", test_approval_protocol_has_3_fields),
        ("発火原因: 空なら空文字", test_fire_cause_section_empty),
        ("発火原因: 値で prefix 付与", test_fire_cause_section_with_value),
        ("世界モデル: stub", test_world_model_stub),
        ("log block: 空 OK", test_log_block_empty),
        ("log block: entries", test_log_block_with_entries),
        ("tool block: allowed_tools 絞込", test_tool_block_filters_by_allowed),
        ("tool block: registry fallback (B 案)", test_tool_block_registry_fallback),
        ("tool block: registry なし後方互換", test_tool_block_no_registry_still_works),
        ("assemble: 5 要素含む", test_assemble_contains_all_five_sections),
        ("assemble: 順序", test_assemble_section_order),
        ("assemble: fire_cause 省略", test_assemble_fire_cause_omitted),
        ("assemble: Magic-If 痕跡なし", test_assemble_no_magic_if),
        ("assemble: 予算内", test_assemble_within_budget),
        ("assemble: 予算超過 raise", test_assemble_overbudget_raises),
        ("assemble: 予算超過 warn", test_assemble_overbudget_warns_no_raise),
        ("assemble: allowed_tools", test_assemble_allowed_tools_filter),
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
