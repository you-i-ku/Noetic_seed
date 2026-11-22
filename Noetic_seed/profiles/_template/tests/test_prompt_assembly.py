"""prompt_assembly.py テスト。

成功条件:
  - V07 各セクションの内容・順序・省略条件を検証する
  - 主観履歴 / 世界の事実 / 未対応 / 対応済みを混同しない
  - 発火原因メタ注入が動的
  - prompt 予算超過で警告 or raise
  - build_log_block は assembly では未使用。残存関数の単体テストとして維持する

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_prompt_assembly.py
"""
import re
import sys
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

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
        "raw_events": [], "subjective_entries": [],
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

def test_approval_protocol_reason_and_note():
    print("== 承認プロトコル: 理由・予想 + 自由欄 ==")
    s = build_approval_protocol()
    return all([
        _assert("tool_intent" in s, "tool_intent 明記"),
        _assert("tool_expected_outcome" in s, "tool_expected_outcome 明記"),
        _assert("note は自由に書ける欄" in s, "自由欄"),
        _assert("確認相手" not in s and "message" not in s, "確認相手なし"),
        _assert("口調" not in s and "許可してください" not in s, "口調指定なし"),
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


def test_world_model_section_none():
    print("== 世界モデル: world_model=None で空文字 (セクション省略) ==")
    s = build_world_model_section(None)
    return _assert(s == "", "空文字返却")


def test_world_model_section_renders():
    print("== 世界モデル: 段階6-C v3 実装 (観察で channel が生えた WM を描画) ==")
    from core.world_model import init_world_model, ensure_channel
    from core.channel_registry import channel_from_device_input, channel_from_mcp_client
    wm = init_world_model()
    # (v3) 起動直後は channels 空、観察で生える。テスト目的で device/claude を明示登録
    ensure_channel(wm, **channel_from_device_input())
    ensure_channel(wm, **channel_from_mcp_client("claude-code"))
    s = build_world_model_section(wm)
    return all([
        _assert("## 世界モデル" in s, "section heading"),
        _assert("### チャネル" in s, "チャネル heading"),
        _assert("device (direct)" in s, "device 行"),
        _assert("claude (social)" in s, "claude 行"),
    ])


def test_log_block_empty():
    """V07 assembly では未使用の旧 builder を単体で確認する。"""
    print("== log block: 空 log でも落ちない ==")
    s = build_log_block(_fresh_state(), budget_tok=1000)
    return _assert(isinstance(s, str), "文字列返却")


def test_log_block_with_entries():
    """V07 assembly では未使用の旧 builder を単体で確認する。"""
    print("== log block: entry あり → 1 行ずつレンダ ==")
    state = _fresh_state()
    # 段階13 Phase 0.1.D: 哲学最 literal で raw / subjective 二層分割
    # raw=monitor footage (time/tool/result)、subjective=journal (intent)、id 共有
    state["raw_events"] = [
        {"id": "e1", "time": "09:00", "tool": "read_file", "result": "OK"},
        {"id": "e2", "time": "09:05", "tool": "write_file", "result": "done"},
    ]
    state["subjective_entries"] = [
        {"id": "e1", "intent": "設定確認"},
        {"id": "e2", "intent": "更新"},
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

def _assemble(**kwargs):
    """保存済み記憶・実ファイル状態だけ隔離し、実際の builder で組み立てる。"""
    snapshot = {
        "state_json": {"size": 11, "mtime": None},
        "raw_events_jsonl": {"size": 22, "mtime": None},
        "orphan": [],
    }
    with patch("core.memory.get_relevant_memories", return_value=[]), \
            patch("core.memory.list_records", return_value=[]), \
            patch("core.memory_links.list_links", return_value=[]), \
            patch("core.world_state_view._file_snapshot", return_value=snapshot):
        return assemble_system_prompt(**kwargs)


def _populated_state():
    state = _fresh_state()
    # 最新 5 件を使い、古い 1 件の混入を検出する。
    state["raw_events"] = [
        {"id": f"e{i}", "tool": "read_file", "args": {"path": f"file-{i}"},
         "result": f"fact-marker-{i}"}
        for i in range(6)
    ]
    state["subjective_entries"] = [
        {"id": f"e{i}", "intent": f"intent-marker-{i}", "e1": 0.5}
        for i in range(6)
    ]
    state["pending"] = [
        {"id": "unresolved", "type": "pending", "gap": 0.8,
         "content": "unresolved-marker", "source_action": "read_file"},
        {"id": "resolved", "type": "pending", "gap": 0.0,
         "content": "resolved-marker", "observed_content": "done",
         "observed_time": "2026-05-16 09:00"},
    ]
    return state


def _world_model():
    from core.world_model import init_world_model, ensure_channel
    from core.channel_registry import channel_from_device_input
    wm = init_world_model()
    ensure_channel(wm, **channel_from_device_input())
    return wm


def _block(prompt, tag):
    return prompt.partition(f"<{tag}>\n")[2].partition(f"\n</{tag}>")[0]


def test_assemble_contains_v07_sections():
    print("== assemble: V07 各要素の内容と分離 ==")
    # 旧 log builder が再接続された場合も検出する。
    with patch("core.prompt_assembly.build_log_block",
               side_effect=AssertionError("V07 assembly must not use build_log_block")):
        prompt = _assemble(
            state=_populated_state(), tools_dict=_sample_tools(),
            fire_cause="threshold breach", world_model=_world_model(),
            force_tool="read_file",
        )
    results = []
    for tag in ("subjective_state", "world_state", "pending", "recent_resolved",
                "recent_history", "available_tools"):
        results.append(_assert(
            prompt.count(f"<{tag}>") == prompt.count(f"</{tag}>") == 1,
            f"{tag}: 開始・終了タグが各 1 個"))
    subject = _block(prompt, "subjective_state")
    world = _block(prompt, "world_state")
    history = _block(prompt, "recent_history")
    pending = _block(prompt, "pending")
    resolved = _block(prompt, "recent_resolved")
    results.extend([
        _assert("## Tool 呼び出し" in prompt, "承認プロトコル"),
        _assert("[発火原因: threshold breach]" in prompt, "発火原因"),
        _assert('"name": "iku"' in subject and "graph_summary:" in subject
                and "related_memory:" in subject, "主観状態の中身"),
        _assert("file_snapshot:" in world and "recent_events:" in world,
                "世界状態の中身"),
        _assert("## 世界モデル" in prompt and "device (direct)" in prompt,
                "世界モデルと channel"),
        _assert("[pending dismiss=unresolved " in pending
                and "unresolved-marker" in pending
                and "[完了" not in pending, "未対応だけを pending に配置"),
        _assert("resolved-marker → 観測済" in resolved
                and "unresolved-marker" not in resolved, "対応済みを独立配置"),
        _assert("e1=0.5" in history, "主観履歴の評価値"),
        _assert("fact-marker-" not in history and "args=" not in history,
                "主観履歴に世界の事実を混ぜない"),
        _assert("intent-marker-" not in world, "世界状態に主観履歴を混ぜない"),
        _assert(_block(prompt, "available_tools").splitlines() == [
            "  read_file: ファイル読込", "  write_file: ファイル書込", "  wait: 待機"],
            "tool 一覧の名前と説明"),
        _assert("[強制実行指示]" in prompt and "ツール「read_file」" in prompt,
                "force_tool の対象"),
        _assert("[log]" not in prompt, "旧 log block は含めない"),
        _assert("fact-marker-0" not in world and "intent-marker-0" not in history,
                "最新 5 件より前は直近欄に含めない"),
    ])
    for i in range(1, 6):
        results.extend([
            _assert(f"fact-marker-{i}" in world, f"世界の事実 {i}"),
            _assert(f"intent-marker-{i}" in history, f"主観履歴 {i}"),
        ])
    return all(results)


def test_assemble_section_order():
    print("== assemble: optional 要素も含めた V07 の順序 ==")
    prompt = _assemble(
        state=_populated_state(), tools_dict=_sample_tools(),
        fire_cause="test cause", world_model=_world_model(), force_tool="read_file",
    )
    expected = [
        "## Tool 呼び出し", "[発火原因: test cause]",
        "<subjective_state>", "<world_state>", "## 世界モデル",
        "<pending>", "<recent_resolved>", "<recent_history>",
        "<available_tools>", "[強制実行指示]",
    ]
    # find() の -1 を成功と誤認しない。欠落・重複・順序変更の全てで落ちる。
    headings = re.findall(r"^(?:" + "|".join(re.escape(h) for h in expected) + r")$",
                          prompt, flags=re.MULTILINE)
    return all([
        _assert(headings == expected, "全 section が期待順に各 1 回"),
        _assert(prompt.startswith(expected[0]), "承認が先頭"),
        _assert(prompt.endswith("text のみの応答は許可されません。"), "強制指示が末尾"),
    ])


def test_assemble_wm_omitted_when_none():
    print("== assemble: world_model=None でも V07 の必須 section は残る ==")
    prompt = _assemble(
        state=_fresh_state(), tools_dict=_sample_tools(),
        fire_cause="", world_model=None,
    )
    return all([
        _assert("## 世界モデル" not in prompt, "世界モデル heading 省略"),
        _assert("## Tool 呼び出し" in prompt, "承認は残る"),
        _assert('"name": "iku"' in _block(prompt, "subjective_state"), "主観は残る"),
        _assert("(no recent events)" in _block(prompt, "world_state"), "空の世界状態"),
        _assert(_block(prompt, "pending") == "  なし", "空の未対応事項"),
        _assert("(no subjective entries)" in _block(prompt, "recent_history"),
                "空の主観履歴"),
        _assert("read_file: ファイル読込" in _block(prompt, "available_tools"),
                "tool 一覧残る"),
        _assert("<recent_resolved>" not in prompt, "対応済みなしなら省略"),
        _assert("[強制実行指示]" not in prompt, "force_tool なしなら省略"),
    ])


def test_assemble_fire_cause_omitted():
    print("== assemble: 空の発火原因だけ省略し、他の中身は維持 ==")
    kwargs = dict(state=_populated_state(), tools_dict=_sample_tools(),
                  world_model=_world_model())
    with_fire = _assemble(**kwargs, fire_cause="fire-marker")
    without_fire = _assemble(**kwargs, fire_cause="")
    return all([
        _assert("[発火原因:" not in without_fire, "発火原因なし"),
        _assert(without_fire == with_fire.replace("[発火原因: fire-marker]\n\n", ""),
                "発火原因以外は変えない"),
        _assert("read_file: ファイル読込" in _block(without_fire, "available_tools"),
                "tool 一覧残る"),
    ])


def test_assemble_no_magic_if():
    print("== assemble: Magic-If 語彙が残っていない ==")
    prompt = _assemble(
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
    # 段階13 Phase 0.1.D: raw / subj 二層分割
    state["raw_events"] = [{"id": f"e{i}", "time": "09:00",
                     "tool": "read_file", "result": "x"} for i in range(30)]
    state["subjective_entries"] = [{"id": f"e{i}", "intent": f"intent {i}"}
                     for i in range(30)]
    prompt = _assemble(
        state=state, tools_dict=_sample_tools(),
        fire_cause="",
    )
    from core.config import estimate_tokens
    return _assert(estimate_tokens(prompt) <= SYSTEM_PROMPT_SOFT_LIMIT,
                   f"token={estimate_tokens(prompt)} <= {SYSTEM_PROMPT_SOFT_LIMIT}")


def test_assemble_overbudget_raises():
    print("== assemble: 実測 token 数が上限を超えると指定時だけ ValueError ==")
    from core.config import estimate_tokens
    kwargs = dict(state=_fresh_state(), tools_dict=_sample_tools())
    tokens = estimate_tokens(_assemble(**kwargs))
    limit = tokens - 1
    stderr = StringIO()
    with patch("core.prompt_assembly.SYSTEM_PROMPT_SOFT_LIMIT", limit), redirect_stderr(stderr):
        try:
            _assemble(**kwargs, raise_on_overbudget=True)
        except ValueError as e:
            return all([
                _assert("[assemble_system_prompt]" in str(e) and "超過" in str(e),
                        "assembly の予算例外"),
                _assert(f"{tokens} > {limit}" in str(e), "実測値と上限が例外に入る"),
                _assert(stderr.getvalue() == "", "raise 時は stderr 警告なし"),
            ])
    return _assert(False, "ValueError 期待")


def test_assemble_overbudget_warns_no_raise():
    print("== assemble: デフォルト / False は超過を stderr に警告して返す ==")
    from core.config import estimate_tokens
    kwargs = dict(state=_fresh_state(), tools_dict=_sample_tools())
    baseline = _assemble(**kwargs)
    tokens = estimate_tokens(baseline)
    limit = tokens - 1
    results = []
    for options in ({}, {"raise_on_overbudget": False}):
        stderr = StringIO()
        with patch("core.prompt_assembly.SYSTEM_PROMPT_SOFT_LIMIT", limit), redirect_stderr(stderr):
            prompt = _assemble(**kwargs, **options)
        results.extend([
            _assert(prompt == baseline, "超過しても prompt を切り詰めない"),
            _assert(stderr.getvalue() == (
                f"[assemble_system_prompt] system_prompt トークン超過: {tokens} > {limit}\n"),
                "超過警告を stderr に 1 回出す"),
        ])
    return all(results)


def test_assemble_budget_boundary():
    print("== assemble: 上限一致・上限未満は警告も例外もなし ==")
    from core.config import estimate_tokens
    kwargs = dict(state=_fresh_state(), tools_dict=_sample_tools())
    baseline = _assemble(**kwargs)
    tokens = estimate_tokens(baseline)
    results = []
    for limit in (tokens, tokens + 1):
        for strict in (False, True):
            stderr = StringIO()
            with patch("core.prompt_assembly.SYSTEM_PROMPT_SOFT_LIMIT", limit), redirect_stderr(stderr):
                prompt = _assemble(**kwargs, raise_on_overbudget=strict)
            results.extend([
                _assert(prompt == baseline, f"limit={limit}, strict={strict}: prompt 維持"),
                _assert(stderr.getvalue() == "", f"limit={limit}, strict={strict}: 警告なし"),
            ])
    return all(results)


def test_assemble_allowed_tools_filter():
    print("== assemble: allowed_tools で tool 一覧が絞られる ==")
    state = _fresh_state()
    state["raw_events"] = [{"tool": "write_file", "result": "past-write-marker"}]
    prompt = _assemble(
        state=state, tools_dict=_sample_tools(),
        allowed_tools={"read_file"},
    )
    return all([
        _assert(_block(prompt, "available_tools") == "  read_file: ファイル読込",
                "tool 一覧は read_file のみ"),
        _assert("[write_file]" in _block(prompt, "world_state")
                and "past-write-marker" in _block(prompt, "world_state"),
                "過去の世界の事実は allowed_tools で削除しない"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("理由・予想 + 自由欄", test_approval_protocol_reason_and_note),
        ("発火原因: 空なら空文字", test_fire_cause_section_empty),
        ("発火原因: 値で prefix 付与", test_fire_cause_section_with_value),
        ("世界モデル: None で空", test_world_model_section_none),
        ("世界モデル: 段階2 render", test_world_model_section_renders),
        ("log block: 空 OK", test_log_block_empty),
        ("log block: entries", test_log_block_with_entries),
        ("tool block: allowed_tools 絞込", test_tool_block_filters_by_allowed),
        ("tool block: registry fallback (B 案)", test_tool_block_registry_fallback),
        ("tool block: registry なし後方互換", test_tool_block_no_registry_still_works),
        ("assemble: V07 各要素", test_assemble_contains_v07_sections),
        ("assemble: 順序", test_assemble_section_order),
        ("assemble: wm=None で省略", test_assemble_wm_omitted_when_none),
        ("assemble: fire_cause 省略", test_assemble_fire_cause_omitted),
        ("assemble: Magic-If 痕跡なし", test_assemble_no_magic_if),
        ("assemble: 予算内", test_assemble_within_budget),
        ("assemble: 予算超過 raise", test_assemble_overbudget_raises),
        ("assemble: 予算超過 warn", test_assemble_overbudget_warns_no_raise),
        ("assemble: 予算境界", test_assemble_budget_boundary),
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
