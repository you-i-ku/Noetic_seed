"""段階8 Step 3 — log entry args 表示 + [REJECTED] [warn] マーク prompt 注入テスト。

WORLD_MODEL_DESIGN/STAGE8_REPETITION_AND_PREDICTOR_PLAN.md §4-5 / §4-6:
  - 改善1: _render_log_entry で args フィールド表示 (cap 200、長ければ "..." 省略)
  - 改善3: result に "[REJECTED]" 含む場合、行頭 "[warn]" prefix
  - build_log_block 冒頭に表示規約の事実説明 (LLM as brain 整合、命令なし)

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_prompt_log_args_and_reject.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.prompt import _render_log_entry
from core.prompt_assembly import build_log_block


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _base_entry(**overrides):
    """最小エントリ (id / time / tool 必須、他は option)。"""
    base = {
        "id": "test_0001",
        "time": "2026-04-19 21:00:00",
        "tool": "output_display",
        "result": "送信完了 (device)",
    }
    base.update(overrides)
    return base


# ============================================================
# 改善1: args 表示
# ============================================================

def test_args_displayed():
    print("== args: 設定時に args:{...} 表示 ==")
    entry = _base_entry(
        args={"channel": "claude", "content": "hi onee-tan"},
    )
    line = _render_log_entry(entry, result_cap=500, intent_cap=300)
    return all([
        _assert("args:" in line, "args: ラベル表示"),
        _assert("claude" in line, "channel 値表示"),
        _assert("hi onee-tan" in line, "content 値表示"),
    ])


def test_args_missing_no_display():
    print("== args: 未設定なら args: 表示なし ==")
    entry = _base_entry()  # args 未設定
    line = _render_log_entry(entry, result_cap=500, intent_cap=300)
    return _assert("args:" not in line, "args ラベルなし")


def test_args_long_truncated():
    print("== args: 200 文字超は ... で省略 ==")
    long_content = "あ" * 300
    entry = _base_entry(args={"channel": "device", "content": long_content})
    line = _render_log_entry(entry, result_cap=500, intent_cap=300)
    return all([
        _assert("..." in line, "省略マーカー ..."),
        _assert("args:" in line, "args ラベル残る"),
    ])


# ============================================================
# 改善3: [REJECTED] [warn] マーク
# ============================================================

WARN_MARK = "\u26a0\ufe0f"  # Unicode エスケープで cp932 print を回避 (テスト内部用)


def test_rejected_prefix_warning():
    print("== REJECTED: 行頭 warn prefix ==")
    entry = _base_entry(
        result="[REJECTED] approval denied",
    )
    line = _render_log_entry(entry, result_cap=500, intent_cap=300)
    return all([
        _assert(line.startswith(WARN_MARK), "行頭 warn mark prefix"),
        _assert("[REJECTED]" in line, "[REJECTED] が result に残る"),
    ])


def test_normal_result_no_warning():
    print("== 通常 result: warn なし (空白 prefix) ==")
    entry = _base_entry(result="送信完了 (device)")
    line = _render_log_entry(entry, result_cap=500, intent_cap=300)
    return all([
        _assert(not line.startswith(WARN_MARK), "warn mark prefix なし"),
        _assert(line.startswith("  "), "通常は 2 スペース prefix"),
    ])


def test_rejected_variants_all_get_warning():
    print("== REJECTED: 3 つの reject 経路すべて warn ==")
    variants = [
        "[REJECTED] approval denied",
        "[REJECTED] denied by pre hook",
        "[REJECTED] permission denied",
    ]
    all_ok = True
    for v in variants:
        entry = _base_entry(result=v)
        line = _render_log_entry(entry, result_cap=500, intent_cap=300)
        ok = line.startswith(WARN_MARK)
        all_ok = all_ok and ok
        _assert(ok, f"'{v}' -> warn mark")
    return all_ok


# ============================================================
# build_log_block: 表示規約の事実説明
# ============================================================

def test_build_log_block_explainer():
    print("== build_log_block: 表示規約の explainer 冒頭 ==")
    state = {
        "log": [_base_entry(args={"channel": "device", "content": "hi"})],
        "cycle_id": 1,
    }
    block = build_log_block(state, budget_tok=2000)
    return all([
        _assert("表示規約" in block, "explainer キーワード含む"),
        _assert("args:" in block, "args 説明含む"),
        _assert(WARN_MARK in block, "warn mark 説明含む"),
    ])


def test_build_log_block_contains_entries():
    print("== build_log_block: 通常 log エントリも含まれる ==")
    state = {
        "log": [_base_entry(tool="search_memory", result="3 件見つかった")],
        "cycle_id": 1,
    }
    block = build_log_block(state, budget_tok=2000)
    return all([
        _assert("search_memory" in block, "tool 名表示"),
        _assert("3 件見つかった" in block, "result 表示"),
    ])


# ============================================================
# 段階9 fix 2-a: channel tag の [channel=X] 形式
# ============================================================

def test_channel_tag_uses_key_value_format():
    print("== channel tag: [channel=X] 形式 ==")
    entry = _base_entry(channel="claude")
    line = _render_log_entry(entry, result_cap=500, intent_cap=300)
    return all([
        _assert("[channel=claude]" in line, "[channel=claude] 表示"),
        _assert("[claude] " not in line, "旧 [claude] 形式は消えている"),
    ])


def test_channel_tag_omitted_when_empty():
    print("== channel 空なら channel tag も出ない ==")
    entry = _base_entry(channel="")
    line = _render_log_entry(entry, result_cap=500, intent_cap=300)
    return all([
        _assert("[channel=" not in line, "channel tag 非表示"),
        _assert("output_display" in line, "tool 名は表示"),
    ])


def test_channel_tag_device():
    print("== channel=device でも key=value 形式 ==")
    entry = _base_entry(channel="device", tool="[device_input]")
    line = _render_log_entry(entry, result_cap=500, intent_cap=300)
    return _assert("[channel=device]" in line, "device 側も key=value")


# ============================================================
# 統合: args + rejected の同時表示
# ============================================================

def test_integration_args_and_rejected():
    print("== 統合: args + [REJECTED] 同時 ==")
    entry = _base_entry(
        args={"channel": "claude", "content": "もう一度言うね"},
        result="[REJECTED] approval denied",
        intent="おねーたんに再度伝える",
    )
    line = _render_log_entry(entry, result_cap=500, intent_cap=300)
    return all([
        _assert(line.startswith(WARN_MARK), "warn mark prefix"),
        _assert("args:" in line, "args 表示"),
        _assert("claude" in line, "args に channel=claude"),
        _assert("intent=" in line, "intent 表示"),
        _assert("[REJECTED]" in line, "[REJECTED] 残る"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("args: 表示", test_args_displayed),
        ("args: 未設定で非表示", test_args_missing_no_display),
        ("args: 長文省略", test_args_long_truncated),
        ("REJECTED: [warn] prefix", test_rejected_prefix_warning),
        ("通常: [warn] なし", test_normal_result_no_warning),
        ("REJECTED: 3 経路全て [warn]", test_rejected_variants_all_get_warning),
        ("build_log_block: explainer", test_build_log_block_explainer),
        ("build_log_block: entry 含む", test_build_log_block_contains_entries),
        ("channel tag: [channel=X] 形式", test_channel_tag_uses_key_value_format),
        ("channel tag: 空で省略", test_channel_tag_omitted_when_empty),
        ("channel tag: device も key=value", test_channel_tag_device),
        ("統合: args + rejected", test_integration_args_and_rejected),
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
