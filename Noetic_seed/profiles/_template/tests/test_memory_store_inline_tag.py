"""memory_store inline 新タグ登録テスト (段階7 Step 5)。

成功条件:
  - 未登録タグ + rules → 登録 + append 成功
  - 未登録タグ + rules 欠落 → エラー (タグ未登録 / jsonl 未書込)
  - 登録済タグでの store → rules 無視、append 成功 (タグ上書きなし)
  - 再起動シナリオ: register 後 _load_from_disk で復元
  - 承認 preview: 未登録タグで「新タグ発明」フラグ挿入
  - 承認 preview: 登録済タグで「新タグ発明」フラグなし

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_memory_store_inline_tag.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.memory as memory_mod
import core.tag_registry as tr
from tools.memory_tool import _tool_memory_store
from core.approval_callback import _format_preview


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup(tmp_path: Path):
    memory_mod.MEMORY_DIR = tmp_path
    reg_file = tmp_path / "registered_tags.json"
    tr._reset_for_testing(registry_file=reg_file)
    tr.register_standard_tags()
    return reg_file


def test_inline_register_success(tmp_path: Path):
    print("== 未登録タグ + rules → register + append 成功 ==")
    _setup(tmp_path)
    result = _tool_memory_store({
        "network": "dream",
        "content": "夢で見た情景",
        "rules": {"beta_plus": False, "bitemporal": False},
        "display_format": "[dream] {content}",
        "tool_intent": "想像を残したい",
    })
    jsonl = tmp_path / "dream.jsonl"
    return all([
        _assert(tr.is_tag_registered("dream"), "dream 登録済"),
        _assert("記憶保存完了" in result, f"result に保存完了: {result[:60]}"),
        _assert(jsonl.exists(), "dream.jsonl 生成"),
        _assert(tr.get_tag_rules("dream")["origin"] == "dynamic", "origin=dynamic"),
        _assert(tr.get_tag_rules("dream")["intent"] == "想像を残したい", "intent 記録"),
    ])


def test_inline_without_rules_rejects(tmp_path: Path):
    print("== 未登録タグ + rules なし → エラー (未登録のまま) ==")
    _setup(tmp_path)
    result = _tool_memory_store({
        "network": "dream",
        "content": "夢",
        "tool_intent": "想像",
    })
    jsonl = tmp_path / "dream.jsonl"
    return all([
        _assert("rules 必須" in result, f"エラーメッセージ: {result[:80]}"),
        _assert(not tr.is_tag_registered("dream"), "タグ登録されない"),
        _assert(not jsonl.exists(), "jsonl 未書込"),
    ])


def test_existing_tag_ignores_rules(tmp_path: Path):
    print("== 登録済タグ + rules 指定 → rules 無視 (タグ上書きなし) ==")
    _setup(tmp_path)
    original_rules = tr.get_tag_rules("opinion")["learning_rules"]
    result = _tool_memory_store({
        "network": "opinion",
        "content": "信頼できそう",
        "confidence": "0.8",
        "rules": {"beta_plus": False, "bitemporal": True},  # 嘘 rules
        "tool_intent": "意見",
    })
    current_rules = tr.get_tag_rules("opinion")["learning_rules"]
    return all([
        _assert("記憶保存完了" in result, f"append 成功: {result[:60]}"),
        _assert(current_rules == original_rules, "標準タグの rules 保持"),
        _assert(tr.get_tag_rules("opinion")["origin"] == "standard", "origin=standard 保持"),
    ])


def test_persistence_after_reload(tmp_path: Path):
    print("== register 後、再 load で復元 ==")
    reg_file = _setup(tmp_path)
    _tool_memory_store({
        "network": "lesson",
        "content": "失敗から学ぶ",
        "rules": {"beta_plus": True, "bitemporal": False},
        "tool_intent": "教訓",
    })
    tr._reset_for_testing(registry_file=reg_file)
    return all([
        _assert(tr.is_tag_registered("lesson"), "再 load で lesson 復元"),
        _assert(tr.get_tag_rules("lesson")["intent"] == "教訓", "intent 復元"),
        _assert(tr.get_tag_rules("lesson")["learning_rules"]["beta_plus"] is True, "beta_plus 復元"),
    ])


def test_preview_new_tag_flag(tmp_path: Path):
    print("== 承認 preview: 未登録タグで「新タグ発明」フラグ挿入 ==")
    _setup(tmp_path)
    preview = _format_preview(
        "memory_store",
        {
            "network": "hypothesis",
            "content": "X が起きるかも",
            "rules": {"beta_plus": False, "bitemporal": False},
            "tool_intent": "仮説を残す",
            "tool_expected_outcome": "仮説登録",
            "message": "新カテゴリ作るね",
        },
        [],
    )
    return all([
        _assert("新タグ発明" in preview, "新タグ発明 フラグ"),
        _assert("hypothesis" in preview, "タグ名表示"),
        _assert("rules" in preview, "rules 表示"),
    ])


def test_preview_registered_tag_no_flag(tmp_path: Path):
    print("== 承認 preview: 登録済タグでフラグなし ==")
    _setup(tmp_path)
    preview = _format_preview(
        "memory_store",
        {
            "network": "opinion",
            "content": "信頼できそう",
            "confidence": "0.8",
            "tool_intent": "意見",
            "tool_expected_outcome": "保存",
            "message": "意見を残す",
        },
        [],
    )
    return _assert("新タグ発明" not in preview, "標準タグでフラグなし")


def test_preview_new_tag_without_rules(tmp_path: Path):
    print("== 承認 preview: 新タグ + rules 未指定 → handler reject 警告 ==")
    _setup(tmp_path)
    preview = _format_preview(
        "memory_store",
        {
            "network": "dream",
            "content": "夢",
            "tool_intent": "想像",
            "tool_expected_outcome": "保存",
            "message": "",
        },
        [],
    )
    return all([
        _assert("新タグ発明" in preview, "新タグ発明 フラグ"),
        _assert("rules 未指定" in preview, "rules 未指定 警告"),
    ])


def test_preview_other_tool_unaffected(tmp_path: Path):
    print("== 承認 preview: memory_store 以外のツールはフラグなし ==")
    _setup(tmp_path)
    preview = _format_preview(
        "write_file",
        {"path": "x.txt", "content": "y", "tool_intent": "a",
         "tool_expected_outcome": "b", "message": "c"},
        [],
    )
    return _assert("新タグ発明" not in preview, "他ツールは無影響")


def test_world_to_wm_redirect(tmp_path: Path):
    print("== 段階7 Step 6: network='world' → 'wm' リダイレクト ==")
    _setup(tmp_path)
    result = _tool_memory_store({
        "network": "world",
        "content": "デバイスは USB 接続",
        "entity_name": "device",
        "tool_intent": "世界知識",
    })
    world_jsonl = tmp_path / "world.jsonl"
    wm_jsonl = tmp_path / "wm.jsonl"
    return all([
        _assert("記憶保存完了" in result, f"リダイレクト後 append 成功: {result[:60]}"),
        _assert(not world_jsonl.exists(), "world.jsonl は作られない"),
        _assert(wm_jsonl.exists(), "wm.jsonl に書き込まれる"),
    ])


def run_all():
    print("=" * 60)
    print("test_memory_store_inline_tag.py (段階7 Step 5)")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        results = []
        for test_fn in [
            test_inline_register_success,
            test_inline_without_rules_rejects,
            test_existing_tag_ignores_rules,
            test_persistence_after_reload,
            test_preview_new_tag_flag,
            test_preview_registered_tag_no_flag,
            test_preview_new_tag_without_rules,
            test_preview_other_tool_unaffected,
            test_world_to_wm_redirect,
        ]:
            sub = Path(td) / test_fn.__name__
            sub.mkdir(exist_ok=True)
            results.append(test_fn(sub))
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"結果: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
