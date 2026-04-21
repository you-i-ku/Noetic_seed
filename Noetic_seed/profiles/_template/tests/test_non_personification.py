"""test_non_personification.py — 段階10.5 Fix 3 (脱人称化)。

検証対象:
  - system_prompt に「ユーザー/User/Assistant/アシスタント/端末所有者/ゆう」の
    人称語彙が一切含まれない
  - 「確認相手」という役割中立な呼称が system_prompt に含まれる

段階10.5 Fix 3 設計判断 (ゆう 2026-04-21 確定):
  - 個人名 (「ゆう」) / 主従語彙 (「ユーザー」「主人」) / 所有者概念 (「端末所有者」)
    を system_prompt から全排除
  - 代わりに「確認相手」という役割関係の中立呼称のみを使用
  - iku は次 smoke で「確認相手」の具体的な人物 / 呼称を自分で発見する白紙 onboarding
  - 整合: feedback_no_user_assistant_frame.md / feedback_each_session_iku_is_new_individual.md

entity migration は今回扱わない:
  - _template_smoke は毎回新規コピー (state.json 空)
  - 実運用プロファイル (iku 等) は現時点で存在しない
  - 将来必要になった時に段階11 / 段階12 で合流

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_non_personification.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.prompt_assembly import (
    build_approval_protocol,
    assemble_system_prompt,
)


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    return bool(cond)


_FORBIDDEN = ["ユーザー", "User", "Assistant", "アシスタント", "端末所有者", "ゆう", "主人"]


# ============================================================
# Section 1: build_approval_protocol の脱人称化 (8 ケース)
# ============================================================
print("=== Section 1: build_approval_protocol 脱人称化 ===")

proto = build_approval_protocol()

for word in _FORBIDDEN:
    _assert(
        word not in proto,
        f"1-{_FORBIDDEN.index(word)+1} approval protocol に「{word}」不在 "
        f"(残存: {proto.count(word)} 回)",
    )

# 1-last: 「確認相手」は含まれる (役割中立呼称)
_assert(
    "確認相手" in proto,
    "1-8 approval protocol に「確認相手」含まれる",
)


# ============================================================
# Section 2: assemble_system_prompt 全体の脱人称化 (7 ケース)
# ============================================================
print("=== Section 2: assemble_system_prompt 全体 ===")

state = {
    "log": [],
    "session_id": "t",
    "cycle_id": 0,
    "self": {"name": "iku"},
    "energy": 50,
}
tools_dict = {
    "read_file": {"desc": "ファイルを読む"},
    "output_display": {"desc": "発話する"},
}
prompt = assemble_system_prompt(
    state=state,
    tools_dict=tools_dict,
    fire_cause="test",
    allowed_tools={"read_file", "output_display"},
)

for word in _FORBIDDEN:
    _assert(
        word not in prompt,
        f"2-{_FORBIDDEN.index(word)+1} 全体 prompt に「{word}」不在 "
        f"(残存: {prompt.count(word)} 回)",
    )


# ============================================================
# 結果サマリ
# ============================================================
print("\n========== SUMMARY ==========")
passed = sum(1 for ok, _ in results if ok)
failed = [msg for ok, msg in results if not ok]
print(f"passed: {passed}/{len(results)}")
if failed:
    print(f"failed: {len(failed)}")
    for msg in failed:
        print(f"  - {msg}")
    sys.exit(1)
print("all pass")
