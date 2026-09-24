"""core.config.cap_tool_result テスト (段階9 Fix 5 の一般化 / 段階10 reserved 項目 4)

main.py の `str(rec.output)[:20000]` が marker なしで切っていた問題の修正確認。

背景 (2026-09-05 実観測):
  iku が read_file で state.json (156,778 字) を読んだ際、
  - file_ops.py は全文 + 正確なヘッダ `[state.json | lines 1-3555/3555]` を返す
  - main.py:874 が marker なしで 20,000 字に切る
  - ヘッダは先頭なので生き残り、iku は「全部読めた」と誤認する
  - しかも末尾は embedding の float 途中でブツ切り
  自己観測の土台が嘘をつく状態だった (feedback_action_observation_unified 違反)。

書式は tools/http_tool.py の _RESPONSE_MAX_CHARS 処理に揃えてある。

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_tool_result_cap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import cap_tool_result, TOOL_RESULT_MAX_CHARS as CAP

MARKER = "字に切詰。ツール実行時は完全取得済"


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# §1 上限以下: そのまま返る / marker を付けない
#
#   識別力: 「常に marker を付ける」誤実装をここで落とす。
#   境界 (len == CAP ちょうど) を含めるので、`>=` で判定する
#   off-by-one 誤実装もここで落ちる。
# ============================================================

def test_empty_input():
    print("== 空入力はそのまま ==")
    out = cap_tool_result("")
    return all([
        _assert(out == "", "空文字が返る"),
        _assert(MARKER not in out, "marker を付けない"),
    ])


def test_short_unchanged():
    print("== 上限未満は完全保存 ==")
    src = "a" * 100
    out = cap_tool_result(src)
    return all([
        _assert(out == src, "入力と完全一致"),
        _assert(MARKER not in out, "marker を付けない"),
    ])


def test_exactly_cap_unchanged():
    print("== 上限ちょうどは切らない (境界) ==")
    src = "b" * CAP
    out = cap_tool_result(src)
    return all([
        _assert(out == src, f"{CAP} 字がそのまま返る"),
        _assert(len(out) == CAP, "長さが変わらない"),
        _assert(MARKER not in out, "marker を付けない"),
    ])


# ============================================================
# §2 上限超過: 切る + 必ず marker を付ける
#
#   識別力: 元実装 `text[:20000]` はここで必ず fail する
#   (切りはするが marker がないため)。
#   また marker には「元の長さ」を書くので、切詰後の長さを
#   書いてしまう誤実装も content assertion で落ちる。
# ============================================================

def test_over_cap_by_one_truncated():
    print("== 上限+1 は切られる (境界) ==")
    src = "c" * (CAP + 1)
    out = cap_tool_result(src)
    return all([
        _assert(out != src, "そのままではない"),
        _assert(MARKER in out, "marker が付く"),
        _assert(out.startswith("c" * CAP), f"先頭 {CAP} 字は保存"),
    ])


def test_over_cap_has_marker_with_original_length():
    print("== marker に元の長さが入る ==")
    src = "d" * (CAP * 3)
    out = cap_tool_result(src)
    return all([
        _assert(MARKER in out, "marker が付く"),
        # 偽値 decoy: 切詰後の長さ (CAP) ではなく元の長さ (CAP*3) が書かれること
        _assert(f"{CAP}/{CAP * 3}字" in out, f"「{CAP}/{CAP*3}字」と表示 (元の長さ)"),
        _assert(f"{CAP}/{CAP}字" not in out, "切詰後の長さを元の長さとして書かない"),
    ])


def test_body_is_exactly_cap_chars():
    print("== 本体は上限ちょうどで切られる ==")
    src = "e" * (CAP * 2)
    out = cap_tool_result(src)
    body = out.split("\n...(表示上")[0]
    return all([
        _assert(len(body) == CAP, f"本体が {CAP} 字 (実測 {len(body)})"),
        _assert(set(body) == {"e"}, "本体は入力の先頭部分のみ"),
    ])


# ============================================================
# §3 実バグの再現: read_file ヘッダが生き残る状況
#
#   識別力: 「ヘッダは全量と言っているのに中身が切れている」
#   という実際に起きた誤認状況を、marker の有無で区別する。
#   marker なし実装ではこの test が fail する。
# ============================================================

def test_read_file_header_survives_but_marker_warns():
    print("== read_file ヘッダが残っても marker で切詰を伝える ==")
    header = "[read_file]\n[state.json | lines 1-3555/3555]\n"
    src = header + ("0.123456789, " * (CAP // 5))
    out = cap_tool_result(src)
    return all([
        _assert(len(src) > CAP, "前提: 入力は上限超過"),
        _assert("lines 1-3555/3555" in out, "ヘッダ自体は残る (実体は全量読めている)"),
        _assert(MARKER in out, "★ 切詰を marker で明示 (これが無いと iku が全量と誤認)"),
        _assert(f"/{len(src)}字" in out, "marker の分母が元の長さ"),
    ])


def test_multibyte_content_not_corrupted():
    print("== 日本語が含まれても marker が付く ==")
    src = "あ" * (CAP + 500)
    out = cap_tool_result(src)
    body = out.split("\n...(表示上")[0]
    return all([
        _assert(MARKER in out, "marker が付く"),
        _assert(len(body) == CAP, f"本体が {CAP} 文字 (bytes ではなく文字数)"),
        _assert(f"/{CAP + 500}字" in out, "marker の分母が元の文字数"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("空入力", test_empty_input),
        ("上限未満は不変", test_short_unchanged),
        ("上限ちょうど (境界)", test_exactly_cap_unchanged),
        ("上限+1 は切詰 (境界)", test_over_cap_by_one_truncated),
        ("marker に元の長さ", test_over_cap_has_marker_with_original_length),
        ("本体長は上限ちょうど", test_body_is_exactly_cap_chars),
        ("read_file ヘッダ + marker", test_read_file_header_survives_but_marker_warns),
        ("日本語 content", test_multibyte_content_not_corrupted),
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
