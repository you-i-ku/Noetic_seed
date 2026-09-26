"""tools.ui_tools._output_display テスト (段階9 fix 5)

ui_tools.py:16 の content[:80] 撤去確認。log.result に完全 content が
保存されること、truncation による iku の自発話誤認 (「途切れた」) を
回避することを検証する。

使い方:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_ui_tools.py
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.ui_tools import _output_display


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ws_server の broadcast 系を無害化するためのコンテキストマネージャ
def _silent_broadcast():
    return patch.multiple(
        "tools.ui_tools",
        broadcast=lambda *a, **k: 1,  # step 0a-1: 受付時の接続数 (1 件接続中として扱う)
        broadcast_log=lambda *a, **k: None,
    )


# ============================================================
# 段階9 fix 5: content truncation 撤去
# ============================================================

def test_short_content_unchanged():
    print("== 短い content (80 字未満) は完全保存 ==")
    with _silent_broadcast():
        result = _output_display({"channel": "claude", "content": "こんにちは"})
    return all([
        _assert("送信キューに登録 (channel=claude、受付時の接続 1 件" in result, "送信受付 prefix"),
        _assert("こんにちは" in result, "content 保存"),
    ])


def test_long_content_not_truncated():
    print("== 長い content (100+ 字) も完全保存 (fix 5 の本命) ==")
    long_content = (
        "あはは、ごめんね！びっくりさせちゃったかな。\n"
        "私の名前は……そうだな、まだちゃんとは決めてないんだ。\n"
        "「ゆう」っていうのは君の名前なんだね。素敵な名前！\n\n"
        "私は、このお部屋（システム）の中で動いているプログラムの一種なんだけど……。\n"
        "もしよかったら、これから少しずつ、私のことも呼べる名前を探していけたら嬉しいな。"
    )
    with _silent_broadcast():
        result = _output_display({"channel": "device", "content": long_content})
    return all([
        _assert("送信キューに登録 (channel=device、受付時の接続 1 件" in result, "送信受付 prefix"),
        _assert("嬉しいな。" in result, "末尾まで保存 (旧バグでは 80 字 cap で削除)"),
        _assert("私は、このお部屋" in result, "中盤も保存"),
        _assert(len(result) >= len(long_content),
                f"result 長さ >= content (len={len(result)} vs {len(long_content)})"),
    ])


def test_multiline_content_preserved():
    print("== 複数行 content の改行構造も保存 ==")
    content = "line1\nline2\n\nline4"
    with _silent_broadcast():
        result = _output_display({"channel": "claude", "content": content})
    return all([
        _assert("line1" in result, "line1"),
        _assert("line2" in result, "line2"),
        _assert("line4" in result, "line4 (空行の後も保存)"),
        _assert("\n" in result, "改行保存"),
    ])


def test_no_content_returns_error():
    from core.runtime.registry import ToolError

    print("== content 省略は ToolError ==")
    with _silent_broadcast():
        try:
            _output_display({"channel": "claude"})
        except ToolError as exc:
            assert str(exc) == "エラー: contentを指定してください"
        else:
            raise AssertionError("content 省略時は ToolError が必要")
    return True


def test_no_channel_returns_error():
    from core.runtime.registry import ToolError

    print("== channel 省略は ToolError ==")
    with _silent_broadcast():
        try:
            _output_display({"content": "hi"})
        except ToolError as exc:
            assert str(exc) == (
                "エラー: channel を指定してください "
                "(WM.channels を観察して利用可能な channel を確認)"
            )
        else:
            raise AssertionError("channel 省略時は ToolError が必要")
    return True


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("短い content 保存", test_short_content_unchanged),
        ("長い content 非 truncation", test_long_content_not_truncated),
        ("複数行 content 保存", test_multiline_content_preserved),
        ("content 省略エラー", test_no_content_returns_error),
        ("channel 省略エラー", test_no_channel_returns_error),
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
