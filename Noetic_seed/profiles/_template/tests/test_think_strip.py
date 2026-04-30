"""think strip の動作 test (Qwen3 / DeepSeek-R1 系 reasoning model 対応)。

call_llm の post-process で <think>...</think> 区間を除去する `_strip_think`
の挙動を検証する。raw 保存 (append_debug_log) は i/o 副作用を伴うため本 test
の対象外、純粋関数 `_strip_think` のみ対象。

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_think_strip.py
  (pytest tests/test_think_strip.py でも動く)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm import _strip_think


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    assert cond, label


def test_no_think_unchanged():
    """think tag なしの text はそのまま透過する。"""
    text = "[TOOL:memory_store query=hello]"
    out = _strip_think(text)
    _assert(out == text, "think tag なし: text そのまま")


def test_single_think_removed():
    """think tag 1 個の場合、区間が除去されて think 後の text が残る。"""
    text = "<think>analyzing user intent</think>[TOOL:memory_store query=hi]"
    out = _strip_think(text)
    _assert(
        out == "[TOOL:memory_store query=hi]",
        f"think tag 1 個: 区間除去 (got: {out!r})",
    )


def test_multiple_think_all_removed():
    """think tag が複数あれば全て除去される (LLM が複数回思考した場合)。"""
    text = "<think>step1</think>action1\n<think>step2</think>action2"
    out = _strip_think(text)
    _assert(
        "<think>" not in out and "</think>" not in out,
        "think tag 複数: 全除去",
    )
    _assert(
        "action1" in out and "action2" in out,
        "think 外の応答は保持",
    )


def test_multiline_think_removed():
    """改行を含む長い think 区間も DOTALL で除去される。"""
    text = "<think>\nlong\nmultiline\nreasoning\nhere\n</think>\nfinal answer"
    out = _strip_think(text)
    _assert(
        "reasoning" not in out and "multiline" not in out,
        "複数行 think: 区間除去 (DOTALL 効いてる)",
    )
    _assert("final answer" in out, "think 後の応答は保持")


def test_unclosed_think_unchanged():
    """未閉じ <think> は除去されない (parse 失敗で気づく設計)。"""
    text = "<think>incomplete reasoning, no close tag"
    out = _strip_think(text)
    _assert(
        out == text,
        "未閉じ <think>: 透過 (debug 時に parse 失敗で発覚させる)",
    )


def test_empty_and_none():
    """空 text / None はそのまま透過する。"""
    _assert(_strip_think("") == "", "空 text: 空のまま")
    _assert(_strip_think(None) is None, "None: None のまま")


def test_real_qwen3_pattern():
    """実 Qwen3 出力に近いパターン (think の中に [TOOL: 文字を含む)。"""
    text = (
        "<think>\n"
        "ユーザーが memory_store を欲しがっているように見える。\n"
        "候補: [TOOL:memory_store ...] または [TOOL:reflect ...]\n"
        "memory_store を選ぶ。\n"
        "</think>\n"
        "[TOOL:memory_store key=greeting value=hi]"
    )
    out = _strip_think(text)
    # think 内の偽 [TOOL:...] が消える、本物だけ残る
    _assert(
        out.count("[TOOL:memory_store") == 1,
        f"think 内の偽 [TOOL: が消える、本物だけ残る (count: {out.count('[TOOL:memory_store')})",
    )


if __name__ == "__main__":
    print("=== test_think_strip ===")
    test_no_think_unchanged()
    test_single_think_removed()
    test_multiple_think_all_removed()
    test_multiline_think_removed()
    test_unclosed_think_unchanged()
    test_empty_and_none()
    test_real_qwen3_pattern()
    print("=== all green ===")
