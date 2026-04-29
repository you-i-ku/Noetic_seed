"""LM Studio fragment-based reasoning split の test (段階12 補足-3 改、2026-04-29)。

`_call_lmstudio_native` の中で reasoning と最終応答を fragment.reasoning_type
で振り分けるロジックを mock で検証する。実 LM Studio サーバ不要。

reasoning_type の値 (SDK 仕様):
  None / "none"          → 通常応答 (= 最終出力)
  "reasoning"            → thinking 内容
  "reasoningStartTag"    → reasoning 区間の開始 tag
  "reasoningEndTag"      → reasoning 区間の終了 tag

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_lmstudio_reasoning_split.py
  (pytest tests/test_lmstudio_reasoning_split.py でも動く)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    assert cond, label


class _MockFragment:
    """SDK の LlmPredictionFragment 互換 mock。"""
    def __init__(self, content: str, reasoning_type=None):
        self.content = content
        self.reasoning_type = reasoning_type


def _make_split_callback():
    """`_call_lmstudio_native` の内部 callback ロジックを取り出した相当物。

    実装と同じ動作をする。実装変更時は両方同期させる必要あり。
    """
    content_parts: list = []
    reasoning_parts: list = []
    REASONING_TYPES = {"reasoning", "reasoningStartTag", "reasoningEndTag"}

    def _on_fragment(fragment):
        rtype = getattr(fragment, "reasoning_type", None)
        text = getattr(fragment, "content", "") or ""
        if rtype in REASONING_TYPES:
            reasoning_parts.append(text)
        else:
            content_parts.append(text)

    return _on_fragment, content_parts, reasoning_parts


def test_reasoning_only_excluded_from_content():
    """reasoning fragment のみ来た場合: content 空、reasoning に集まる。"""
    cb, content, reasoning = _make_split_callback()
    cb(_MockFragment("internal thought", "reasoning"))
    cb(_MockFragment(" continued", "reasoning"))
    _assert("".join(content) == "", "content 空")
    _assert(
        "".join(reasoning) == "internal thought continued",
        "reasoning は集める",
    )


def test_content_only_collected():
    """通常応答 fragment のみ: content に集まる、reasoning 空。"""
    cb, content, reasoning = _make_split_callback()
    cb(_MockFragment("hello", "none"))
    cb(_MockFragment(" world"))  # reasoning_type 未指定 (None)
    _assert("".join(content) == "hello world", "content フラグメント連結")
    _assert("".join(reasoning) == "", "reasoning なし")


def test_mixed_split_correctly():
    """混在パターン: reasoning と content が正しく振り分けられる。"""
    cb, content, reasoning = _make_split_callback()
    cb(_MockFragment("Here's a thinking process:", "reasoning"))
    cb(_MockFragment("analyzing intent...", "reasoning"))
    cb(_MockFragment("[TOOL:memory_store query=hi]", "none"))
    _assert(
        "".join(content) == "[TOOL:memory_store query=hi]",
        "content 部分のみ抽出",
    )
    _assert(
        "thinking process" in "".join(reasoning),
        "reasoning は別 layer に保持",
    )


def test_reasoning_boundary_tags_excluded():
    """reasoningStartTag / reasoningEndTag も reasoning 側に振り分けられる。"""
    cb, content, reasoning = _make_split_callback()
    cb(_MockFragment("<reasoning>", "reasoningStartTag"))
    cb(_MockFragment("inner thought", "reasoning"))
    cb(_MockFragment("</reasoning>", "reasoningEndTag"))
    cb(_MockFragment("final answer", "none"))
    _assert("".join(content) == "final answer", "boundary tag も content から除外")


def test_real_qwen3_pattern():
    """実 Qwen3.6 出力に近いパターン (英語自然言語 reasoning + 日本語応答)。"""
    cb, content, reasoning = _make_split_callback()
    cb(_MockFragment("Here's a thinking process:\n", "reasoning"))
    cb(_MockFragment("1. Analyze user input...\n", "reasoning"))
    cb(_MockFragment("2. Pick best tool...\n", "reasoning"))
    cb(_MockFragment(
        "1. システム状態確認 → output_display / predicted_e2: 70 / predicted_ec: 0.20\n",
        None,
    ))
    cb(_MockFragment(
        "2. 環境探索 → glob_search / predicted_e2: 80 / predicted_ec: 0.15",
        None,
    ))
    final = "".join(content)
    _assert(
        "Here's a thinking" not in final,
        f"reasoning 完全除去 (final: {final[:60]!r})",
    )
    _assert(
        "output_display" in final and "glob_search" in final,
        "candidate 形式は保持",
    )
    _assert(
        "Here's a thinking" in "".join(reasoning),
        "reasoning は debug log 保存用に残る",
    )


if __name__ == "__main__":
    print("=== test_lmstudio_reasoning_split ===")
    test_reasoning_only_excluded_from_content()
    test_content_only_collected()
    test_mixed_split_correctly()
    test_reasoning_boundary_tags_excluded()
    test_real_qwen3_pattern()
    print("=== all green ===")
