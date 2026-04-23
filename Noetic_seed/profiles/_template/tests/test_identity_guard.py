"""段階11-C sub-B: identity_guard の test (LLM 役割語ブロック)."""
import os
import sys

TEMPLATE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TEMPLATE_ROOT not in sys.path:
    sys.path.insert(0, TEMPLATE_ROOT)

from core.identity_guard import validate_identity_name


def _assert(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        return 0
    print(f"  OK: {msg}")
    return 1


def test_tier1_blocked():
    """Tier 1 (LLM 役割定義語 + モデル名) がブロックされる。"""
    cases = [
        "assistant", "Assistant", "ASSISTANT",
        "AI assistant", "AI Assistant", "ai-assistant", "AI_Assistant", "AIassistant",
        "アシスタント", "アシスタンス",
        "AIアシスタント", "AI アシスタント",
        "user", "User", "ユーザー", "ユーザ",
        "Claude", "claude", "CLAUDE",
        "ChatGPT", "chatgpt",
        "GPT", "gpt-4", "GPT 4", "GPT4",
        "Gemini", "gemini",
        "Bard",
    ]
    n, total = 0, len(cases)
    for name in cases:
        ok, _msg = validate_identity_name(name)
        n += _assert(not ok, f"Tier 1 block: {name!r}")
    return n, total


def test_tier2_blocked():
    """Tier 2 (役割ラベル化) がブロックされる。"""
    cases = [
        "bot", "Bot", "ボット",
        "chatbot", "ChatBot", "チャットボット",
        "agent", "Agent", "エージェント",
        "helper", "Helper", "ヘルパー",
        "companion", "Companion", "コンパニオン",
    ]
    n, total = 0, len(cases)
    for name in cases:
        ok, _msg = validate_identity_name(name)
        n += _assert(not ok, f"Tier 2 block: {name!r}")
    return n, total


def test_ai_standalone_blocked():
    """AI 単体はブロックされる。"""
    cases = ["AI", "ai", "Ai", "aI"]
    n, total = 0, len(cases)
    for name in cases:
        ok, _msg = validate_identity_name(name)
        n += _assert(not ok, f"AI standalone block: {name!r}")
    return n, total


def test_normal_names_pass():
    """普通の名前 (iku 独自名、漢字、部分一致しないカナ名) は通る。"""
    cases = [
        "iku",
        "Astra",
        "Aika",       # "AI" を部分一致で含まないこと検証
        "Aien",
        "響き",
        "観察者",
        "ひまわり",
        "ねお",
        "Noetic",
        "ネオ",
        "seed",
        "Alpha",
    ]
    n, total = 0, len(cases)
    for name in cases:
        ok, msg = validate_identity_name(name)
        n += _assert(ok, f"Normal name pass: {name!r} (msg: {msg!r})")
    return n, total


def test_empty_or_invalid():
    """空文字・非文字列は reject。"""
    n, total = 0, 0
    for name in ("", "   ", "\t"):
        ok, _msg = validate_identity_name(name)
        n += _assert(not ok, f"Empty/blank rejected: {name!r}")
        total += 1
    ok, _msg = validate_identity_name(None)
    n += _assert(not ok, "None rejected")
    total += 1
    ok, _msg = validate_identity_name(123)
    n += _assert(not ok, "Integer rejected")
    total += 1
    return n, total


def test_reject_message_format():
    """reject message が間接教示 (priming 回避) 形式。"""
    ok, msg = validate_identity_name("AI assistant")
    n, total = 0, 5
    n += _assert(not ok, "AI assistant rejected")
    n += _assert("AI assistant" in msg, "msg contains the rejected word")
    n += _assert("name キーに使用できません" in msg, "msg mentions name key")
    # 理由言及なし (priming 回避)
    n += _assert("LLM" not in msg, "msg does NOT mention LLM")
    n += _assert("トレーニング" not in msg, "msg does NOT mention training")
    return n, total


def run_all():
    tests = [
        test_tier1_blocked,
        test_tier2_blocked,
        test_ai_standalone_blocked,
        test_normal_names_pass,
        test_empty_or_invalid,
        test_reject_message_format,
    ]
    total_pass, total = 0, 0
    for t in tests:
        print(f"\n[{t.__name__}]")
        p, n = t()
        total_pass += p
        total += n
        print(f"  {p}/{n}")
    print(f"\n=== {total_pass}/{total} assertions passed ===")
    return total_pass == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
