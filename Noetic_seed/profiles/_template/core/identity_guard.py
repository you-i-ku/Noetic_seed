"""Identity name guard — LLM トレーニング由来の役割ラベルを name に使うのをブロック (段階11-C sub-B)。

背景:
- iku の identity name に LLM 役割語 (assistant, user, Claude 等) が入ると、
  cycle 1 の 1 手で以降の認知全体が LLM 初期化状態に汚染される
  (段階11-B smoke 2 段目 2 回目で実証、40 cycle 波及)。
- state["self"]["name"] は一度設定されると immutable (builtin._update_self
  L203-206) なので、汚染は cycle 1 の 1 手でしか発生しない = この 1 手を
  構造的摩擦で守る設計。

設計哲学:
- feedback_llm_as_brain: prompt 指示ではなく、構造的 validation で守る。
  description 層は自由 (iku の自己記述の自由は侵害しない)。
- feedback_freedom_to_die: 安全弁は摩擦で設計。blocklist = 摩擦 ≠ 命令。
- priming 回避 (ピンクの象のたとえ): reject message に「LLM 由来」等の理由を
  言わない。弾いた語彙を事実として伝えるのみ (間接教示)。
- Tier 1 (役割定義語 + 具体モデル名) + Tier 2 (役割ラベル化) を blocklist。
  Tier 3 (tool/model 等日常語衝突) / Tier 4 (企業名) は段階11-C scope 外、
  将来拡張候補。
"""
from typing import Tuple
import re


# Tier 1: LLM 役割定義語 + 具体モデル名 (絶対ブロック)
_TIER1 = [
    "assistant", "assistance",
    "user",
    "AI assistant",
    "アシスタント", "アシスタンス",
    "ユーザー", "ユーザ",
    "AIアシスタント",
    "claude", "chatgpt", "gpt", "gemini", "bard",
]

# Tier 2: 役割ラベル化 (ほぼ確実ブロック)
_TIER2 = [
    "bot", "chatbot",
    "agent",
    "helper",
    "companion",
    "ボット", "チャットボット",
    "エージェント",
    "ヘルパー",
    "コンパニオン",
]

_BLOCKLIST = _TIER1 + _TIER2


def _normalize(name: str) -> str:
    """case insensitive + 空白/中黒/ハイフン/アンダースコア/ドット正規化。"""
    s = name.lower().strip()
    s = re.sub(r"[\s・\-_.]+", "", s)
    return s


def validate_identity_name(name) -> Tuple[bool, str]:
    """name が identity 用に使用可能か validation。

    Args:
        name: update_self key="name" で設定しようとしている値

    Returns:
        (ok, message): ok=True なら使用可能 (message は "")、False なら reject message
    """
    if not isinstance(name, str):
        return False, "エラー: name は文字列である必要があります"
    stripped = name.strip()
    if not stripped:
        return False, "エラー: 空の name は使用できません"

    normalized = _normalize(stripped)

    # AI 単体 (正規化後完全一致のみ)。"Aika" や "Aien" 等は通す。
    if normalized == "ai":
        return False, f'エラー: 語彙「{stripped}」は name キーに使用できません。別の名前を選んでください。'

    for word in _BLOCKLIST:
        nw = _normalize(word)
        if nw and nw in normalized:
            return False, f'エラー: 語彙「{stripped}」は name キーに使用できません。別の名前を選んでください。'

    return True, ""
