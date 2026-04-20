"""Session Compaction — claw-code 準拠。

claw-code 参照: rust/crates/runtime/src/compact.rs

責務:
  - session.messages のトークン概算
  - 閾値超過で should_compact() が True
  - compact_session() で古いメッセージを要約に差し替え

token カウントは tiktoken がなければ chars / 4 の素朴近似。
"""
from dataclasses import dataclass
from typing import Callable, Optional


DEFAULT_AUTO_COMPACT_THRESHOLD = 100_000
DEFAULT_KEEP_RECENT = 20  # 圧縮後に残す直近メッセージ数


@dataclass
class CompactionResult:
    summary: str
    removed_count: int
    kept_count: int
    tokens_before: int
    tokens_after: int


def estimate_session_tokens(session) -> int:
    """session.messages の合計 token を概算。

    tiktoken があれば利用、なければ chars/4 近似。
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        total = 0
        for m in session.messages:
            for b in (m.get("content") or []):
                if not isinstance(b, dict):
                    continue
                t = b.get("type")
                if t == "text":
                    total += len(enc.encode(b.get("text", "")))
                elif t == "tool_use":
                    import json
                    total += len(enc.encode(
                        str(b.get("name", "")) +
                        json.dumps(b.get("input") or {}, ensure_ascii=False)
                    ))
                elif t == "tool_result":
                    total += len(enc.encode(str(b.get("content", ""))))
        return total
    except ImportError:
        return _approx_tokens_chars4(session)


def _approx_tokens_chars4(session) -> int:
    total_chars = 0
    for m in session.messages:
        for b in (m.get("content") or []):
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                total_chars += len(b.get("text", ""))
            elif t == "tool_use":
                import json
                total_chars += len(b.get("name", ""))
                total_chars += len(json.dumps(b.get("input") or {},
                                              ensure_ascii=False))
            elif t == "tool_result":
                total_chars += len(str(b.get("content", "")))
    return total_chars // 4


def should_compact(session,
                   threshold: int = DEFAULT_AUTO_COMPACT_THRESHOLD) -> bool:
    return estimate_session_tokens(session) >= threshold


def get_compact_continuation_message() -> str:
    """圧縮完了後に LLM 側に渡す継続メッセージ (claw-code 準拠文字列)。"""
    return ("[session compacted: earlier messages summarized, "
            "continue the conversation with this context]")


def compact_session(session,
                    summarize_fn: Optional[Callable] = None,
                    keep_recent: int = DEFAULT_KEEP_RECENT) -> CompactionResult:
    """古いメッセージを要約に差し替える。

    summarize_fn: (messages: list) -> str。未指定なら素朴な head 要約。
    """
    before = estimate_session_tokens(session)
    msgs = session.messages
    if len(msgs) <= keep_recent:
        return CompactionResult(
            summary="", removed_count=0, kept_count=len(msgs),
            tokens_before=before, tokens_after=before,
        )

    old = msgs[:-keep_recent] if keep_recent > 0 else list(msgs)
    keep = msgs[-keep_recent:] if keep_recent > 0 else []

    if summarize_fn is not None:
        try:
            summary = summarize_fn(old)
        except Exception as e:
            summary = f"(summary failed: {e})"
    else:
        summary = _default_summarize(old)

    # 要約を先頭の user-text message として挿入
    new_head = {
        "role": "user",
        "content": [{
            "type": "text",
            "text": f"[compacted history]\n{summary}",
        }],
    }
    session.messages = [new_head] + keep

    after = estimate_session_tokens(session)
    return CompactionResult(
        summary=summary,
        removed_count=len(old),
        kept_count=len(keep),
        tokens_before=before,
        tokens_after=after,
    )


def _default_summarize(old_messages: list) -> str:
    """fallback 要約: 各 message の先頭を連結 + tool_use 名をリスト化。"""
    lines = []
    tool_calls: list = []
    for m in old_messages:
        role = m.get("role", "?")
        for b in (m.get("content") or []):
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                txt = b.get("text", "").strip()
                if txt:
                    lines.append(f"{role}: {txt[:120]}")
            elif t == "tool_use":
                tool_calls.append(b.get("name", "?"))
    summary_parts = []
    if lines:
        summary_parts.append("\n".join(lines[:30]))
    if tool_calls:
        from collections import Counter
        cnt = Counter(tool_calls)
        summary_parts.append("Tools used: " +
                             ", ".join(f"{n}×{c}" for n, c in cnt.most_common()))
    return "\n\n".join(summary_parts) or "(empty)"
