"""Token / Cost Tracking — claw-code 準拠。

claw-code 参照:
  - rust/crates/runtime/src/usage.rs
  - rust/crates/runtime/src/cost_tracker.rs

長時間稼働で provider API 使用量を追跡する。
"""
from dataclasses import dataclass, field
from typing import Optional


# USD/1M tokens の概算 (2026-04 時点。実値は provider 側で変動)。
_PRICING_PER_MTOK: dict = {
    # Anthropic
    "claude-opus-4-6":        (15.0, 75.0),   # (input, output)
    "claude-sonnet-4-6":      (3.0, 15.0),
    "claude-haiku-4-5":       (0.25, 1.25),
    "claude-haiku-4-5-20251213": (0.25, 1.25),
    # xAI
    "grok-3":       (5.0, 15.0),
    "grok-3-mini":  (0.3, 0.5),
    # OpenAI (参考値)
    "gpt-4o":       (2.5, 10.0),
    "gpt-4o-mini":  (0.15, 0.6),
    # Gemini (参考値)
    "gemini-1.5-pro":   (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.3),
}


def pricing_for_model(model: str) -> Optional[tuple]:
    """(input_usd_per_mtok, output_usd_per_mtok) or None。"""
    if model in _PRICING_PER_MTOK:
        return _PRICING_PER_MTOK[model]
    # プレフィックスマッチ
    for key, val in _PRICING_PER_MTOK.items():
        if model.startswith(key):
            return val
    return None


def max_tokens_for_model(model: str) -> int:
    """おおよその max output tokens。"""
    m = model.lower()
    if "opus" in m:
        return 32_000
    if "sonnet" in m:
        return 64_000
    if "haiku" in m:
        return 64_000
    if "grok-3-mini" in m:
        return 64_000
    if "grok" in m:
        return 64_000
    return 4_096  # default


@dataclass
class UsageSummary:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    request_count: int = 0
    models_used: list = field(default_factory=list)

    def add(self, usage_dict: dict,
            model: Optional[str] = None) -> None:
        """provider 応答の usage dict をマージ。"""
        if not usage_dict:
            return
        # Anthropic / OpenAI の異なる key 名を両方吸収
        self.input_tokens += int(
            usage_dict.get("input_tokens")
            or usage_dict.get("prompt_tokens") or 0
        )
        self.output_tokens += int(
            usage_dict.get("output_tokens")
            or usage_dict.get("completion_tokens") or 0
        )
        self.cache_read_tokens += int(
            usage_dict.get("cache_read_input_tokens") or 0
        )
        self.cache_creation_tokens += int(
            usage_dict.get("cache_creation_input_tokens") or 0
        )
        self.request_count += 1
        if model and model not in self.models_used:
            self.models_used.append(model)

    def estimate_cost_usd(self,
                          model: Optional[str] = None) -> Optional[float]:
        """model が分かっていれば USD 概算。未知モデルなら None。"""
        mdl = model or (self.models_used[0] if self.models_used else None)
        if not mdl:
            return None
        pricing = pricing_for_model(mdl)
        if pricing is None:
            return None
        in_price, out_price = pricing
        return ((self.input_tokens + self.cache_read_tokens) / 1_000_000
                * in_price
                + self.output_tokens / 1_000_000 * out_price)


class CostTracker:
    """複数ターンの usage を蓄積。"""

    def __init__(self):
        self.summary = UsageSummary()

    def record(self, usage_dict: Optional[dict],
               model: Optional[str] = None) -> None:
        if usage_dict:
            self.summary.add(usage_dict, model=model)

    def report(self) -> str:
        s = self.summary
        lines = [
            f"Requests: {s.request_count}",
            f"Tokens: in={s.input_tokens} out={s.output_tokens} "
            f"cache_read={s.cache_read_tokens} "
            f"cache_create={s.cache_creation_tokens}",
        ]
        if s.models_used:
            lines.append(f"Models: {', '.join(s.models_used)}")
            cost = s.estimate_cost_usd(s.models_used[0])
            if cost is not None:
                lines.append(f"Estimated cost: ${cost:.4f}")
        return "\n".join(lines)
