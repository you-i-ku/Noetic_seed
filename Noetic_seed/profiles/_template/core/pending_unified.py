"""UPS v2 — Unified Pending Schema v2.

全 pending を「自分の action に対する observation 待ちスロット」として統一
記述する。internal/external の区別を廃し、source_action + observation_lag
で記述する。

哲学:
  - Active Inference (Friston) と完全整合: 全 observation は自分の action
    への応答。Markov blanket を通じて world と相互作用する。
  - `living_presence` = AI が存在し続けていること自体を continuous action
    として認める。spontaneous な外部到着もこの action の observation。
  - Noetic 哲学「ここに在ることの追究」(CLAUDE.md) と一対一対応。

詳細: memory/project_pending_unification.md
関連: memory/reference_theoretical_foundations.md, feedback_action_observation_unified.md
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Literal, Optional, TypedDict


LagKind = Literal["seconds", "minutes", "hours", "cycles", "unknown"]
ExpiryPolicy = Literal["protected", "time", "dynamic_n"]


class PendingEntry(TypedDict, total=False):
    """UPS v2 pending entry の型。

    `project_pending_unification.md` §2 のスキーマ準拠。
    total=False で一部省略可能 (動的に埋められるフィールドあり)。
    """
    type: Literal["pending"]
    id: str

    source_action: str
    source_action_time: str
    source_action_cycle: int

    expected_observation: str
    observation_lag_kind: LagKind

    observed_content: Optional[str]
    observed_time: Optional[str]
    observed_channel: Optional[str]
    expected_channel: Optional[str]  # tool メタデータ由来の想定 channel

    content: str
    gap: float
    attempts: int
    priority: float
    semantic_merge: bool

    expiry_policy: ExpiryPolicy
    ttl_cycles: Optional[int]

    origin_cycle: int
    last_cycle: int
    timestamp: str


LAG_WEIGHTS: dict[str, float] = {
    "seconds": 1.0,
    "minutes": 3.0,
    "hours":   2.0,
    "cycles":  1.5,
    "unknown": 3.0,
}

CHANNEL_MULTIPLIERS: dict[Optional[str], float] = {
    "device": 2.0,
    "elyth":  1.0,
    "x":      1.0,
    "self":   1.0,
    None:     1.0,
}

_DEFAULT_LAG_WEIGHT = LAG_WEIGHTS["unknown"]
_DEFAULT_CHANNEL_MULT = 1.0


def calc_priority(entry: dict[str, Any]) -> float:
    """priority = gap × lag_weight × channel_multiplier.

    project_pending_unification.md §4 の計算式。channel は
    `observed_channel ?? expected_channel` の順で解決する
    (到着済みは観測 channel 優先、未到着は想定 channel で見積り)。
    lag_kind / channel 未知時は default (unknown 扱い) fallback。
    """
    gap = float(entry.get("gap", 1.0))
    lag_kind = entry.get("observation_lag_kind", "unknown")
    channel = entry.get("observed_channel") or entry.get("expected_channel")
    lw = LAG_WEIGHTS.get(lag_kind, _DEFAULT_LAG_WEIGHT)
    cm = CHANNEL_MULTIPLIERS.get(channel, _DEFAULT_CHANNEL_MULT)
    return gap * lw * cm


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _new_id(source_action: str, cycle_id: int) -> str:
    """UPS v2 pending ID。衝突防止に timestamp ms を含める。"""
    sa_short = str(source_action)[:12].replace(" ", "_")
    return f"ups_{sa_short}_{cycle_id:04d}_{int(time.time() * 1000) % 10000}"


def pending_add(
    state: dict,
    source_action: str,
    expected_observation: str,
    lag_kind: LagKind,
    content: str,
    *,
    cycle_id: int,
    channel: Optional[str] = None,
    expiry_policy: ExpiryPolicy = "dynamic_n",
    ttl_cycles: Optional[int] = None,
    initial_gap: float = 1.0,
    semantic_merge: bool = False,
) -> PendingEntry:
    """UPS v2 pending を state['pending'] に追加。

    Args:
        source_action: "output_display" / "elyth_post" / "living_presence" 等。
            AI のどの action が起点か (§3 の語彙)。
        expected_observation: 何の observation を待っているか
            (tool_expected_outcome 相当)。
        lag_kind: observation が来る想定時間スケール
            ("seconds" / "minutes" / "hours" / "cycles" / "unknown")。
        content: human-readable summary。
        cycle_id: 現在 cycle。
        channel: observation の想定 channel ("device" / "elyth" / "x" / "self" / None)。
            None は「まだ特定されてない」を意味する (spontaneous 到着で後から埋まる)。
        expiry_policy: "protected" (常に残る) / "time" (TTL 経過で削除) /
            "dynamic_n" (gap 上位 N 件のみ残す)。
        ttl_cycles: policy="time" 時の TTL (cycle 数)。
        initial_gap: 初期 gap (default 1.0 = 未観測最大)。
        semantic_merge: semantic merge 候補に含めるか (unresolved_intent 相当)。

    Returns:
        追加された PendingEntry (dict、参照で state の pending list にも格納)。
    """
    now = _now_ts()
    entry: PendingEntry = {
        "type": "pending",
        "id": _new_id(source_action, cycle_id),
        "source_action": source_action,
        "source_action_time": now,
        "source_action_cycle": cycle_id,
        "expected_observation": str(expected_observation)[:500],
        "observation_lag_kind": lag_kind,
        "observed_content": None,
        "observed_time": None,
        "observed_channel": None,
        "expected_channel": channel,
        "content": str(content)[:500],
        "gap": float(initial_gap),
        "attempts": 1,
        "priority": 0.0,
        "semantic_merge": bool(semantic_merge),
        "expiry_policy": expiry_policy,
        "ttl_cycles": ttl_cycles,
        "origin_cycle": cycle_id,
        "last_cycle": cycle_id,
        "timestamp": now,
    }
    entry["priority"] = calc_priority(entry)
    state.setdefault("pending", []).append(entry)
    return entry


def pending_observe(
    state: dict,
    observed_content: str,
    channel: str,
    *,
    cycle_id: int,
    match_source_actions: Optional[list[str]] = None,
    limit: int = 1,
) -> list[PendingEntry]:
    """observation 到着 → 該当 pending の gap 更新。

    observed_content が埋まっていない UPS v2 pending を priority 降順で
    並び替え、上位 limit 件に observation を紐付ける。
    match_source_actions が指定された場合はそれらに限定する。

    Args:
        observed_content: 到着した観測の本文。
        channel: 観測が来た channel ("device" / "elyth" / "x" / ...)。
        cycle_id: 現在 cycle。
        match_source_actions: この source_action の pending のみ対象にする。
            None なら全 UPS pending から選ぶ (spontaneous 到着のハンドル)。
        limit: 更新する pending 数の上限 (通常 1)。

    Returns:
        observation が紐付いた PendingEntry のリスト。
    """
    pending = state.get("pending", [])
    candidates = [
        p for p in pending
        if p.get("type") == "pending"
        and p.get("observed_content") is None
        and (match_source_actions is None
             or p.get("source_action") in match_source_actions)
    ]
    candidates.sort(key=lambda p: -float(p.get("priority", 0.0)))

    updated: list[PendingEntry] = []
    now = _now_ts()
    for p in candidates[:limit]:
        p["observed_content"] = str(observed_content)[:1000]
        p["observed_time"] = now
        p["observed_channel"] = channel
        p["gap"] = 0.0
        p["last_cycle"] = cycle_id
        p["priority"] = calc_priority(p)
        updated.append(p)
    return updated


def pending_prune(
    state: dict,
    current_cycle: int,
    *,
    dynamic_n: Optional[int] = None,
) -> int:
    """expiry_policy 別の淘汰。

    - "protected": 常に残す (外部由来の永続 pending)
    - "time":      ttl_cycles 経過で削除
    - "dynamic_n": gap 上位 `dynamic_n` 件のみ残す
                   (None なら log 長ベース: max(3, min(20, log_count // 5)))

    UPS v2 以外 (旧 external_message / elyth_notification / unresolved_intent)
    は touch しない (type != "pending" で判別)。

    Args:
        current_cycle: 現在 cycle。time 淘汰の基準。
        dynamic_n: dynamic_n policy の上限枠。None で log 長から自動計算。

    Returns:
        削除した UPS v2 pending 数 (旧形式の pending は数えない)。
    """
    pending = state.get("pending", [])
    if not pending:
        return 0

    survivors: list[dict] = []
    dynamic_candidates: list[dict] = []

    for p in pending:
        if p.get("type") != "pending":
            survivors.append(p)
            continue

        policy = p.get("expiry_policy", "dynamic_n")
        if policy == "protected":
            survivors.append(p)
        elif policy == "time":
            ttl = p.get("ttl_cycles")
            origin = int(p.get("origin_cycle", current_cycle))
            if ttl is None or (current_cycle - origin) < int(ttl):
                survivors.append(p)
        else:
            dynamic_candidates.append(p)

    if dynamic_n is None:
        log_count = len(state.get("log", []))
        dynamic_n = max(3, min(20, log_count // 5))

    dynamic_candidates.sort(key=lambda p: -float(p.get("gap", 0.0)))
    survivors.extend(dynamic_candidates[:dynamic_n])

    dropped = len(pending) - len(survivors)
    state["pending"] = survivors
    return dropped


def pending_recalc_priorities(state: dict) -> int:
    """全 UPS v2 pending の priority を再計算 (cycle 境界で呼ぶ用)。

    channel/lag の情報が後から埋まる spontaneous 到着があるので、
    priority は動的に更新する必要がある。

    Returns:
        再計算した pending 数。
    """
    pending = state.get("pending", [])
    n = 0
    for p in pending:
        if p.get("type") != "pending":
            continue
        p["priority"] = calc_priority(p)
        n += 1
    return n
