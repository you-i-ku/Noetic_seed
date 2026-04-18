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

    # 遡及 E2 修正用 (pending_feedback 吸収、§6 "遡及 E2" の自動発火対象)
    # action 実行時の log entry id を保持し、observation 到着で該当 entry の
    # e2 を上方修正する (従来 pending_feedback の +40% 挙動を UPS に統合)
    retro_log_entry_id: Optional[str]

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


def _new_id(source_action: str, cycle_id: int, session_id: str = "x") -> str:
    """UPS v2 pending ID。

    形式: p_{session_id}_{cycle_id:04d}_{source_action[:8]}_{ms%1000}
    例:   p_30c4d130_0008_listen_a_123

    log entry id ({session_id}_{cycle_id:04d}) と同じ session+cycle prefix
    を含めることで、LLM が log 欄と pending 欄の対応関係を視覚的に追える。
    prefix "p_" で log entry id と明確に区別。
    """
    sa_short = str(source_action)[:8].replace(" ", "_")
    return f"p_{session_id}_{cycle_id:04d}_{sa_short}_{int(time.time() * 1000) % 1000}"


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
    retro_log_entry_id: Optional[str] = None,
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
        retro_log_entry_id: 遡及 E2 修正の対象 log entry id。
            observation 到着時に pending_observe で該当 log entry の e2 を
            上方修正する (§6 "pending_feedback 遅延 E2 遡及"の吸収)。

    Returns:
        追加された PendingEntry (dict、参照で state の pending list にも格納)。
    """
    now = _now_ts()
    session_id = state.get("session_id", "x")
    entry: PendingEntry = {
        "type": "pending",
        "id": _new_id(source_action, cycle_id, session_id),
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
        "retro_log_entry_id": retro_log_entry_id,
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
    retro_e2_bonus: int = 40,
) -> list[PendingEntry]:
    """observation 到着 → 該当 pending の gap 更新 + 遡及 E2 修正。

    observed_content が埋まっていない UPS v2 pending を priority 降順で
    並び替え、上位 limit 件に observation を紐付ける。
    match_source_actions が指定された場合はそれらに限定する。

    §6 "pending_feedback 遅延 E2 遡及" の吸収: observe 対象 pending が
    `retro_log_entry_id` を持つ場合、state.log 中の該当 entry の `e2` を
    +retro_e2_bonus% 上方修正 (上限 100%)。

    Args:
        observed_content: 到着した観測の本文。
        channel: 観測が来た channel ("device" / "elyth" / "x" / ...)。
        cycle_id: 現在 cycle。
        match_source_actions: この source_action の pending のみ対象にする。
            None なら全 UPS pending から選ぶ (spontaneous 到着のハンドル)。
        limit: 更新する pending 数の上限 (通常 1)。
        retro_e2_bonus: 遡及 E2 修正の bonus 値 (% 単位)。default 40 は
            旧 pending_feedback の既存挙動を継承。0 で無効化。

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
        # 遡及 E2 修正 (旧 pending_feedback 相当の自動発火)
        log_id = p.get("retro_log_entry_id")
        if log_id and retro_e2_bonus > 0:
            _apply_retro_e2(state, log_id, retro_e2_bonus)
        updated.append(p)
    return updated


def _apply_retro_e2(state: dict, log_entry_id: str, bonus_pct: int) -> bool:
    """指定 log entry の `e2` を `+bonus_pct%` 遡及修正 (上限 100%)。

    旧 pending_feedback の `_resolved["status"] = "resolved"` + e2 上方修正
    ロジックを UPS 側に吸収したもの。該当 log entry が見つからない、または
    既存 e2 が数値パース不可なら何もせず False を返す (defensive)。

    Returns:
        修正を実行した場合 True、対象 entry 無し or e2 未設定の場合 False。
    """
    import re
    for entry in state.get("log", []):
        if entry.get("id") != log_entry_id:
            continue
        old_e2 = entry.get("e2", "")
        m = re.search(r"(\d+)", str(old_e2))
        if not m:
            return False
        new_val = min(100, int(m.group(1)) + int(bonus_pct))
        entry["e2"] = f"{new_val}%"
        return True
    return False


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
