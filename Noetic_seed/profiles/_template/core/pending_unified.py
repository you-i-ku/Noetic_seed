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

import uuid
from datetime import datetime
from typing import Any, Literal, Optional, TypedDict


LagKind = Literal["seconds", "minutes", "hours", "cycles", "unknown"]
ExpiryPolicy = Literal["protected", "time", "dynamic_n", "deprecated"]


class MatchPattern(TypedDict, total=False):
    """段階8 v4 + 段階10.5 Fix 2: pending 側の自己消化条件。

    tool 側に rules を持たせず、pending が「誰が自分を消化できるか」を
    自己属性として持つ対称設計。全 tool が同じ hook (try_observe_all) で
    処理され、特別扱いゼロ。Active Inference 対称性:
    tool = 行動 → observation / pending = 期待 → match の分離。

    段階10.5 Fix 2 (案 P 確定、PLAN §4-2 新スキーマ):
      - 旧 tool_name_any (OR list) → source_action (単一 tool 名、構造 match)
      - 旧 channel_match (bool) → expected_channel (具体値、構造 match)
      - 旧 content_similarity_threshold → observable_similarity_threshold
        (比較対象を content → content_observable に切り替え、LLM 生成文面微差で
        merge すり抜ける現象を根治)

    全フィールド optional。None or 省略は「その条件は skip」。
    複数フィールド指定時は AND 判定。
    """
    # 消化できる tool 名 (単一)。None = どの tool でも OK
    source_action: Optional[str]
    # 消化時の channel (tool args.channel と一致判定)。None = channel 判定 skip
    expected_channel: Optional[str]
    # 意味類似度閾値 (tool_result vs pending.content_observable)。None = skip
    observable_similarity_threshold: Optional[float]


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

    # 段階10.5 Fix 2 (案 P): content 二層化 (PLAN §4-1)
    # - content_observable: 機械生成 (source_action + channel + cycle、match 用 what)
    # - content_intent: LLM 生成 (表示用 why、旧 content 相当)
    content_observable: str
    content_intent: str
    gap: float
    attempts: int
    priority: float
    semantic_merge: bool

    expiry_policy: ExpiryPolicy
    ttl_cycles: Optional[int]

    origin_cycle: int
    last_cycle: int
    timestamp: str

    # 段階8 v4: pending 側の自己消化条件 (None なら明示 dismiss のみで消える)
    match_pattern: Optional[MatchPattern]

    # 段階11-A Step 6: pending は iku の「自己 action への observation 待ち」なので
    # デフォルト self/actual。他視点 pending (例: 他者の action への帰属観察待ち
    # imaginary) は caller が perspective kwarg で指定。
    perspective: Optional[dict]


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

    形式: p_{session_id}_{cycle_id:04d}_{source_action[:8]}_{uuid6}
    例:   p_30c4d130_0008_listen_a_4f2a3b

    log entry id ({session_id}_{cycle_id:04d}) と同じ session+cycle prefix
    を含めることで、LLM が log 欄と pending 欄の対応関係を視覚的に追える。
    prefix "p_" で log entry id と明確に区別。

    段階8 hotfix (2026-04-20): 以前は `int(time.time() * 1000) % 1000` の ms 粒度
    で衝突可能 (同 ms 内に同 source_action で pending_add を連続呼出すと同一 ID)。
    `try_observe_all` の target_id 精密指定機構が id のユニーク性を前提にするため、
    uuid.uuid4().hex[:6] (約 1677 万通り) に変更して構造的に衝突を排除。
    """
    sa_short = str(source_action)[:8].replace(" ", "_")
    return f"p_{session_id}_{cycle_id:04d}_{sa_short}_{uuid.uuid4().hex[:6]}"


def _make_observable(source_action: str, channel: Optional[str],
                     cycle_id: int) -> str:
    """段階10.5 Fix 2: content_observable を機械生成 (PLAN §4-3)。

    f"{source_action} to channel={channel or 'self'} @ cycle {cycle_id}"

    目的: LLM 生成 intent 文字列の stochastic 文面微差で semantic merge が
    すり抜ける現象 (cycle 32 で cycle 26 再演) の根治。同 cycle 同 source_action
    同 channel なら observable は**完全一致**するため、merge が構造的に機能する。
    """
    ch = channel or "self"
    return f"{source_action} to channel={ch} @ cycle {cycle_id}"


def pending_add(
    state: dict,
    source_action: str,
    expected_observation: str,
    lag_kind: LagKind,
    content_intent: str,
    *,
    cycle_id: int,
    channel: Optional[str] = None,
    expiry_policy: ExpiryPolicy = "dynamic_n",
    ttl_cycles: Optional[int] = None,
    initial_gap: float = 1.0,
    semantic_merge: bool = False,
    retro_log_entry_id: Optional[str] = None,
    match_pattern: Optional[MatchPattern] = None,
    perspective: Optional[dict] = None,
) -> PendingEntry:
    """UPS v2 pending を state['pending'] に追加。

    段階10.5 Fix 2 (案 P、PLAN §4): content 二層化。
      - content_intent: 引数で受け取る LLM 生成テキスト (表示用、why)
      - content_observable: _make_observable() で機械生成 (match 用、what)

    Args:
        source_action: "output_display" / "elyth_post" / "living_presence" 等。
            AI のどの action が起点か (§3 の語彙)。
        expected_observation: 何の observation を待っているか
            (tool_expected_outcome 相当)。
        lag_kind: observation が来る想定時間スケール
            ("seconds" / "minutes" / "hours" / "cycles" / "unknown")。
        content_intent: human-readable summary (LLM 生成、表示用)。
            旧 content 引数から段階10.5 Fix 2 で rename。
        cycle_id: 現在 cycle。
        channel: observation の想定 channel ("device" / "elyth" / "x" / "self" / None)。
            None は「まだ特定されてない」を意味する (spontaneous 到着で後から埋まる)。
            content_observable 生成時は None → 'self' に fallback。
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
        "content_observable": _make_observable(source_action, channel, cycle_id),
        "content_intent": str(content_intent)[:500],
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
        "match_pattern": match_pattern,
    }
    # 段階11-A Step 6: perspective 付与 (None → default_self_perspective、iku の
    # 自己 action への observation 待ちがデフォルト)
    if perspective is None:
        from core.perspective import default_self_perspective
        perspective = default_self_perspective()
    entry["perspective"] = perspective
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
    target_id: Optional[str] = None,
) -> list[PendingEntry]:
    """observation 到着 → 該当 pending の gap 更新 + 遡及 E2 修正。

    observed_content が埋まっていない UPS v2 pending を priority 降順で
    並び替え、上位 limit 件に observation を紐付ける。
    match_source_actions が指定された場合はそれらに限定する。
    target_id が指定された場合は該当 id の pending のみを対象にする
    (段階8 hotfix: try_observe_all からの精密指定、同 source_action 複数
    pending が並立する場合の誤消化を防ぐ)。

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
        target_id: 特定の pending id のみを消化対象にする (段階8 hotfix)。
            None なら従来動作 (match_source_actions + priority 降順)。

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
    if target_id is not None:
        candidates = [p for p in candidates if p.get("id") == target_id]
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


PENDING_ATTEMPTS_SAFETY_CAP = 50  # 段階8: 強制 deprecate 閾値 (メモリ保護のみ)


def pending_prune(
    state: dict,
    current_cycle: int,
    *,
    dynamic_n: Optional[int] = None,
) -> int:
    """expiry_policy 別の淘汰。

    - "protected":  常に残す (外部由来の永続 pending)
    - "time":       ttl_cycles 経過で削除
    - "dynamic_n":  gap 上位 `dynamic_n` 件のみ残す
                    (None なら log 長ベース: max(3, min(20, log_count // 5)))
    - "deprecated": 段階8 安全弁 (attempts >= 50)。state に残すが
                    dynamic_n 競争外 = 実質選ばれない (prompt 表示フィルタは別経路)

    段階8 安全弁: attempts >= PENDING_ATTEMPTS_SAFETY_CAP で強制 deprecated
    マーク。**認知的諦めではなくメモリ保護のみ**、ランタイム異常防止。
    認知的諦めは iku の wait(dismiss) 明示行使に委ねる
    (feedback_freedom_to_die 整合)。

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

        # 安全弁: attempts が閾値超なら強制 deprecated マーク (memory protection)
        if (p.get("attempts", 1) >= PENDING_ATTEMPTS_SAFETY_CAP
                and p.get("expiry_policy") != "deprecated"):
            p["expiry_policy"] = "deprecated"

        policy = p.get("expiry_policy", "dynamic_n")
        if policy == "deprecated":
            # state に残すが dynamic_n 競争外
            survivors.append(p)
        elif policy == "protected":
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


# ============================================================
# 段階8: 外部入力 → 内部応答意図 (改善5 案 5-A)
# ============================================================

def pending_add_response_intent(
    state: dict,
    channel: str,
    text: str,
    cycle_id: int,
) -> PendingEntry:
    """外部入力受信時に iku の「応答する」内部意図を pending 化する helper。

    案 5-A 確定 (PLAN §4-1): 外部入力そのものを pending 化せず、
    iku の内部応答意図を unresolved_intent として一発生成。UPS v2.1
    一本化原則整合 (source_action は iku 内部 "response_to_external")。

    match_pattern は pending 側で「output_display (channel 一致時) で
    消化される」という自己消化条件を持つ (段階8 v4 対称設計)。
    応答しない自由は iku の wait(dismiss=p_xxx) 明示行使で実現
    (feedback_freedom_to_die 整合)。

    Args:
        state: 認知状態 dict。
        channel: 外部入力の channel ("device" / "claude" 等、応答先)。
        text: 外部入力の本文 (pending content のプレビューに使う)。
        cycle_id: 現在 cycle。

    Returns:
        生成された PendingEntry。
    """
    snippet = str(text)[:50]
    return pending_add(
        state=state,
        source_action="response_to_external",
        expected_observation="output_display 実行 (or wait dismiss)",
        lag_kind="cycles",
        content_intent=f"{channel} からの入力 '{snippet}...' への応答",
        cycle_id=cycle_id,
        channel=channel,
        expiry_policy="dynamic_n",
        semantic_merge=True,
        initial_gap=1.0,
        match_pattern={
            "source_action": "output_display",
            "expected_channel": channel,
        },
    )


# ============================================================
# 段階8 v4: pending 側 match_pattern 対称消化判定
# ============================================================

def _matches(
    mp: dict,
    tool_name: str,
    tool_args: dict,
    tool_result: str,
    channel: Optional[str],
    pending: dict,
) -> bool:
    """段階10.5 Fix 2 (案 P、PLAN §4-2): match_pattern 新構造 3 フィールドで判定。

    全条件 AND (未指定フィールドは skip)。

    Args:
        mp: pending.match_pattern (dict)。
        tool_name: 実行された tool 名。
        tool_args: tool の input args。
        tool_result: tool の output 文字列。
        channel: tool args.channel (or fallback "self")。
        pending: 対象 pending entry。

    Returns:
        全条件 OK なら True。
    """
    # 1. source_action: 消化できる tool 名 (単一) と一致するか
    required_source = mp.get("source_action")
    if required_source is not None:
        if tool_name != required_source:
            return False

    # 2. expected_channel: 消化時 channel と一致するか (構造 match)
    required_channel = mp.get("expected_channel")
    if required_channel is not None:
        if channel != required_channel:
            return False

    # 3. observable_similarity_threshold: content_observable との意味類似度判定
    # (段階10.5 Fix 2: 旧 content_similarity_threshold から rename、比較対象を
    # pending.content → pending.content_observable に切り替え。LLM 生成 intent
    # 文面微差で merge すり抜ける現象を根治)
    threshold = mp.get("observable_similarity_threshold")
    if threshold is not None:
        target_text = pending.get("content_observable", "")
        if not _sim_check(tool_result, target_text, float(threshold)):
            return False

    return True


def _sim_check(text_a: str, text_b: str, threshold: float) -> bool:
    """embedding cosine 類似度が threshold 以上なら True。

    embedding 不可 or 空文字列は False (安全側に倒す)。
    """
    if not text_a or not text_b:
        return False
    try:
        from core.embedding import _vector_ready, _embed_sync, cosine_similarity
    except ImportError:
        return False
    if not _vector_ready:
        return False
    try:
        vecs = _embed_sync([text_a[:200], text_b[:200]])
        if not vecs or len(vecs) != 2:
            return False
        sim = cosine_similarity(vecs[0], vecs[1])
        return sim >= threshold
    except Exception:
        return False


def try_observe_all(
    state: dict,
    tool_name: str,
    tool_args: dict,
    tool_result: str,
    channel: Optional[str],
    cycle_id: int,
) -> list[PendingEntry]:
    """全 pending を scan、match_pattern で自己消化判定しマッチすれば pending_observe 発火。

    段階8 v4 の対称設計: tool 側に rules を持たせず、pending 側が match_pattern
    で「誰が自分を消化できるか」を自己属性として持つ。全 tool が同じ hook で
    処理され、特別扱いゼロ。PostToolUse hook の Step 8 から呼ばれる。

    1 tool 実行あたり最大 1 pending を消化 (最初に match した pending のみ)。
    複数 match がある場合は priority 降順で選ぶ。

    Args:
        state: 認知状態 dict。
        tool_name: 実行された tool 名。
        tool_args: tool input args。
        tool_result: tool output 文字列。
        channel: tool args.channel or "self" fallback。
        cycle_id: 現在 cycle。

    Returns:
        消化された PendingEntry のリスト (通常 0 件 or 1 件)。
    """
    pending = state.get("pending", [])
    candidates = []
    for p in pending:
        if p.get("type") != "pending":
            continue
        if p.get("observed_content") is not None:
            continue  # 消化済 skip
        mp = p.get("match_pattern")
        if not mp:
            continue  # match_pattern なし = 明示 dismiss のみで消化
        if _matches(mp, tool_name, tool_args, tool_result, channel, p):
            candidates.append(p)

    if not candidates:
        return []

    # priority 降順で最初の 1 件を消化
    candidates.sort(key=lambda p: -float(p.get("priority", 0.0)))
    target = candidates[0]
    return pending_observe(
        state=state,
        observed_content=tool_result,
        channel=channel or "self",
        cycle_id=cycle_id,
        match_source_actions=[target["source_action"]],
        limit=1,
        target_id=target["id"],  # 段階8 hotfix: 同 source_action の別 pending 誤消化防止
    )


# ============================================================
# 段階10.5 Fix 2: 旧 pending 形式 drop migration
# ============================================================


def migrate_pending_observable_split(state: dict) -> int:
    """state["pending"] から旧形式 pending (content_observable 欠落) を drop。

    段階10.5 Fix 2 (案 P、PLAN §4-4): 旧形式 (`content` フィールドのみで
    `content_observable` を持たない) pending は、新スキーマの match 判定と
    整合しないため新 smoke 起点で drop。

    Returns:
        drop した pending 数 (0 なら migration 不要)
    """
    pending = state.get("pending")
    if not isinstance(pending, list):
        return 0
    survivors = [p for p in pending if "content_observable" in p]
    dropped = len(pending) - len(survivors)
    if dropped > 0:
        state["pending"] = survivors
    return dropped
