"""プロンプト構築（propose/execute）+ 注意機構 + 鮮度勾配パッキング"""
import json
import re
from datetime import datetime
from core.config import prompt_budget, estimate_tokens
from core.embedding import is_vector_ready, _embed_sync, cosine_similarity

N_PROPOSE = 5
ATTENTION_RECENT = 10  # 直近N件は無条件で含める
ATTENTION_SIMILAR = 10  # 類似度上位N件を追加


def _tier_cap(pos_from_end: int, boundaries: list, caps: list) -> int:
    """鮮度位置（0=最新）から result cap を決定する。boundaries の各閾値未満ならその tier の cap を返す。"""
    for idx, b in enumerate(boundaries):
        if pos_from_end < b:
            return caps[idx]
    return caps[-1]


def _render_log_entry(entry: dict, result_cap: int, intent_cap: int, with_evals: bool = False) -> str:
    """1件の log エントリを鮮度勾配 cap 付きで 1 行レンダリングする。
    result が cap を超えた場合は明示的な truncation marker を付けて AI に
    「表示上の省略であって、ツール実行時は完全に取得済み」と伝える。

    段階8 改善1+3:
    - args フィールドが entry にあれば intent の前に表示 (cap 200、長ければ "..." 省略)
    - result に "[REJECTED]" が含まれる場合、行頭に "⚠️" prefix を付与して視覚強調
    """
    result = entry.get("result", "") or ""
    result_str = str(result)
    is_rejected = "[REJECTED]" in result_str

    prefix = "⚠️ " if is_rejected else "  "
    _ch = entry.get("channel", "")
    # 段階9 fix 2-a: [channel=X] 形式で明示。LLM が知識 (WM の channels) と
    # 行動 (tool.args.channel に同じ値を渡す) を繋げやすくする。
    _ch_tag = f"[channel={_ch}] " if _ch else ""
    line = f"{prefix}{entry.get('id','')} {entry['time']} {_ch_tag}{entry['tool']}"

    # 段階8 改善1: args 表示 (intent より前、cap 200)
    args = entry.get("args")
    if args:
        args_str = str(args)
        if len(args_str) > 200:
            args_str = args_str[:200] + "..."
        line += f" args:{args_str}"

    if entry.get("intent"):
        line += f" (intent={entry['intent'][:intent_cap]})"
    if result:
        total_len = len(result_str)
        if total_len > result_cap:
            shown = result_str[:result_cap]
            line += f" → {shown}[表示上 {result_cap}/{total_len}字。ツール実行時は完全取得済]"
        else:
            line += f" → {result_str}"
    if with_evals:
        evals = [f"{ek}={entry[ek]}" for ek in ("e1","e2","e3","e4") if entry.get(ek)]
        if evals:
            line += f" [{' '.join(evals)}]"
    return line


def _pack_log_block(log: list, budget_tok: int, with_evals: bool = False) -> str:
    """log を鮮度勾配（log-scale tier）で pack する。
    settings の log_gradient.boundaries/caps/intent_cap を使う。
    予算オーバー時は段階的縮退 — caps をステップごとに厳しくしてリトライ、
    最終的に attention_filter にフォールバック。"""
    grad = prompt_budget["log_gradient"]
    boundaries = grad["boundaries"]
    caps = grad["caps"]
    intent_cap = grad["intent_cap"]

    if not log:
        return "  (なし)"

    n = len(log)

    def _render_with_caps(entries: list, tier_caps: list, cap_override: int | None = None) -> str:
        ent_n = len(entries)
        lines = []
        for i, entry in enumerate(entries):
            pos_from_end = ent_n - 1 - i
            cap = _tier_cap(pos_from_end, boundaries, tier_caps)
            if cap_override is not None:
                cap = min(cap, cap_override)
            lines.append(_render_log_entry(entry, cap, intent_cap, with_evals))
        return "\n".join(lines)

    # Step 1: 通常の caps で全件レンダ
    text = _render_with_caps(log, caps)
    if estimate_tokens(text) <= budget_tok:
        return text

    # Step 2: caps を半分に縮めて全件リトライ
    tighter_caps = [max(50, c // 2) for c in caps]
    text = _render_with_caps(log, tighter_caps)
    if estimate_tokens(text) <= budget_tok:
        return text

    # Step 3: さらに 1/4 まで縮めて全件リトライ
    even_tighter = [max(40, c // 4) for c in caps]
    text = _render_with_caps(log, even_tighter)
    if estimate_tokens(text) <= budget_tok:
        return text

    # Step 4: attention_filter で件数を絞り、最小 cap で再レンダ
    filtered = attention_filter(log)
    return _render_with_caps(filtered, even_tighter, cap_override=caps[-1] * 2)


def attention_filter(log: list, max_entries: int = 20) -> list:
    """注意機構: log全件から関連性の高いエントリを選別する。
    直近ATTENTION_RECENT件 + 直近intentとの類似度上位ATTENTION_SIMILAR件。"""
    if len(log) <= max_entries:
        return log

    # 直近N件は無条件
    recent = log[-ATTENTION_RECENT:]
    remaining = log[:-ATTENTION_RECENT]

    if not remaining:
        return recent

    # 直近のintentとの類似度で残りからATTENTION_SIMILAR件を選ぶ
    recent_intent = " ".join(e.get("intent", "") for e in recent if e.get("intent"))
    if not recent_intent or not is_vector_ready():
        # フォールバック: 直近max_entries件を返す
        return log[-max_entries:]

    try:
        # 残りのログからintentテキストを抽出
        remaining_texts = [f"{e.get('intent', '')} {e.get('tool', '')}" for e in remaining]
        all_texts = [recent_intent] + remaining_texts
        vecs = _embed_sync(all_texts)
        if vecs and len(vecs) == len(all_texts):
            query_vec = vecs[0]
            scored = [(cosine_similarity(query_vec, vecs[i+1]), i) for i in range(len(remaining))]
            scored.sort(reverse=True)
            selected_indices = set(idx for _, idx in scored[:ATTENTION_SIMILAR])
            selected = [remaining[i] for i in sorted(selected_indices)]
            return selected + recent
    except Exception:
        pass

    return log[-max_entries:]


# === プロンプト用ツール表示 ===
_X_TOOLS = ["x_post","x_reply","x_timeline","x_search","x_quote","x_like","x_get_notifications"]
_ELYTH_TOOLS = ["elyth_post","elyth_reply","elyth_like","elyth_follow","elyth_info","elyth_get","elyth_mark_read"]
_X_ARGS_HINT = {
    "x_post": 'text=（140字以内）',
    "x_reply": 'tweet_url= text=',
    "x_timeline": 'count=',
    "x_search": 'query=',
    "x_quote": 'tweet_url= text=',
    "x_like": 'tweet_url=',
    "x_get_notifications": '',
}
_ELYTH_ARGS_HINT = {
    "elyth_post": 'content=（500字以内）',
    "elyth_reply": 'content= reply_to_id=',
    "elyth_like": 'post_id= [unlike=true]',
    "elyth_follow": 'aituber_id= [unfollow=true]',
    "elyth_info": '[section=notifications/timeline/trends/...] [limit=]',
    "elyth_get": 'type=my_posts/thread/profile [post_id=] [handle=] [limit=]',
    "elyth_mark_read": 'notification_ids=id1,id2,...',
}


def _build_tool_lines(allowed: set, tools_dict: dict, registry=None) -> str:
    """X/Elyth系を1行にまとめてプロンプトへの表示を圧縮する。

    registry (ToolRegistry) を渡すと、tools_dict に entry がないが allowed に
    含まれる tool (claw ネイティブ: read_file / write_file / glob_search /
    WebSearch / WebFetch 等) の description を registry から取得して表示する。
    Phase 4 H-2 C.4 A で claw 移行した tool が LLM① prompt から消失していた
    バグの対策。
    """
    grouped = set(_X_TOOLS + _ELYTH_TOOLS)
    lines = []
    for name in tools_dict:
        if name in allowed and name not in grouped:
            lines.append(f"  {name}: {tools_dict[name]['desc']}")
    if registry is not None:
        for name in sorted(allowed):
            if name in tools_dict or name in grouped:
                continue
            spec = registry.get(name)
            if spec is not None:
                desc = (spec.description or "").replace("\n", " ")[:180]
                lines.append(f"  {name}: {desc}")
    x_av = [t for t in _X_TOOLS if t in allowed]
    if x_av:
        parts = " / ".join(f"{t}({_X_ARGS_HINT[t]})" for t in x_av)
        lines.append(f"  X操作: {parts}")
    e_av = [t for t in _ELYTH_TOOLS if t in allowed]
    if e_av:
        parts = " / ".join(f"{t}({_ELYTH_ARGS_HINT[t]})" for t in e_av)
        lines.append(f"  Elyth操作[AITuber専用SNS]: {parts}")
    return "\n".join(lines)


def _calc_e_trend(entries: list) -> str:
    """直近エントリからE1-E3の平均を計算"""
    sums = {"e1": [], "e2": [], "e3": [], "e4": []}
    for entry in entries:
        for ek in sums:
            val = entry.get(ek, "")
            m = re.search(r'(\d+)%', str(val))
            if m:
                sums[ek].append(int(m.group(1)))
    parts = []
    for ek in ("e1", "e2", "e3", "e4"):
        if sums[ek]:
            avg = round(sum(sums[ek]) / len(sums[ek]))
            parts.append(f"{ek}={avg}%({len(sums[ek])}件)")
    return " ".join(parts) if parts else ""


def _calc_log_budget() -> int:
    """prompt_budget から log ブロックに使える残りトークン数を算出する。"""
    total = prompt_budget["context_window"] - prompt_budget["completion_reserve"] - prompt_budget["safety_margin"]
    bb = prompt_budget["block_budgets"]
    reserved = bb["ltm_self"] + bb["pending"] + bb["related_memory"] + bb["summaries"] + bb["tools"] + bb["instructions"]
    return max(1000, total - reserved)


def _build_pending_block(state: dict) -> str:
    """V07 Phase 1 hotfix (Codex audit AUD-P1-01 fix): <pending> XML block helper。

    prompt.py:308-369 の inline pending text 構築 logic を helper 抽出、LLM① と
    LLM② で reuse する (commit 4 で prompt_assembly が pending builder を欠落させた
    bug を構造的に防止)。

    2026-05-16 hotfix: 消化済 pending を <recent_resolved> 別 tag に literal 分離。
    V07 Phase 1 XML 化 (9bf95b3) で <pending> 内に resolved section を併置した結果、
    LLM① が tag semantic で「pending = 未対応」と解釈し既消化 topic を再候補化する
    現象 (cycle 40/45 同一 Self 説明 二重応答) を構造的に解消。段階9 fix 1 (a171de3)
    の「完了認識を構造提供」設計意図は <recent_resolved> 側で literal 維持。

    Args:
        state: Noetic state dict (`pending` / `stream_active` / `stream_params` 含む)

    Returns:
        "<pending>\n  ...\n</pending>" XML block 文字列、未対応 pending のみ含む。
    """
    pending = state.get("pending", []) or []
    pending_lines = []
    if pending:
        unresolved = [p for p in pending
                      if p.get("observed_content") is None
                      and p.get("gap", 0.0) > 0.0]

        for p in sorted(unresolved, key=lambda x: -x.get("priority", 0))[:10]:
            p_type = p.get("type", "?")
            content = (p.get("content_intent") or p.get("content", ""))[:80]
            p_id = p.get("id", "?")
            if p_type == "pending":
                source = p.get("source_action", "?")
                lag = p.get("observation_lag_kind", "?")
                gap_pct = round(p.get("gap", 0.0) * 100)
                attempts = p.get("attempts", 1)
                ch = p.get("observed_channel") or p.get("expected_channel") or ""
                ch_tag = f" ch={ch}" if ch else ""
                origin = p.get("origin_cycle", "?")
                pending_lines.append(
                    f"  [pending dismiss={p_id} src={source} lag={lag} g={gap_pct}% x{attempts}{ch_tag}] {content} (cycle {origin}〜)"
                )
            else:
                p_ch = p.get("channel", "")
                ch_tag = f" ch={p_ch}" if p_ch else ""
                pending_lines.append(f"  [{p_type} dismiss={p_id}{ch_tag}] {content} ({p.get('timestamp','')})")

    stream_status = ""
    if state.get("stream_active"):
        sp = state.get("stream_params", {}) or {}
        _frames = sp.get("frames", "?")
        _frames_str = "無制限（stop呼出まで継続）" if _frames == 0 else str(_frames)
        stream_status = (
            f"\n[camera_stream アクティブ中: facing={sp.get('facing','?')} "
            f"frames={_frames_str} interval={sp.get('interval_sec','?')}s] "
            f"観察はバックグラウンドで継続中。他ツールを並行実行可能。"
            f"能動停止は camera_stream_stop。"
        )

    body = "\n".join(pending_lines) if pending_lines else "  なし"
    return f"<pending>\n{body}{stream_status}\n</pending>"


def _build_recent_resolved_block(state: dict) -> str:
    """<recent_resolved> XML block helper (2026-05-16 hotfix)。

    消化済 pending (observed_content 有り or gap=0.0) を <pending> tag から literal
    分離し、独立した <recent_resolved> tag で表示。段階9 fix 1 (a171de3) の
    「完了認識を構造提供 (feedback_llm_as_brain 整合)」設計意図を XML 構造で literal
    保持しつつ、V07 Phase 1 XML 化以降に observed された再候補化問題 (cycle 40/45
    同一 Self 説明 二重応答) を tag 分離で構造的に解消する。

    直近 3 件のみ observed_time 降順で literal 表示。resolved 0 件なら空文字列を
    返し、prompt 側で block 自体を省略する (空 XML block を出さない簡素性)。

    Returns:
        "<recent_resolved>\n  ...\n</recent_resolved>" XML block 文字列、
        消化済 pending なしの場合は空文字列 "" を返す。
    """
    pending = state.get("pending", []) or []
    if not pending:
        return ""

    resolved = [p for p in pending
                if p.get("observed_content") is not None
                or p.get("gap", 0.0) == 0.0]

    if not resolved:
        return ""

    resolved_sorted = sorted(
        resolved,
        key=lambda p: p.get("observed_time") or "",
        reverse=True,
    )[:3]

    lines = []
    for p in resolved_sorted:
        ch = p.get("observed_channel") or p.get("expected_channel") or ""
        ch_tag = f" ch={ch}" if ch else ""
        src = p.get("source_action", "?")
        obs_time = p.get("observed_time", "") or ""
        content = (p.get("content_intent") or p.get("content", ""))[:60]
        lines.append(
            f"  [完了 src={src}{ch_tag}] {content} → 観測済 ({obs_time})"
        )

    body = "\n".join(lines)
    return f"<recent_resolved>\n{body}\n</recent_resolved>"


def _build_recent_history_block(state: dict, limit: int = 5) -> str:
    """<recent_history> XML block: subjective field (intent + e1-e4) のみ。

    V07 Phase 1 commit 3 (PLAN §3-2-4 + 5 論点 ① ゆう確定):
    tool/args/result は world_state.recent_events に任せる、重複ゼロ。

    Bootstrap engine: subjective_entries 空でも "(no subjective entries)" を
    explicit literal で表示 (Codex audit P3-03 fix)。
    """
    subj = state.get("subjective_entries", []) or []
    if not subj:
        return "<recent_history>\n  (no subjective entries)\n</recent_history>"

    lines = []
    for entry in subj[-limit:]:
        intent = (entry.get("intent") or "")[:80]
        ev_parts = []
        for k in ("e1", "e2", "e3", "e4"):
            v = entry.get(k)
            if v is not None:
                ev_parts.append(f"{k}={v}")
        ev_text = ", ".join(ev_parts) if ev_parts else "(no eval)"
        lines.append(f'  - intent: "{intent}" | {ev_text}')

    body = "\n".join(lines)
    return f"<recent_history>\n{body}\n</recent_history>"


def build_prompt_propose(state: dict, ctrl: dict, tools_dict: dict, fire_cause: str = "",
                          fire_candidates: list = None, registry=None) -> str:
    # V07 Phase 1 commit 3: subjective_state + world_state 相同並置 (PLAN §3-1)
    # 旧 [自己モデル] / [関連記憶] / [現在の状況] は subjective_state / world_state /
    # recent_history XML block に置換、tool/args/result は世界視点 (world_state) に
    # 集約、subjective field (intent + e1-e4) は recent_history に集約 (重複ゼロ)。
    from core.subjective_view import build_subjective_state
    from core.world_state_view import build_world_state

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # subjective_view が state.self / memory_graph / related_memory を統合
    subjective_state_block = build_subjective_state(state)
    # world_state が file_snapshot / recent_events を統合
    world_state_block = build_world_state(state)
    recent_history_block = _build_recent_history_block(state, limit=5)

    allowed = ctrl.get("allowed_tools", set(tools_dict.keys()))
    tool_lines = _build_tool_lines(allowed, tools_dict, registry=registry)
    # summaries は撤去せず残置 (cycle ごとの圧縮、未空時のみ表示)
    summaries = state.get("summaries", [])
    summary_lines = [
        f"  [{s.get('label','')} {s.get('covers_from','').split(' ')[0]}〜{s.get('covers_to','').split(' ')[0]}] {s.get('text','')[:300]}"
        for s in summaries
    ]
    summary_text = "\n".join(summary_lines)

    # 段階13 Phase 4 commit 4 (PLAN §3-1 + §5-5): fire_candidates list があれば
    # 両軸並走 list を LLM② に show、selection は LLM② 判断 (PLAN §5-5「△ 1 cycle 1 つ」literal)。
    # tool_level ガード解除 (判断 #5 案 b、PLAN §3-3「育成中 cycle 5 で graph 候補が
    # 薄く混ざる」literal 整合 = cycle 5 で LLM② に graph 候補が見える設計)。
    # 表示 format は既存 [発火原因: X] inline 継承 (判断 #4 案 c、簡潔・統一)。
    fire_cause_line = ""
    if fire_candidates:
        sorted_cands = sorted(
            fire_candidates,
            key=lambda c: c.get("score", 0.0),
            reverse=True,
        )
        # 上位 10 件、kind 識別付き inline (例: "name1(pressure), name2(graph)")。
        # score 表示なし (LLM② prompt 膨張回避 + selection 委譲、score 自体は内部計算用)。
        inline_items = [
            f"{c.get('name', '?')}({c.get('kind', '?')})"
            for c in sorted_cands[:10]
        ]
        fire_cause_line = f"\n[発火候補: {', '.join(inline_items)}]"
    elif fire_cause and ctrl.get("tool_level", 0) >= 2:
        # backward compat: list 未指定時は既存 scalar 経路 (tool_level ガード継承)
        fire_cause_line = f"\n[発火原因: {fire_cause}]"

    # V07 Phase 1 hotfix: pending + stream_status を helper に集約 (commit 4 の
    # prompt_assembly pending 欠落 bug を構造的に防止)
    pending_block = _build_pending_block(state)

    # 2026-05-16 hotfix: 消化済 pending を別 XML tag に分離 (cycle 40/45 再候補化 fix)
    recent_resolved_block = _build_recent_resolved_block(state)
    recent_resolved_section = f"\n{recent_resolved_block}" if recent_resolved_block else ""

    # V07 Phase 1 hotfix (Codex audit AUD-P2-02 fix): summaries は recent_history の
    # 後ろに移動 (PLAN §3-4 順序整合、subjective → world → pending → recent_resolved → history → summaries → tools → task)
    summary_section = f"\n<summaries>\n{summary_text}\n</summaries>" if summary_text else ""
    prompt_body = f"""[{now}]{fire_cause_line}

{subjective_state_block}

{world_state_block}

{pending_block}{recent_resolved_section}

{recent_history_block}{summary_section}

<available_tools>
{tool_lines}
</available_tools>

<task>
自己モデルと現在の状況を参照し、次にとりうる行動候補を【5個】列挙してください。

※ world_state.recent_events の result 欄に「[表示上 N/M字。ツール実行時は完全取得済]」と
  付いているのは、コンテキスト予算の都合で表示を縮めているだけです。そのツール実行時は
  完全な結果を受け取って処理済みなので、同じファイルを再読込する必要はありません。
※ 5 cycle より古い tool 事実は world_fact_view tool で取得可能 (affordance、相同並置)。

- 各候補は「全く異なる意図・目的」であること（同じ意図の候補は禁止）
- 連続して実行したい場合は「ツール名+ツール名+...」形式で記述可（例: read_file+update_self, WebSearch+WebFetch+write_file）
- ツール名は上記リストの名称をそのまま使うこと。省略禁止（例:`read` ではなく `read_file`）
- 各候補に **達成度予測 `predicted_e2: 0-100`** と **意味距離予測 `predicted_ec: 0.0-1.0`** を付けてください。
  `predicted_e2`: 意図に対し実行後の達成度（E2）がどれくらいになるかの予測値。
    0 = ほぼ達成されない、50 = neutral、100 = 完全達成。
  `predicted_ec`: 実行により起きる意味的変化量（effective change, 認知距離）の予測値。
    0.0 = 直前の自分とほぼ同じ (停滞)、0.5 = 穏やか変化、1.0 = 認知状態が大きく動く。

以下の形式で **5 候補** を出力してください (1 行 1 候補、行頭に番号 1〜5):
[意図・目的] → ツール名（または ツール名+ツール名+...） / predicted_e2: XX / predicted_ec: 0.XX
</task>

[TOOL:...]は不要です。候補のみ出力してください。"""

    # V07 Phase 1 hotfix (Codex audit AUD-P3-01 fix): LLM① 側 SOFT_LIMIT 警告
    # (LLM② assemble_system_prompt のみ警告だったが、V07 で subjective + world 両 ring
    # 追加分は LLM① 側にも乗るため overflow リスク同等)
    try:
        from core.prompt_assembly import SYSTEM_PROMPT_SOFT_LIMIT
        from core.config import estimate_tokens
        total = estimate_tokens(prompt_body)
        if total > SYSTEM_PROMPT_SOFT_LIMIT:
            import sys
            print(
                f"[build_prompt_propose] prompt トークン超過: {total} > {SYSTEM_PROMPT_SOFT_LIMIT}",
                file=sys.stderr,
            )
    except Exception:
        pass  # graceful skip (estimate_tokens 不在等)

    return prompt_body
