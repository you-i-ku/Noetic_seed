"""プロンプト構築（propose/execute）+ 注意機構 + 鮮度勾配パッキング"""
import json
import re
from datetime import datetime
from core.config import prompt_budget, estimate_tokens
from core.embedding import _vector_ready, _embed_sync, cosine_similarity

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
    if not recent_intent or not _vector_ready:
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


def build_prompt_propose(state: dict, ctrl: dict, tools_dict: dict, fire_cause: str = "", registry=None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    self_text = json.dumps(state["self"], ensure_ascii=False) if state["self"] else "(なし)"
    energy = round(state.get("energy", 50), 1)
    e_trend = _calc_e_trend(state["log"][-10:])
    # 鮮度勾配で log 部を pack（全件維持、tier ごとに result を cap）
    log_text = _pack_log_block(state["log"], _calc_log_budget(), with_evals=False)
    allowed = ctrl.get("allowed_tools", set(tools_dict.keys()))
    tool_lines = _build_tool_lines(allowed, tools_dict, registry=registry)
    summaries = state.get("summaries", [])
    summary_lines = [
        f"  [{s.get('label','')} {s.get('covers_from','').split(' ')[0]}〜{s.get('covers_to','').split(' ')[0]}] {s.get('text','')[:300]}"
        for s in summaries
    ]
    summary_text = "\n".join(summary_lines)

    fire_cause_line = f"\n[発火原因: {fire_cause}]" if fire_cause and ctrl.get("tool_level", 0) >= 2 else ""

    # pending（未対応事項） — UPS v2 (type='pending') / 旧形式両対応
    # id 形式: p_{session}_{cycle:04d}_{source[:8]}_{ms} (log entry の
    # {session}_{cycle:04d} に対応)。dismiss 時はこの p_ prefix id を渡す。
    pending = state.get("pending", [])
    if pending:
        # 段階9 fix 1: 未消化と消化済を分離。消化済 (observed_content 埋まり or
        # gap=0.0) を "未対応事項" に出すと LLM が「まだ応答してない」と誤認する。
        unresolved = [p for p in pending
                      if p.get("observed_content") is None
                      and p.get("gap", 0.0) > 0.0]
        resolved = [p for p in pending
                    if p.get("observed_content") is not None
                    or p.get("gap", 0.0) == 0.0]

        pending_lines = []
        for p in sorted(unresolved, key=lambda x: -x.get("priority", 0))[:10]:
            p_type = p.get("type", "?")
            content = p.get("content", "")[:80]
            p_id = p.get("id", "?")
            if p_type == "pending":
                # UPS v2: source_action + lag_kind + gap + attempts + channel
                source = p.get("source_action", "?")
                lag = p.get("observation_lag_kind", "?")
                gap_pct = round(p.get("gap", 0.0) * 100)
                attempts = p.get("attempts", 1)
                ch = p.get("observed_channel") or p.get("expected_channel") or ""
                ch_tag = f" ch={ch}" if ch else ""
                origin = p.get("origin_cycle", "?")
                pending_lines.append(
                    f"  [pending dismiss_id={p_id} src={source} lag={lag} g={gap_pct}% x{attempts}{ch_tag}] {content} (cycle {origin}〜)"
                )
            else:
                # 旧形式 fallback (migration 期間 safety; Phase 5 iku 再生成後は消える)
                p_ch = p.get("channel", "")
                ch_tag = f" ch={p_ch}" if p_ch else ""
                pending_lines.append(f"  [{p_type} dismiss_id={p_id}{ch_tag}] {content} ({p.get('timestamp','')})")

        # 段階9 fix 1: 副次セクション。直近 3 件の消化済を参考表示し、
        # LLM に「これは完了したこと」を構造的に認識させる。
        if resolved:
            resolved_sorted = sorted(
                resolved,
                key=lambda p: p.get("observed_time") or "",
                reverse=True,
            )[:3]
            pending_lines.append("")
            pending_lines.append("  【最近完了した応答 (参考、既に済)】")
            for p in resolved_sorted:
                ch = p.get("observed_channel") or p.get("expected_channel") or ""
                ch_tag = f" ch={ch}" if ch else ""
                src = p.get("source_action", "?")
                obs_time = p.get("observed_time", "") or ""
                content = p.get("content", "")[:60]
                pending_lines.append(
                    f"  [完了 src={src}{ch_tag}] {content} → 観測済 ({obs_time})"
                )

        pending_text = "\n".join(pending_lines) if pending_lines else "  なし"
    else:
        pending_text = "  なし"

    # 関連記憶（Entity/Opinionネットワーク）
    from core.memory import get_relevant_memories, format_memories_for_prompt
    memories = get_relevant_memories(state, limit=8)
    memory_text = format_memories_for_prompt(memories) if memories else ""

    # camera_stream アクティブ時の状態表示（並行活動を可視化）
    stream_status_line = ""
    if state.get("stream_active"):
        sp = state.get("stream_params", {}) or {}
        _frames = sp.get("frames", "?")
        _frames_str = "無制限（stop呼出まで継続）" if _frames == 0 else str(_frames)
        stream_status_line = (
            f"\n[camera_stream アクティブ中: facing={sp.get('facing','?')} "
            f"frames={_frames_str} interval={sp.get('interval_sec','?')}s] "
            f"観察はバックグラウンドで継続中。他ツールを並行実行可能。"
            f"能動停止は camera_stream_stop。"
        )

    # Step E-3b: 反応待ち表示は旧 pending_feedback 由来だったが、UPS v2 pending
    # (retro_log_entry_id 付き + observed_content=None) が上記 [未対応事項] に
    # 表示されることで代替される。重複表示を避けて削除。

    return f"""[{now}]{fire_cause_line}

[LTM — 自己モデル]
{self_text}

[未対応事項]
{pending_text}{stream_status_line}
{f'{chr(10)}[関連記憶]{chr(10)}{memory_text}{chr(10)}' if memory_text else ''}
[STM — 現在の状況]
{f'summaries:{chr(10)}{summary_text}{chr(10)}' if summary_text else ''}log:
{log_text}

[利用可能なツール]
{tool_lines}

[候補生成プロトコル]
LTM（自己モデル）と STM（現在の状況）を参照し、次にとりうる行動候補を【5個】列挙してください。

※ log の result 欄に「[表示上 N/M字。ツール実行時は完全取得済]」と付いているのは、
  コンテキスト予算の都合で表示を縮めているだけです。そのツール実行時は完全な結果を
  受け取って処理済みなので、同じファイルを再読込する必要はありません。

- 各候補は「全く異なる意図・目的」であること（同じ意図の候補は禁止）
- 連続して実行したい場合は「ツール名+ツール名+...」形式で記述可（例: read_file+update_self, web_search+fetch_url+write_file）
- ツール名は上記リストの名称をそのまま使うこと。省略禁止（例:`read` ではなく `read_file`）
- 各候補に **達成度予測 `predicted_e2: 0-100`** を付けてください。
  意図に対し実行後の達成度（E2）がどれくらいになるかの予測値です。
  0 = ほぼ達成されない、50 = neutral、100 = 完全達成。

以下の形式で箇条書きのみ出力してください:
1. [意図・目的] → ツール名（または ツール名+ツール名+...） / predicted_e2: XX
2. [意図・目的] → ツール名（または ツール名+ツール名+...） / predicted_e2: XX
3. [意図・目的] → ツール名（または ツール名+ツール名+...） / predicted_e2: XX
4. [意図・目的] → ツール名（または ツール名+ツール名+...） / predicted_e2: XX
5. [意図・目的] → ツール名（または ツール名+ツール名+...） / predicted_e2: XX

[TOOL:...]は不要です。候補のみ出力してください。"""
