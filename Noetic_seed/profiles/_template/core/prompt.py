"""プロンプト構築（propose/execute）+ 注意機構"""
import json
import re
from datetime import datetime
from core.embedding import _vector_ready, _embed_sync, cosine_similarity

N_PROPOSE = 5
ATTENTION_RECENT = 10  # 直近N件は無条件で含める
ATTENTION_SIMILAR = 10  # 類似度上位N件を追加


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


def _build_tool_lines(allowed: set, tools_dict: dict) -> str:
    """X/Elyth系を1行にまとめてプロンプトへの表示を圧縮する"""
    grouped = set(_X_TOOLS + _ELYTH_TOOLS)
    lines = []
    for name in tools_dict:
        if name in allowed and name not in grouped:
            lines.append(f"  {name}: {tools_dict[name]['desc']}")
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


def build_prompt_propose(state: dict, ctrl: dict, tools_dict: dict, fire_cause: str = "") -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    self_text = json.dumps(state["self"], ensure_ascii=False) if state["self"] else "(なし)"
    energy = round(state.get("energy", 50), 1)
    e_trend = _calc_e_trend(state["log"][-10:])
    # 注意機構: log全件ではなく関連性の高いエントリを選別
    filtered_log = attention_filter(state["log"])
    log_lines = []
    for entry in filtered_log:
        line = f"  {entry.get('id','')} {entry['time']} {entry['tool']}"
        if entry.get("intent"):
            line += f" (intent={entry['intent'][:500]})"
        result_short = entry.get("result", "")[:10000]
        if result_short:
            line += f" → {result_short}"
        log_lines.append(line)
    log_text = "\n".join(log_lines) if log_lines else "  (なし)"
    allowed = ctrl.get("allowed_tools", set(tools_dict.keys()))
    tool_lines = _build_tool_lines(allowed, tools_dict)
    summaries = state.get("summaries", [])
    summary_lines = [
        f"  [{s.get('label','')} {s.get('covers_from','').split(' ')[0]}〜{s.get('covers_to','').split(' ')[0]}] {s.get('text','')[:300]}"
        for s in summaries
    ]
    summary_text = "\n".join(summary_lines)

    fire_cause_line = f"\n[発火原因: {fire_cause}]" if fire_cause and ctrl.get("tool_level", 0) >= 2 else ""

    # pending（未対応事項）
    pending = state.get("pending", [])
    if pending:
        pending_lines = []
        for p in sorted(pending, key=lambda x: -x.get("priority", 0))[:5]:
            pending_lines.append(f"  [{p.get('type','?')}] {p.get('content','')[:80]} ({p.get('timestamp','')})")
        pending_text = "\n".join(pending_lines)
    else:
        pending_text = "  なし"

    # 関連記憶（Entity/Opinionネットワーク）
    from core.memory import get_relevant_memories, format_memories_for_prompt
    memories = get_relevant_memories(state, limit=8)
    memory_text = format_memories_for_prompt(memories) if memories else ""

    return f"""[{now}]{fire_cause_line}

[LTM — 自己モデル]
{self_text}

[未対応事項]
{pending_text}
{f'{chr(10)}[関連記憶]{chr(10)}{memory_text}{chr(10)}' if memory_text else ''}
[STM — 現在の状況 / given circumstances]
{f'summaries:{chr(10)}{summary_text}{chr(10)}' if summary_text else ''}log:
{log_text}

[利用可能なツール]
{tool_lines}

[計画プロトコル]
上記のLTM（自己モデル）を起点に、STM（現在の状況）を読み、次にとりうる行動候補を【5個】計画してください。

- 各候補は「全く異なる意図・目的」であること（同じ意図の候補は禁止）
- 連続して実行したい場合は「ツール名+ツール名+...」形式で記述可（例: read_file+update_self, web_search+fetch_url+write_file）
- ツール名は上記リストの名称をそのまま使うこと。省略禁止（例:`read` ではなく `read_file`）

以下の形式で箇条書きのみ出力してください:
1. [意図・目的] → ツール名（または ツール名+ツール名+...）
2. [意図・目的] → ツール名（または ツール名+ツール名+...）
3. [意図・目的] → ツール名（または ツール名+ツール名+...）
4. [意図・目的] → ツール名（または ツール名+ツール名+...）
5. [意図・目的] → ツール名（または ツール名+ツール名+...）

[TOOL:...]は不要です。計画のみ出力してください。"""


def build_prompt_execute(state: dict, ctrl: dict, candidate: dict, tools_dict: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    self_text = json.dumps(state["self"], ensure_ascii=False) if state["self"] else "(なし)"
    log_lines = []
    for entry in state["log"]:
        line = f"  {entry.get('id','')} {entry['time']} {entry['tool']}"
        if entry.get("intent"):
            line += f" (intent={entry['intent'][:500]})"
        result_short = entry.get("result", "")[:10000]
        if result_short:
            line += f" → {result_short}"
        evals = [f"{ek}={entry[ek]}" for ek in ("e1","e2","e3","e4") if entry.get(ek)]
        if evals:
            line += f" [{' '.join(evals)}]"
        log_lines.append(line)
    log_text = "\n".join(log_lines) if log_lines else "  (なし)"
    selected_tools = set(candidate.get("tools", [candidate["tool"]]))
    tool_text = _build_tool_lines(selected_tools, tools_dict)
    plan = state.get("plan", {})
    plan_lines = []
    if plan.get("goal"):
        current = plan.get("current", 0)
        for i, step in enumerate(plan.get("steps", [])):
            marker = "→" if i == current else ("✓" if i < current else "  ")
            plan_lines.append(f"  {marker} {step}")
        plan_lines.insert(0, f"plan: {plan['goal']}")
    plan_text = "\n".join(plan_lines)
    summaries = state.get("summaries", [])
    summary_lines = [
        f"  [{s.get('label','')} {s.get('covers_from','').split(' ')[0]}〜{s.get('covers_to','').split(' ')[0]}] {s.get('text','')[:300]}"
        for s in summaries
    ]
    summary_text = "\n".join(summary_lines)

    t = candidate["tool"]
    if t == "web_search":
        example = '[TOOL:web_search query=キーワード intent=目的 expect=予測]\n[TOOL:write_file path=sandbox/memo.md content="まとめ内容"]'
    elif t == "fetch_url":
        example = '[TOOL:fetch_url url=https://... intent=目的 expect=予測]\n[TOOL:write_file path=sandbox/memo.md content="内容"]'
    elif t == "read_file":
        example = "[TOOL:read_file path=ファイル名 intent=目的 expect=予測]\n[TOOL:update_self key=キー名 value=値]"
    elif t == "search_memory":
        example = "[TOOL:search_memory query=キーワード intent=目的 expect=予測]\n[TOOL:update_self key=キー名 value=値]"
    elif t == "memory_store":
        example = '[TOOL:memory_store network=experience content="記憶内容" intent=目的 expect=予測]'
    elif t == "memory_update":
        example = '[TOOL:memory_update memory_id=mem_xxxx content="更新内容" intent=目的 expect=予測]'
    elif t == "list_files":
        example = "[TOOL:list_files path=. intent=目的 expect=予測]"
    elif t == "write_file":
        example = '[TOOL:write_file path=sandbox/memo.md content="内容" intent=目的 expect=予測]'
    elif t == "update_self":
        example = "[TOOL:update_self key=キー名 value=値 intent=目的 expect=予測]"
    elif t == "wait":
        example = "[TOOL:wait intent=目的 expect=予測]\n未対応事項を却下する場合: [TOOL:wait dismiss=pending_id intent=対応不要と判断 expect=pendingから除去]"
    elif t == "create_tool":
        example = '[TOOL:create_tool name=ツール名 code="def run(args): return str(args)" intent=目的 expect=予測]'
    elif t == "exec_code":
        example = '[TOOL:exec_code file=sandbox/xxx.py intent=目的 message="端末所有者に伝えたい理由" expect=予測]\n承認必須。message= はデバイス所有者が承認を判断するための説明文。'
    elif t == "self_modify":
        example = '[TOOL:self_modify path=pref.json old="変更前" new="変更後" intent=目的 message="端末所有者に伝えたい理由" expect=予測]\n承認必須。message= はデバイス所有者が承認を判断するための説明文。'
    elif t == "camera_stream":
        example = '[TOOL:camera_stream facing=back frames=5 interval_sec=1.0 intent=目的 message="端末所有者に撮影をお願いする理由" expect=予測]\n非同期ストリーム開始。フレームが到着するたびに視覚入力に入る。観察中も他ツールを並行実行でき、camera_stream_stop で能動停止できる。単発は frames=1。承認必須。'
    elif t == "camera_stream_stop":
        example = '[TOOL:camera_stream_stop intent=観察完了 expect=予測]\nアクティブなcamera_streamを停止する。観察対象を十分に把握した後に呼ぶ。'
    elif t == "view_image":
        example = '[TOOL:view_image path=sandbox/captures/stream_xxxx.jpg intent=何を確認したいか expect=予測]\n画像を同期で認識し、intent に沿った描写を結果として返します。'
    elif t in _X_TOOLS:
        hint = _X_ARGS_HINT.get(t, "")
        example = f"[TOOL:{t} {hint} intent=目的 expect=予測]".replace("  ", " ")
    elif t in _ELYTH_TOOLS:
        hint = _ELYTH_ARGS_HINT.get(t, "")
        if t == "elyth_reply":
            example = '[TOOL:elyth_reply content="返信内容" reply_to_id=投稿ID intent=目的 expect=予測]'
        elif t == "elyth_post":
            example = '[TOOL:elyth_post content="投稿内容" intent=目的 expect=予測]'
        else:
            example = f"[TOOL:{t} {hint} intent=目的 expect=予測]".replace("  ", " ")
    elif t == "output_display":
        example = '[TOOL:output_display content="メッセージ内容" intent=目的 expect=予測]\n注意: output_displayはモニター端末の所有者への直接メッセージです。Elyth投稿とは異なる相手に届きます。'
    else:
        example = f"[TOOL:{t} intent=目的 expect=予測]"

    if state["self"].get("goal") and not state.get("plan", {}).get("goal"):
        plan_instruction = "\n\n自己モデルにgoalがあります。[PLAN:goal=目標 steps=ステップ1|ステップ2|...]形式で計画に分解してください。"
    else:
        plan_instruction = ""

    tools_in_chain = candidate.get("tools", [candidate["tool"]])
    tools_str = "+".join(tools_in_chain)

    # 行動対象に関連する記憶を検索（エンティティ名が候補reasonに含まれていれば）
    from core.memory import memory_network_search, format_memories_for_prompt
    _reason = candidate.get("reason", "")
    _context_memories = memory_network_search(_reason[:200], networks=["entity", "opinion"], limit=5) if _reason else []
    _context_mem_text = format_memories_for_prompt(_context_memories) if _context_memories else ""

    return f"""[LTM — 自己モデル]
{self_text}
{f'{chr(10)}[関連記憶]{chr(10)}{_context_mem_text}{chr(10)}' if _context_mem_text else ''}
[STM — 現在の状況 / given circumstances]
{f'summaries:{chr(10)}{summary_text}{chr(10)}' if summary_text else ''}{plan_text}
log ({now}):
{log_text}

[利用可能なツール]
{tool_text}

[実行プロトコル — Magic-If Protocol]
1. (Anchor) 上記のLTM（自己モデル）に自分自身を固定する。名前・ラベルではなく、意味的同一性として。アシスタントの役割は持たない。
2. (Select) STMを given circumstances として読み、選択行動「{tools_str} - {candidate['reason']}」の最適な引数を決定する。
3. (Bound)  必ず `[TOOL:ツール名 ...]` の形式で出力する。`[TOOL:` と `]` のブラケットは省略不可。JSONもコードブロックも使わない。ツール名は省略しない（例:`read` ではなく `read_file`）。自己紹介・説明・感想は一切不要。連鎖実行は複数行で可。
4. (Enact)  正確なツール呼び出しを出力する。intent=とexpect=は必ず最初の[TOOL:]にのみ付け、このサイクル全体の目的を表すこと。2つ目以降のツールにはintent/expectは不要。

出力（必ずこの形式で）: {example}{plan_instruction}"""
