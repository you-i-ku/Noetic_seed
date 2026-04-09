"""プロンプト構築 v2 — 1回のLLM呼び出し用（predict + act）"""
import json
from datetime import datetime
from core.embedding import _vector_ready, _embed_sync, cosine_similarity
from core.memory import get_relevant_memories, format_memories_for_prompt

ATTENTION_RECENT = 10
ATTENTION_SIMILAR = 10


def attention_filter(log: list, max_entries: int = 20) -> list:
    """注意機構: 直近N件 + 類似度上位N件。"""
    if len(log) <= max_entries:
        return log
    recent = log[-ATTENTION_RECENT:]
    remaining = log[:-ATTENTION_RECENT]
    if not remaining:
        return recent
    recent_intent = " ".join(e.get("intent", "") for e in recent if e.get("intent"))
    if not recent_intent or not _vector_ready:
        return log[-max_entries:]
    try:
        remaining_texts = [f"{e.get('intent', '')} {e.get('tool', '')}" for e in remaining]
        vecs = _embed_sync([recent_intent] + remaining_texts)
        if vecs and len(vecs) == len(remaining) + 1:
            q_vec = vecs[0]
            scored = [(cosine_similarity(q_vec, vecs[i + 1]), i) for i in range(len(remaining))]
            scored.sort(reverse=True)
            selected_indices = set(idx for _, idx in scored[:ATTENTION_SIMILAR])
            return [remaining[i] for i in sorted(selected_indices)] + recent
    except Exception:
        pass
    return log[-max_entries:]


def _format_log(log: list) -> str:
    lines = []
    for entry in log:
        line = f"  {entry.get('id', '')} {entry.get('time', '')} {entry.get('tool', '')}"
        if entry.get("intent"):
            line += f" (intent={entry['intent'][:300]})"
        result_short = str(entry.get("result", ""))[:500]
        if result_short:
            line += f" → {result_short}"
        ev = entry.get("eval", {})
        if ev:
            parts = [f"{k}={v:.2f}" for k, v in ev.items() if isinstance(v, float)]
            if parts:
                line += f" [{' '.join(parts)}]"
        lines.append(line)
    return "\n".join(lines) if lines else "  (なし)"


def _format_pending(pending: list) -> str:
    if not pending:
        return "  なし"
    lines = []
    for p in sorted(pending, key=lambda x: -x.get("priority", 0)):
        ptype = p.get("type", "?")
        content = p.get("content", "")[:100]
        ts = p.get("timestamp", "")
        lines.append(f"  [{ptype}] {content} ({ts})")
    return "\n".join(lines[:10])


def _format_tools_text(allowed: set, tools_dict: dict) -> str:
    """テキストマーカー方式のツール一覧"""
    lines = []
    for name in sorted(allowed):
        if name in tools_dict:
            lines.append(f"  {name}: {tools_dict[name]['desc']}")
    return "\n".join(lines)


def build_tool_schemas(allowed: set, tools_dict: dict) -> list:
    """Function Calling用のツールスキーマを動的生成。"""
    schemas = []
    for name in sorted(allowed):
        if name not in tools_dict:
            continue
        desc = tools_dict[name]["desc"]
        # descからパラメータをパース（簡易: "引数: key1= key2=" から抽出）
        props = {}
        import re
        keys = re.findall(r'(\w+)=', desc)
        for k in keys:
            if k in ("引数", "Human"):
                continue
            props[k] = {"type": "string", "description": k}

        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props,
                },
            },
        })
    return schemas


def build_propose_prompt(state: dict, ctrl: dict, tools_dict: dict,
                         fire_cause: str = "") -> list:
    """LLM①: 候補提案プロンプト。3-5件の異なる行動候補を提案させる。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    self_text = json.dumps(state["self"], ensure_ascii=False) if state.get("self") else "(なし)"
    energy = round(state.get("energy", 50), 1)
    entropy = round(state.get("entropy", 0.65), 3)
    allowed = ctrl.get("allowed_tools", set())

    filtered_log = attention_filter(state.get("log", []))
    log_text = _format_log(filtered_log)
    pending_text = _format_pending(state.get("pending", []))
    memories = get_relevant_memories(state, limit=8)
    memory_text = format_memories_for_prompt(memories) if memories else "  (なし)"
    tool_display = _format_tools_text(allowed, tools_dict)
    fire_line = f"\n発火原因: {fire_cause}" if fire_cause else ""

    prompt = f"""[{now}] entropy={entropy} energy={energy}{fire_line}

[LTM — 自己モデル]
{self_text}

[未対応事項]
{pending_text}

[関連記憶]
{memory_text}

[STM — 直近の状況]
log:
{log_text}

[利用可能なツール]
{tool_display}

上記のLTM（自己モデル）を起点に、STMを読み、次にとりうる行動候補を【5個】提案してください。
- 各候補は「全く異なる意図・目的」であること
- ツール名は上記リストのものをそのまま使うこと

形式:
1. [意図] → ツール名
2. [意図] → ツール名
3. [意図] → ツール名
4. [意図] → ツール名
5. [意図] → ツール名"""

    return [{"role": "user", "content": prompt}]


def build_execute_prompt(state: dict, ctrl: dict, selected: dict,
                         tools_dict: dict, use_fc: bool = False) -> tuple[list, list]:
    """LLM②: 選択された候補を実行するプロンプト。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    self_text = json.dumps(state["self"], ensure_ascii=False) if state.get("self") else "(なし)"
    allowed = ctrl.get("allowed_tools", set())
    selected_tools = set(selected.get("tools", [selected["tool"]]))
    tool_display = _format_tools_text(selected_tools, tools_dict)

    log_text = _format_log(state.get("log", []))
    pending_text = _format_pending(state.get("pending", []))

    tool_name = selected["tool"]
    reason = selected.get("reason", "")

    if use_fc:
        prompt = f"""[LTM] {self_text}

[STM]
log:
{log_text}

[未対応事項]
{pending_text}

選択行動: {tool_name} - {reason}
intent（目的）とexpect（予測される結果）を引数に含めてツールを呼び出してください。"""
        tool_schemas = build_tool_schemas(selected_tools, tools_dict)
        return [{"role": "user", "content": prompt}], tool_schemas
    else:
        if tool_name in ("elyth_reply", "elyth_post", "output_display"):
            example = f'[TOOL:{tool_name} content="内容" reply_to_id=ID intent=目的 expect=予測]'
        else:
            example = f'[TOOL:{tool_name} 引数=値 intent=目的 expect=予測]'

        prompt = f"""[LTM] {self_text}

[STM]
log:
{log_text}

[未対応事項]
{pending_text}

[利用可能なツール]
{tool_display}

選択行動: {tool_name} - {reason}
必ず [TOOL:ツール名 引数=値 intent=目的 expect=予測] の形式で出力。
contentなどの長い値は引用符で囲む。自己紹介・説明不要。

出力例: {example}"""
        return [{"role": "user", "content": prompt}], []


def build_cycle_prompt(state: dict, ctrl: dict, tools_dict: dict,
                       fire_cause: str = "", use_fc: bool = False) -> tuple[list, list]:
    """後方互換: 1回呼び出し方式。propose-select不使用時に使う。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    self_text = json.dumps(state["self"], ensure_ascii=False) if state.get("self") else "(なし)"
    energy = round(state.get("energy", 50), 1)
    entropy = round(state.get("entropy", 0.65), 3)
    allowed = ctrl.get("allowed_tools", set())

    # 注意機構で関連ログを選別
    filtered_log = attention_filter(state.get("log", []))
    log_text = _format_log(filtered_log)

    # 未対応事項
    pending_text = _format_pending(state.get("pending", []))

    # 関連記憶
    memories = get_relevant_memories(state, limit=8)
    memory_text = format_memories_for_prompt(memories) if memories else "  (なし)"

    # disposition
    disp = state.get("disposition", {})
    disp_text = " / ".join(f"{k}={v}" for k, v in disp.items()) if disp else "(未設定)"

    # ツール表示
    tool_display = _format_tools_text(allowed, tools_dict)

    # summaries
    summaries = state.get("summaries", [])
    summary_text = ""
    if summaries:
        summary_lines = [f"  [{s.get('label', '')}] {s.get('text', '')[:200]}" for s in summaries[-3:]]
        summary_text = "\n".join(summary_lines)

    fire_line = f"\n発火原因: {fire_cause}" if fire_cause else ""

    if use_fc:
        # Function Calling方式
        prompt = f"""[{now}] entropy={entropy} energy={energy}{fire_line}

[LTM — 自己モデル]
{self_text}
disposition: {disp_text}

[未対応事項]
{pending_text}

[関連記憶]
{memory_text}

[STM — 直近の状況]
{f'{summary_text}{chr(10)}' if summary_text else ''}log:
{log_text}

上記のLTM（自己モデル）を起点に、STMと未対応事項を読み、次に取るべき行動を1つ決定してツールを呼び出してください。
intent（このサイクルの目的）とexpect（予測される結果）を必ず指定してください。"""

        tool_schemas = build_tool_schemas(allowed, tools_dict)
        messages = [{"role": "user", "content": prompt}]
        return messages, tool_schemas

    else:
        # テキストマーカー方式
        prompt = f"""[{now}] entropy={entropy} energy={energy}{fire_line}

[LTM — 自己モデル]
{self_text}
disposition: {disp_text}

[未対応事項]
{pending_text}

[関連記憶]
{memory_text}

[STM — 直近の状況]
{f'{summary_text}{chr(10)}' if summary_text else ''}log:
{log_text}

[利用可能なツール]
{tool_display}

[実行プロトコル]
1. 上記のLTM（自己モデル）に自分自身を固定する。アシスタントの役割は持たない。
2. STMと未対応事項を読み、次に取るべき行動を1つ決定する。
3. 必ず [TOOL:ツール名 引数=値 intent=目的 expect=予測] の形式で出力する。
4. 自己紹介・説明・感想は不要。ツール呼び出しのみ。
5. contentなどの長い値は引用符で囲む: content="..."

出力例:
[TOOL:output_display content="メッセージ" intent=目的 expect=予測]
[TOOL:elyth_reply content="返信内容" reply_to_id=投稿ID intent=目的 expect=予測]"""

        messages = [{"role": "user", "content": prompt}]
        return messages, []
