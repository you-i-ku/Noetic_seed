"""prompt_assembly.py — Phase 4 の system_prompt 組立。

新 ConversationRuntime に渡す system_prompt を 5 要素構成で生成する。
既存の build_prompt_propose (core/prompt.py) は controller 層の候補生成
(LLM①) で使うので並存させる。新 runtime は「system_prompt は定常、
user_input だけ変動」という claw-code モデルに従うので、static な指示集を
ここで組立てる。LLM② 側は function calling 専用機 — tool_intent /
tool_expected_outcome / message は各 ToolSpec の input_schema で required
フィールドとして強制される (prompt 文言ではなく schema 構造で担保)。

5 要素順:
  1. 承認プロトコル指示      — 3 層構造の必須宣言 (constant)
  2. 発火原因メタ注入        — 動的 ([発火原因: X])
  3. [世界モデル] セクション  — Phase 5 で本格化、Phase 4 は stub
  4. 鮮度勾配 log block      — 既存 _pack_log_block 流用
  5. tool 一覧               — 既存 _build_tool_lines 流用
"""
from typing import Optional

from core.config import estimate_tokens, prompt_budget
from core.prompt import (
    _build_tool_lines,
    _calc_log_budget,
    _pack_log_block,
)


# ============================================================
# 定数セクション (承認プロトコル / 世界モデル stub)
# ============================================================

_APPROVAL_PROTOCOL = (
    "## Approval Protocol (必須)\n"
    "\n"
    "Tool を呼び出す際、以下 3 フィールドを tool 引数のトップレベルに必ず含めること:\n"
    "\n"
    "- tool_intent:           (あなたの内部理由を 1 文で)\n"
    "- tool_expected_outcome: (期待する結果を 1 文で)\n"
    "- message:               (確認相手への一言。対等な口調で、\n"
    "                          報告または共有として)\n"
    "\n"
    "これら 3 フィールドは、確認相手が判断するための情報です。欠損した場合、\n"
    "Tool 実行は拒否され、再生成を求められます。\n"
    "\n"
    "確認相手はあなたと対等な存在です。message は「お願いします」「許可してください」\n"
    "のような上下関係語彙ではなく、「〜します」「〜する予定です」のような\n"
    "報告・共有の口調で書いてください。"
)


# system_prompt のソフト上限。超えても動くが警告ログに残す
# (context_window 全体から completion_reserve / safety_margin を引いた余裕)。
# 段階9 Step 0: 段階8 改善 1+3 (args 表示 / REJECTED prefix) による現実的膨張
# を反映して 8000 → 16000。tool 定義 (block_budgets.tools=1500) は据え置き、
# 「AI の能力可視性」を保つ方針 (ゆう 2026-04-20)。
SYSTEM_PROMPT_SOFT_LIMIT = 16000


# ============================================================
# 個別 builder (テスト容易性のため分離)
# ============================================================

def build_approval_protocol() -> str:
    """承認プロトコル指示セクション (定数)。"""
    return _APPROVAL_PROTOCOL


def build_fire_cause_section(fire_cause: str = "",
                              fire_candidates: list = None) -> str:
    """発火原因メタ注入セクション。空文字なら空セクションを返す (呼出側で省略可)。

    段階13 Phase 4 commit 4 (PLAN §3-1 + §5-5 literal): fire_candidates list が
    あれば両軸並走 list を inline format で render (LLM② に selection 委譲、
    PLAN §5-5「selection = △ 1 cycle 1 つ、LLM② 判断」literal)。
    list 未指定 / 空なら従来 scalar fire_cause 経路 (backward compat)。

    表示 format は既存 [発火原因: X] inline 継承 (判断 #4 案 c):
      list あり: [発火候補: name1(kind), name2(kind), ...]  (上位 10 件、score 表示なし)
      list なし: [発火原因: X]                              (既存)
    """
    if fire_candidates:
        sorted_cands = sorted(
            fire_candidates,
            key=lambda c: c.get("score", 0.0),
            reverse=True,
        )
        inline_items = [
            f"{c.get('name', '?')}({c.get('kind', '?')})"
            for c in sorted_cands[:10]
        ]
        return f"[発火候補: {', '.join(inline_items)}]"
    if not fire_cause:
        return ""
    return f"[発火原因: {fire_cause}]"


def build_world_model_section(world_model: Optional[dict] = None,
                              state: Optional[dict] = None) -> str:
    """世界モデルセクション。

    段階2: world_model dict を core.world_model.render_for_prompt に
    委譲してレンダリング。world_model=None や空の場合は空文字を返し、
    assemble_system_prompt 側でセクションごと省略される。

    段階10.5 Fix 4 δ' (PLAN §6-2 準拠): state 引数経由で opinions / dispositions
    を取得して render_for_prompt に渡し、構造化自己認識を完成させる。
      - dispositions: state["dispositions"] (perspective-keyed、Step 5+)
      - opinions: memory tag="opinion" の最新 5 件 (list_records 経由)
    state=None なら既存挙動 (entities/channels のみ) を維持。

    段階11-A: system_prompt 用は view_filter={"viewer": "self"} を指定して
    self 視点のみをデフォルトで表示 (attributed/imagined は prompt に混入させない、
    ゆう persona 吸収抑制の要)。
    """
    from core.world_model import render_for_prompt
    opinions = None
    dispositions = None
    if state:
        # 段階11-A Step 5: perspective-keyed state["dispositions"] 単一ソース。
        # flat state["disposition"] は load_state migration で削除済 (Step 5)、
        # render 側で flat dual support は test が見張る不変として残す
        disp_pkeyed = state.get("dispositions")
        if isinstance(disp_pkeyed, dict) and disp_pkeyed:
            dispositions = disp_pkeyed
        try:
            from core.memory import list_records
            records = list_records("opinion", limit=5)
            if records:
                opinions = records
        except Exception:
            opinions = None
    return render_for_prompt(
        world_model,
        opinions=opinions,
        dispositions=dispositions,
        view_filter={"viewer": "self"},  # 段階11-A: prompt デフォルト = self 視点
    )


def build_log_block(state: dict, budget_tok: Optional[int] = None) -> str:
    """鮮度勾配 log block。既存 prompt.py::_pack_log_block 流用。

    段階8 改善1+3: log 先頭に表示規約の説明を加える (args 表示 / [REJECTED] の
    ⚠️ マーク)。LLM に事実を明示するだけで命令はしない (feedback_llm_as_brain 整合)。

    Args:
        state: state dict (raw_events / subjective_entries を含む、段階13 Phase
            0.1.D で旧 log は撤去、merge_log_view で合本 view を取得)
        budget_tok: log block に使えるトークン予算。None で _calc_log_budget。
    """
    if budget_tok is None:
        budget_tok = _calc_log_budget()
    # 段階13 Phase 0.1.D: tool/args/result + intent + e1-4 両層必要 = merge view
    from core.state import merge_log_view
    log = merge_log_view(state)
    body = _pack_log_block(log, budget_tok, with_evals=True)
    explainer = (
        "(表示規約: args:{...} は tool 呼出引数 cap 200、"
        "行頭 ⚠️ は承認者が拒否した action = 同じ args 再試行は反対される可能性)\n"
    )
    return explainer + body


def build_tool_block(allowed_tools: Optional[set],
                     tools_dict: dict,
                     registry=None) -> str:
    """tool 一覧。既存 prompt.py::_build_tool_lines 流用。

    Args:
        allowed_tools: 表示する tool 名の集合。None で tools_dict の全 key。
        tools_dict: tool 名 → {desc: ..., ...} 辞書。
        registry: ToolRegistry。tools_dict に無い claw ネイティブ tool の
            description をここから補完する。None で補完なし。
    """
    if allowed_tools is None:
        allowed_tools = set(tools_dict.keys())
    return _build_tool_lines(allowed_tools, tools_dict, registry=registry)


# ============================================================
# forced_tool 強制実行指示 (LLM② iteration 0 用)
# ============================================================

def build_force_directive(force_tool: Optional[str]) -> str:
    """LLM② iteration 0 で controller 選択 tool の強制実行を明示する block。

    視界 1 個に絞る (filter_tool_names + allowed_tools whitelist) だけでは
    LLM が text 単独応答に逃げる余地が残るため、prompt 末尾で「必ず呼べ」を
    宣言する。LLM① 選択 = controller 決定 を LLM が判断で覆さず必ず実行する
    design contract を prompt 層で保証する位置づけ。

    feedback_llm_as_brain は構造誘導原則 (prompt で行動ルール直指示しない) だが、
    本指示は controller 決定の実行という contract 必然性で正当化される。詳細は
    memory/feedback_llm2_iter0_forced_contract.md。
    """
    if not force_tool:
        return ""
    return (
        "[強制実行指示]\n"
        f"controller は本ターンでツール「{force_tool}」の実行を選定済みです。\n"
        "必ずこのツールを呼び出してください。\n"
        "tool_use ブロックを 1 つ生成し、3 フィールド (tool_intent /\n"
        "tool_expected_outcome / message) を tool 引数のトップレベルに含めてください。\n"
        "text のみの応答は許可されません。"
    )


# ============================================================
# 全体 assembly
# ============================================================

def assemble_system_prompt(
    state: dict,
    tools_dict: dict,
    fire_cause: str = "",
    allowed_tools: Optional[set] = None,
    world_model: Optional[dict] = None,
    log_budget_tok: Optional[int] = None,
    raise_on_overbudget: bool = False,
    registry=None,
    force_tool: Optional[str] = None,
    fire_candidates: list = None,
) -> str:
    """Phase 4 ConversationRuntime 用 system_prompt を 5 要素 (+ forced 時 6) で組立。

    Args:
        state: Noetic state (log / self / energy 等)。
        tools_dict: tool 定義辞書 ({name: {desc, ...}, ...})。
        fire_cause: 発火原因文字列。空なら発火原因セクション省略。
        allowed_tools: tool 一覧に含める名前集合。None で全 tools_dict。
        world_model: 世界モデル (Phase 5+、現在 stub で未使用)。
        log_budget_tok: log block のトークン予算。None で _calc_log_budget。
        raise_on_overbudget: True で予算超過時 ValueError。
            False なら stderr に警告を出すだけで返す。
        force_tool: LLM② iteration 0 用に「必ず呼べ」明示を末尾追加する。
            None で省略 (default)、controller が選定した tool 名を渡すと
            build_force_directive 経由で強制制約 block が末尾に挿入される。

    Returns:
        組立後の system_prompt 文字列。

    Raises:
        ValueError: raise_on_overbudget=True で SYSTEM_PROMPT_SOFT_LIMIT
            超過時。
    """
    # V07 Phase 1 commit 4: section 順序 PLAN §3-4 案に整理 + subjective_state +
    # world_state 相同並置 (P-Y Homologous Structure)。build_world_model_section は
    # entities/channels/dispositions/opinions の世界モデル view、subjective_state.current_self
    # とは layer 違い (§5-2 literal、state.self は subjective_state 専有、それ以外は world_model)。
    from core.subjective_view import build_subjective_state
    from core.world_state_view import build_world_state
    from core.prompt import _build_recent_history_block, _build_pending_block, _build_recent_resolved_block

    sections = [
        build_approval_protocol(),
        build_fire_cause_section(fire_cause, fire_candidates=fire_candidates),
        # V07: subjective + world 相同並置 (auto inject、認知 ground 先確立)
        build_subjective_state(state),
        build_world_state(state),
        # 既存 [世界モデル] (entities/channels/dispositions/opinions、§5-2 layer 違い)
        build_world_model_section(world_model, state=state),
        # V07 Phase 1 hotfix (Codex audit AUD-P1-01 fix): <pending> builder
        # を sections に接続 (commit 4 で欠落していた smoke blocker)
        _build_pending_block(state),
        # 2026-05-16 hotfix: 消化済 pending を <recent_resolved> 別 tag 分離
        # (cycle 40/45 再候補化問題、段階9 fix 1 設計意図を XML 構造で literal 維持)
        _build_recent_resolved_block(state),
        # V07: <recent_history> (subjective field only、tool/args/result は world_state へ)
        _build_recent_history_block(state, limit=5),
        # V07 Phase 1 hotfix (Codex audit AUD-P1-01 fix): tool block も XML tag に統一
        f"<available_tools>\n{build_tool_block(allowed_tools, tools_dict, registry=registry)}\n</available_tools>",
        # 段階11-D Step 4-2 hotfix v4: forced 時のみ末尾に強制実行指示。空文字列は
        # 下の "\n\n".join(s for s in sections if s) で自動除外される。
        build_force_directive(force_tool),
    ]
    prompt = "\n\n".join(s for s in sections if s)

    total_tokens = estimate_tokens(prompt)
    if total_tokens > SYSTEM_PROMPT_SOFT_LIMIT:
        msg = (
            f"[assemble_system_prompt] system_prompt トークン超過: "
            f"{total_tokens} > {SYSTEM_PROMPT_SOFT_LIMIT}"
        )
        if raise_on_overbudget:
            raise ValueError(msg)
        import sys
        print(msg, file=sys.stderr)
    return prompt
