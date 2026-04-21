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
    "Tool を呼び出す際、以下 3 フィールドを tool_input に必ず含めること:\n"
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


def build_fire_cause_section(fire_cause: str) -> str:
    """発火原因メタ注入セクション。空文字なら空セクションを返す (呼出側で省略可)。"""
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
      - dispositions: state["disposition"] dict (単数キー、reflection が更新)
      - opinions: memory tag="opinion" の最新 5 件 (list_records 経由)
    state=None なら既存挙動 (entities/channels のみ) を維持。
    """
    from core.world_model import render_for_prompt
    opinions = None
    dispositions = None
    if state:
        disp = state.get("disposition")
        if isinstance(disp, dict) and disp:
            dispositions = disp
        try:
            from core.memory import list_records
            records = list_records("opinion", limit=5)
            if records:
                opinions = records
        except Exception:
            opinions = None
    return render_for_prompt(world_model, opinions=opinions, dispositions=dispositions)


def build_log_block(state: dict, budget_tok: Optional[int] = None) -> str:
    """鮮度勾配 log block。既存 prompt.py::_pack_log_block 流用。

    段階8 改善1+3: log 先頭に表示規約の説明を加える (args 表示 / [REJECTED] の
    ⚠️ マーク)。LLM に事実を明示するだけで命令はしない (feedback_llm_as_brain 整合)。

    Args:
        state: state dict (log list を含む)
        budget_tok: log block に使えるトークン予算。None で _calc_log_budget。
    """
    if budget_tok is None:
        budget_tok = _calc_log_budget()
    log = state.get("log", [])
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
) -> str:
    """Phase 4 ConversationRuntime 用 system_prompt を 5 要素で組立。

    Args:
        state: Noetic state (log / self / energy 等)。
        tools_dict: tool 定義辞書 ({name: {desc, ...}, ...})。
        fire_cause: 発火原因文字列。空なら発火原因セクション省略。
        allowed_tools: tool 一覧に含める名前集合。None で全 tools_dict。
        world_model: 世界モデル (Phase 5+、現在 stub で未使用)。
        log_budget_tok: log block のトークン予算。None で _calc_log_budget。
        raise_on_overbudget: True で予算超過時 ValueError。
            False なら stderr に警告を出すだけで返す。

    Returns:
        組立後の system_prompt 文字列。

    Raises:
        ValueError: raise_on_overbudget=True で SYSTEM_PROMPT_SOFT_LIMIT
            超過時。
    """
    sections = [
        build_approval_protocol(),
        build_fire_cause_section(fire_cause),
        # 段階10.5 Fix 4 δ': state 経由で opinions / dispositions を渡し構造化自己認識を完成
        build_world_model_section(world_model, state=state),
        "[STM — log]\n" + build_log_block(state, log_budget_tok),
        "[利用可能なツール]\n" + build_tool_block(allowed_tools, tools_dict, registry=registry),
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
