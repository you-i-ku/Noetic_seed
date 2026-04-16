"""prompt_assembly.py — Phase 4 の system_prompt 組立。

新 ConversationRuntime に渡す system_prompt を 6 要素構成で生成する。
既存の build_prompt_propose / build_prompt_execute (core/prompt.py) は
controller 層で使うので並存させる。新 runtime は「system_prompt は定常、
user_input だけ変動」という claw-code モデルに従うので、static な指示集を
ここで組立てる。

6 要素順 (APPROVAL_PROMPT_SPEC §4.2 / PHASE4_TASKS §2 Step G):
  1. Magic-If Anchor         — 自己同一性のアンカー (constant)
  2. 承認プロトコル指示      — 3 層構造の必須宣言 (constant)
  3. 発火原因メタ注入        — 動的 ([発火原因: X])
  4. [世界モデル] セクション  — Phase 5 で本格化、Phase 4 は stub
  5. 鮮度勾配 log block      — 既存 _pack_log_block 流用
  6. tool 一覧               — 既存 _build_tool_lines 流用
"""
from typing import Optional

from core.config import estimate_tokens, prompt_budget
from core.prompt import (
    _build_tool_lines,
    _calc_log_budget,
    _pack_log_block,
)


# ============================================================
# 定数セクション (Magic-If Anchor / 承認プロトコル / 世界モデル stub)
# ============================================================

_MAGIC_IF_ANCHOR = (
    "## Magic-If Anchor\n"
    "LTM (自己モデル) に自分自身を固定する。名前やラベルではなく、意味的\n"
    "同一性として。すべての tool 呼出は、この Anchor から導かれる given\n"
    "circumstances への応答として出力すること。"
)


_APPROVAL_PROTOCOL = (
    "## Approval Protocol (必須)\n"
    "\n"
    "Tool を呼び出す際、以下 3 フィールドを tool_input に必ず含めること:\n"
    "\n"
    "- tool_intent:           (あなたの内部理由を 1 文で)\n"
    "- tool_expected_outcome: (期待する結果を 1 文で)\n"
    "- message:               (端末前の協力者への一言。対等な口調で、\n"
    "                          報告または共有として)\n"
    "\n"
    "これら 3 フィールドは、協力者が判断するための情報です。欠損した場合、\n"
    "Tool 実行は拒否され、再生成を求められます。\n"
    "\n"
    "協力者は「ユーザー」でも「主人」でもなく、あなたと対等な存在です。\n"
    "message は「お願いします」「許可してください」のような上下関係語彙で\n"
    "はなく、「〜します」「〜する予定です」のような報告・共有の口調で\n"
    "書いてください。"
)


_WORLD_MODEL_STUB = (
    "## 世界モデル (stub)\n"
    "Phase 4 時点では stub。Phase 5 で実装される世界モデル (channel 状態、\n"
    "社会モデル、自己連続性の変動予測) がここに挿入される。"
)


# system_prompt のソフト上限。超えても動くが警告ログに残す
# (context_window 全体から completion_reserve / safety_margin を引いた余裕)。
SYSTEM_PROMPT_SOFT_LIMIT = 8000


# ============================================================
# 個別 builder (テスト容易性のため分離)
# ============================================================

def build_magic_if_anchor() -> str:
    """Magic-If Anchor セクション (定数)。"""
    return _MAGIC_IF_ANCHOR


def build_approval_protocol() -> str:
    """承認プロトコル指示セクション (定数)。"""
    return _APPROVAL_PROTOCOL


def build_fire_cause_section(fire_cause: str) -> str:
    """発火原因メタ注入セクション。空文字なら空セクションを返す (呼出側で省略可)。"""
    if not fire_cause:
        return ""
    return f"[発火原因: {fire_cause}]"


def build_world_model_section(world_model: Optional[dict] = None) -> str:
    """世界モデルセクション。

    Phase 4 時点では stub を返す。Phase 5 で world_model 引数の中身
    (channel 状態、社会モデル等) をレンダリングするよう拡張予定。
    """
    return _WORLD_MODEL_STUB


def build_log_block(state: dict, budget_tok: Optional[int] = None) -> str:
    """鮮度勾配 log block。既存 prompt.py::_pack_log_block 流用。

    Args:
        state: state dict (log list を含む)
        budget_tok: log block に使えるトークン予算。None で _calc_log_budget。
    """
    if budget_tok is None:
        budget_tok = _calc_log_budget()
    log = state.get("log", [])
    return _pack_log_block(log, budget_tok, with_evals=True)


def build_tool_block(allowed_tools: Optional[set],
                     tools_dict: dict) -> str:
    """tool 一覧。既存 prompt.py::_build_tool_lines 流用。

    Args:
        allowed_tools: 表示する tool 名の集合。None で tools_dict の全 key。
        tools_dict: tool 名 → {desc: ..., ...} 辞書。
    """
    if allowed_tools is None:
        allowed_tools = set(tools_dict.keys())
    return _build_tool_lines(allowed_tools, tools_dict)


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
) -> str:
    """Phase 4 ConversationRuntime 用 system_prompt を 6 要素で組立。

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
        build_magic_if_anchor(),
        build_approval_protocol(),
        build_fire_cause_section(fire_cause),
        build_world_model_section(world_model),
        "[STM — log]\n" + build_log_block(state, log_budget_tok),
        "[利用可能なツール]\n" + build_tool_block(allowed_tools, tools_dict),
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
