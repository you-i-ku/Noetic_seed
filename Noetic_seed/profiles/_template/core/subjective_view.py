"""subjective_view.py — V07 Phase 1 commit 2 (Homologous Structure: subjective ring)

<subjective_state> XML block を構築する helper。current_self + graph_summary + related_memory
を thin format で auto inject する。P-Y (Homologous Structure) 整合、world_state と相同並置。

V07_PROMPT_PARADIGM_SHIFT_PLAN §3-2-1 / §4 Phase 1 commit 2 literal。

設計指針:
- ★ cheap metadata only (Codex audit 2026-05-14 P2-03 fix): _graph_summary は state 直 read +
  軽い helper のみ、memory_graph_tool.estimate_clusters / call_llm パス絶対通さない (毎 cycle
  prompt assembly 中に LLM 再帰呼出 trigger 防止)。詳細 cluster label は memory_graph_tool
  affordance で。
- current_self: state.self literal、prompt.py:234 self_text 構築 + :368-373 [自己モデル] block の
  literal 移植。
- related_memory: get_relevant_memories 経由、暫定で content[:80] + "…" marker (commit 5 で
  memory.py に excerpt_80chars field 追加後、commit 6 で正規化、Codex audit P2-04 additive-only
  fix)。
"""
import json
from typing import Optional

from core.dynamic_composition import _frontier_count, compute_graph_maturity


# ============================================================
# current_self (state.self literal、prompt.py:234 / 368-373 移植)
# ============================================================

def _current_self(state: dict) -> str:
    """state.self を JSON dump で返す。

    V7-A1 整合: 全 field 表示 (name / context_integrity_check / 等)、subjective_state
    の core component、bootstrap 期 (cycle 1) でも seed.txt 由来の name は常に値あり。

    None / 空 dict の場合は "(なし)" literal (既存 prompt.py:234 同パターン)。
    """
    self_data = state.get("self")
    if not self_data:
        return "(なし)"
    return json.dumps(self_data, ensure_ascii=False)


# ============================================================
# graph_summary (cheap metadata only、Codex audit P2-03 fix)
# ============================================================

def _ego_summary(state: dict) -> dict:
    """ego mode: self が touch する link 数 + 1-hop neighbor count (cheap)。

    state.memory_links は cached だが、list_links() helper を cheap path で呼ぶ。
    LLM 呼出 / estimate_clusters は通さない (P2-03 fix)。
    """
    try:
        from core.memory_links import list_links
        links = list_links(limit=200) or []
    except Exception:
        links = []

    self_edges = 0
    neighbor_ids = set()
    for link in links:
        subj = link.get("subject", "")
        obj = link.get("object", "")
        if subj == "self" or obj == "self":
            self_edges += 1
            if subj == "self" and obj:
                neighbor_ids.add(obj)
            if obj == "self" and subj:
                neighbor_ids.add(subj)
    return {
        "self_edges": self_edges,
        "memory_1hop_count": len(neighbor_ids),
    }


def _global_summary(state: dict) -> dict:
    """global mode: frontier + graph_maturity + cluster proxy (cheap)。

    cheap helper のみ:
    - _frontier_count(state): state direct (dynamic_composition.py:160)
    - compute_graph_maturity(state): 5 軸 sigmoid 合成 (dynamic_composition.py:237)
    - cluster_mi / cluster_inter_ratio: phase6_metrics 由来 (reflect で更新済 cache)

    Bootstrap 期 (cycle 1) でも全部 0.0 / 0 を返す (graceful skip、空 graph 整合)。
    """
    try:
        frontier = _frontier_count(state)
    except Exception:
        frontier = 0
    try:
        maturity = compute_graph_maturity(state)
    except Exception:
        maturity = 0.0

    p6 = state.get("phase6_metrics", {}) or {}
    return {
        "frontier_node_count": int(frontier),
        "graph_maturity": round(float(maturity), 3),
        "cluster_mi": (
            round(float(p6.get("cluster_mi")), 3)
            if p6.get("cluster_mi") is not None
            else None
        ),
        "cluster_link_pairs": p6.get("last_cluster_link_pairs"),
    }


def _graph_summary(state: dict) -> dict:
    """ego + global 両 mode の cheap metadata only summary。

    PLAN §3-2-1 (V7-A1 確定 = both) literal:
    - ego mode: self が touch する edge / 1-hop neighbor の count
    - global mode: frontier + graph_maturity + cluster proxy (phase6_metrics 由来)

    Codex audit P2-03 fix: estimate_clusters / call_llm 絶対通さない。詳細 cluster label
    が欲しい時は memory_graph_tool affordance 経由 (LLM 自発呼出)。
    """
    return {
        "ego": _ego_summary(state),
        "global": _global_summary(state),
    }


# ============================================================
# related_memory (hybrid 80 字、暫定 fallback、commit 6 で正規化)
# ============================================================

def _excerpt_80(content: str, length: int = 80) -> str:
    """content を 80 字に切る (超過時 "…" marker)。

    暫定実装: memory.py に excerpt_80chars field が追加される (commit 5) まで本関数で生成。
    commit 6 で `_related_memory` を memory.py 側 field 利用に置換予定 (Codex audit P2-04
    additive-only fix 整合)。
    """
    if not content:
        return ""
    if len(content) <= length:
        return content
    return content[:length] + "…"


def _related_memory(state: dict, limit: int = 5, excerpt_len: int = 80) -> list:
    """get_relevant_memories 経由で hybrid format (id + 80 字抜粋) を返す。

    V7-A2 確定 (80 字) + 5 論点 ③ (hybrid 形式)。

    暫定実装 (commit 5/6 で正規化): get_relevant_memories の content から
    `_excerpt_80` で抜粋を生成。commit 5 で memory.py に excerpt_80chars field が
    追加されたら commit 6 で本関数を field 直 read に置換 (Codex audit P2-04
    additive-only 整合)。

    Returns:
        list of {id, excerpt_80chars, cycles_ago} dict (空 list 許容)
    """
    try:
        from core.memory import get_relevant_memories
        from core.config import llm_cfg
        retrieval_cfg = llm_cfg.get("retrieval", {}) or {}
        memories = get_relevant_memories(
            state, limit=limit,
            use_links=bool(retrieval_cfg.get("use_links", False)),
            link_depth=int(retrieval_cfg.get("link_depth", 1)),
            link_top_n=int(retrieval_cfg.get("link_top_n", 3)),
        )
    except Exception:
        memories = []

    if not memories:
        return []

    current_cycle = state.get("cycle_id") or 0
    result = []
    for m in memories[:limit]:
        m_id = m.get("id", "")
        # commit 5/6 で `excerpt_80chars` field 直 read に置換 (P2-04 additive)
        excerpt = _excerpt_80(m.get("content", ""), excerpt_len)
        cycles_ago = _calc_cycles_ago(m, current_cycle)
        result.append({
            "id": m_id,
            "excerpt_80chars": excerpt,
            "cycles_ago": cycles_ago,
        })
    return result


def _calc_cycles_ago(memory: dict, current_cycle: int) -> Optional[int]:
    """memory entry の origin cycle と current cycle の差を返す。

    cycle_id field がない / parse 失敗で None。
    """
    origin = memory.get("cycle_id") or memory.get("origin_cycle")
    if origin is None:
        return None
    try:
        return max(0, int(current_cycle) - int(origin))
    except (TypeError, ValueError):
        return None


# ============================================================
# build_subjective_state (entry point)
# ============================================================

def build_subjective_state(state: dict) -> str:
    """<subjective_state> XML block を構築する (V07 Phase 1 commit 2 entry point).

    P-Y (Homologous Structure) 整合: world_state と対称配置、auto inject、thin view、
    同等 affordance (memory_graph_tool / search_memory は維持)。

    Bootstrap engine: 空 graph でも "cluster=0 / frontier=0 / maturity=0" 等の explicit
    literal を表示、空白こそが affordance を駆動 (`feedback_intent_over_form_homologous_structure`
    §How to apply、Codex audit P3-03 fix)。

    Args:
        state: Noetic state dict (`self` / `phase6_metrics` / `cycle_id` 等を含む、欠損許容)

    Returns:
        "<subjective_state>\n  ...\n</subjective_state>" XML block 文字列
    """
    cs = _current_self(state)
    gs = _graph_summary(state)
    rm = _related_memory(state)

    self_lines = [
        "  current_self:",
        f"    {cs}",
    ]

    ego = gs["ego"]
    glob = gs["global"]
    graph_lines = [
        "  graph_summary:",
        f"    ego:    self_edges={ego['self_edges']}, memory_1hop_count={ego['memory_1hop_count']}",
        f"    global: cluster_mi={glob['cluster_mi']}, frontier={glob['frontier_node_count']}, maturity={glob['graph_maturity']}",
    ]

    memory_lines = ["  related_memory:"]
    if rm:
        for m in rm:
            cy_tag = (
                f" [{m['cycles_ago']} cycle 前]" if m["cycles_ago"] is not None else ""
            )
            memory_lines.append(
                f"    {m['id']}{cy_tag}: \"{m['excerpt_80chars']}\""
            )
    else:
        memory_lines.append("    (no related memories)")

    body = "\n".join(self_lines + graph_lines + memory_lines)
    return f"<subjective_state>\n{body}\n</subjective_state>"
