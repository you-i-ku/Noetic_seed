"""memory_graph — 段階11-D Phase 0 Step 0.2 + 段階13 軽減 2: ego/global/both view.

self / memory を unified node とした graph 構造 tool。
self virtual entries は state.self を描画時に on-the-fly で生成 (永続化なし)。
self ↔ memory edges は描画時に similarity 計算 (永続化なし)。
memory ↔ memory edges は memory_links.jsonl の永続 link を参照。

5 層スキーマ:
  nodes   : self (continuous, id 不変) + memory (kind 区別)
  edges   : memory↔memory (永続) + self↔memory (on-the-fly, similarity only)
  clusters: cluster_estimation 経由 (Phase 5 既実装、global/both view で表示)
  frontier: Phase 4 で本実装、現状 section 出さない
  trace   : 直近 memory / link 総数 (簡易 MVP)

view:
  ego    : self 中心 (self facets + edges_self_to_memory + edges_memory_to_memory)
  global : graph 全体俯瞰 (memory_total + clusters + topology summary)
  both   : ego + global の重畳

channel は描画対象外 (channel = 界面 layer、self/memory = internal layer、layer 違反回避)。
channel 永続化廃止論点は reserved memo に温存。

出力: 中立 JSON 構造化 text (自然言語ゼロ、feedback_llm_as_brain 整合)。
"""
import json
from typing import Optional

from core.embedding import is_vector_ready, _embed_sync, cosine_similarity
from core.memory import list_records
from core.memory_links import list_links
from core.state import load_state
from core.tag_registry import list_registered_tags


# Step 0.2 MVP 定数
DEFAULT_DEPTH = 2
DEFAULT_SELF_TO_MEMORY_TOP_K = 3
DEFAULT_SELF_TO_MEMORY_THRESHOLD = 0.5
SELF_FACET_CONTENT_MAX_LEN = 500
MEMORY_CONTENT_MAX_LEN = 500
ALL_MEMORY_LIMIT_PER_TAG = 200


def _self_to_virtual_entries(state: dict) -> list:
    """state.self を memory entry 形式の list に live 変換 (案 ③ 永続化なし)。

    各 key を 1 facet として扱う。value は str() 化 (JSON-encoded string も
    MVP では生扱い、parse 精緻化は Phase 1+ で検討)。
    """
    entries = []
    self_dict = state.get("self", {}) or {}
    for key, value in self_dict.items():
        if not value:
            continue
        entries.append({
            "id": f"self.{key}",
            "kind": "self",
            "facet": key,
            "content": str(value),
        })
    return entries


def _build_self_node(state: dict, virtual_entries: list) -> dict:
    """continuous self node (id 不変、attributes は live read、LLM 呼出非依存)."""
    return {
        "id": "self",
        "kind": "self",
        "facets": [e["facet"] for e in virtual_entries],
        "metrics": {
            "cycle": state.get("cycle_id"),
            "entropy": state.get("entropy"),
            "pressure": state.get("pressure"),
            "energy": state.get("energy"),
        },
    }


def _list_all_memory_entries(limit_per_tag: int = ALL_MEMORY_LIMIT_PER_TAG) -> list:
    """全登録 tag の memory entry を集約取得 (新しい順)."""
    out = []
    for tag in list_registered_tags():
        try:
            recs = list_records(tag, limit=limit_per_tag)
        except Exception:
            continue
        out.extend(recs)
    return out


def _compute_self_to_memory_edges(virtual_entries: list, all_memory: list,
                                   *, top_k: int = DEFAULT_SELF_TO_MEMORY_TOP_K,
                                   threshold: float = DEFAULT_SELF_TO_MEMORY_THRESHOLD) -> list:
    """on-the-fly: 各 self facet と memory entry の similarity を計算、
    top-K かつ threshold 超を edge 化。永続化なし。

    relation 推定は MVP では skip (similarity のみ)、LLM judge は Phase 1+ で。
    """
    edges = []
    if not virtual_entries or not all_memory or not is_vector_ready():
        return edges

    self_texts = [e["content"][:SELF_FACET_CONTENT_MAX_LEN] for e in virtual_entries]
    mem_texts = [m.get("content", "")[:MEMORY_CONTENT_MAX_LEN] for m in all_memory]

    try:
        all_vecs = _embed_sync(self_texts + mem_texts)
    except Exception:
        return edges
    if not all_vecs or len(all_vecs) != len(self_texts) + len(mem_texts):
        return edges

    self_vecs = all_vecs[:len(self_texts)]
    mem_vecs = all_vecs[len(self_texts):]

    for i, sv_entry in enumerate(virtual_entries):
        sims = []
        for j, mem in enumerate(all_memory):
            try:
                sim = float(cosine_similarity(self_vecs[i], mem_vecs[j]))
            except Exception:
                continue
            if sim < threshold:
                continue
            sims.append((mem, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        for mem, sim in sims[:top_k]:
            edges.append({
                "from": sv_entry["id"],
                "to": mem.get("id", ""),
                "similarity": round(sim, 3),
            })
    return edges


def _compute_memory_edges() -> list:
    """memory ↔ memory edges を memory_links.jsonl から取得 (永続 link).

    Step 0.2 MVP: 全 link を flatten して返す。depth/top_n 制御は Phase 1+ で
    follow_links 経路と統合検討。
    """
    edges = []
    for l in list_links(limit=10000):
        lt = l.get("link_type", "none")
        if lt == "none":
            continue
        edges.append({
            "from": l.get("from_id", ""),
            "to": l.get("to_id", ""),
            "relation": lt,
            "confidence": float(l.get("confidence", 0.0)),
        })
    return edges


def _compute_trace(all_memory: list, memory_edges: list) -> dict:
    """簡易 trace: 総数のみ (Phase 4+ で cycle 別 trace を本実装)."""
    return {
        "memory_total": len(all_memory),
        "link_total": len(memory_edges),
    }


def _compute_clusters_with_label(all_memory: list) -> list:
    """全 graph cluster 推定 (Phase 5 cluster_estimation 経由、label LLM 生成)。

    軽減 2 (段階13 着手前 base 整備) で配線完成。estimate_clusters の graceful
    fallback (numpy / vector_ready / embed 失敗時に 1 cluster) に依存、tool 側で
    例外処理は持たない。
    """
    from core.cluster_estimation import estimate_clusters
    from core.llm import call_llm

    raw = estimate_clusters(all_memory, method="hybrid", llm_call_fn=call_llm)
    return [
        {
            "cluster_id": c["cluster_id"],
            "label": c.get("label", ""),
            "memory_ids": c["memory_ids"],
            "size": len(c["memory_ids"]),
            "method": c["method"],
        }
        for c in raw
    ]


def _compute_topology_summary(memory_edges: list) -> dict:
    """全 link の集計 (link_total / by_relation / avg_confidence)。

    edges 詳細は ego/both view で出るので、global view では量を抑えて summary のみ。
    """
    by_relation: dict = {}
    confidence_sum = 0.0
    for e in memory_edges:
        rel = e.get("relation", "unknown")
        by_relation[rel] = by_relation.get(rel, 0) + 1
        confidence_sum += float(e.get("confidence", 0.0))
    avg_conf = (confidence_sum / len(memory_edges)) if memory_edges else 0.0
    return {
        "link_total": len(memory_edges),
        "by_relation": by_relation,
        "avg_confidence": round(avg_conf, 3),
    }


def _memory_graph(args: dict) -> str:
    """memory_graph tool 本体。出力は JSON 構造化 text (中立、自然言語ゼロ).

    args (PLAN §6-6 signature 互換):
        view: "ego" (Step 0.2 で実装)、global / both は Phase 4/5 で descended
        depth: int (default 2)
        focus_node: ego view 中心切替用 (Phase 1+ で使用、Step 0.2 では受取のみ)
        cluster_count: global view cluster 件数上限 (Phase 5 で使用、Step 0.2 では受取のみ)
        frontier_count: frontier 候補件数上限 (Phase 4 で使用、Step 0.2 では受取のみ)
    """
    view = args.get("view", "ego")
    try:
        depth = int(args.get("depth", DEFAULT_DEPTH) or DEFAULT_DEPTH)
    except (ValueError, TypeError):
        depth = DEFAULT_DEPTH

    # PLAN §6-6 signature 互換の placeholder 引数 (受取のみ、reject しない)。
    args.get("focus_node")
    args.get("cluster_count")
    args.get("frontier_count")

    if view not in ("ego", "global", "both"):
        return json.dumps({
            "error": f"view={view} は未対応",
            "supported_views": ["ego", "global", "both"],
        }, ensure_ascii=False, indent=2)

    state = load_state()
    all_memory = _list_all_memory_entries()
    memory_edges = _compute_memory_edges()
    trace = _compute_trace(all_memory, memory_edges)

    output: dict = {"view": view, "depth": depth}

    if view in ("ego", "both"):
        virtual_entries = _self_to_virtual_entries(state)
        self_node = _build_self_node(state, virtual_entries)
        output["self"] = self_node
        output["edges_self_to_memory"] = _compute_self_to_memory_edges(
            virtual_entries, all_memory)
        output["edges_memory_to_memory"] = memory_edges

    if view in ("global", "both"):
        output["memory_total"] = len(all_memory)
        output["clusters"] = _compute_clusters_with_label(all_memory)
        output["topology"] = _compute_topology_summary(memory_edges)

    output["trace_recent"] = trace
    return json.dumps(output, ensure_ascii=False, indent=2)
