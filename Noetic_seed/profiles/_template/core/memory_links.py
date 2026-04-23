"""memory_links — 段階11-B Phase 4 (A-MEM NeurIPS 2025 準拠)。

memory entry 間に関係性 link を LLM judge で生成、Zettelkasten 形式の graph 化。
既存 entity facts (段階4) とは別 layer として並立、データ重複なし。

link_type 候補: similar / contradict / elaborate / causal / temporal
confidence 閾値: 0.7 以上のみ保存 (link 爆発防止、escape hatch で tune 可)
link 生成タイミング: memory_store 同期 + top-K=5 近傍のみ LLM judge
(Phase 3 keywords 同期と一貫性)

Phase 4 スコープ: storage のみ。retrieval (follow_links) は Phase 5 smoke
観察後に判定 (link_grad_density > 0.2 なら拡張、< 0.2 なら保留)。
"""
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from core.config import MEMORY_DIR


LINK_FILE_NAME = "memory_links.jsonl"
LINK_CONFIDENCE_THRESHOLD = 0.7      # Phase 4 Step 4.2: これ未満は discard
LINK_GENERATION_TOP_K = 5            # Phase 4 Step 4.3: 近傍 top-K のみ judge
LINK_TYPES = ("similar", "contradict", "elaborate", "causal", "temporal")


def _link_file() -> Path:
    MEMORY_DIR.mkdir(exist_ok=True)
    return MEMORY_DIR / LINK_FILE_NAME


def _build_link_prompt(entry_a: dict, entry_b: dict) -> str:
    """link 判定用 LLM prompt (PLAN Step 4.2 準拠、軽量 JSON 出力)。"""
    a_kws = entry_a.get("keywords", []) or []
    b_kws = entry_b.get("keywords", []) or []
    return (
        "以下 2 つの記憶 entry の関係性を判定してください:\n"
        f"\nEntry A: {entry_a.get('content', '')[:300]}\n"
        f"  tag: {entry_a.get('network', '')}, keywords: {a_kws}\n"
        f"Entry B: {entry_b.get('content', '')[:300]}\n"
        f"  tag: {entry_b.get('network', '')}, keywords: {b_kws}\n"
        "\nlink_type 候補:\n"
        "- similar: 類似内容 (重複に近い)\n"
        "- contradict: 矛盾 (Phase 3 reconciliation と補完関係)\n"
        "- elaborate: 片方が他方を詳述 / 具体化\n"
        "- causal: 原因-結果 / 行動-観察\n"
        "- temporal: 時系列的連続\n"
        "- (none): 関係薄い → link 作らない\n"
        "\n出力は JSON のみ (他の文字を含めない):\n"
        '{"link_type": str, "confidence": float, "reason": str}\n'
        '- link_type は上記 5 種 or "none"\n'
        "- confidence 0.0-1.0、reason は 1 文"
    )


def _parse_link_response(response: str) -> dict:
    """LLM 応答から {link_type, confidence, reason} を抽出 (robust)。

    失敗時 / none 時 / 閾値未満は link 作らない扱い (link_type="none")。
    """
    default = {"link_type": "none", "confidence": 0.0, "reason": ""}
    try:
        m = re.search(r'\{.*\}', response, re.DOTALL)
        if not m:
            return default
        data = json.loads(m.group(0))
        lt = str(data.get("link_type", "none")).strip().lower()
        if lt not in LINK_TYPES:
            lt = "none"
        conf = float(data.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        return {
            "link_type": lt,
            "confidence": conf,
            "reason": str(data.get("reason", ""))[:200],
        }
    except Exception:
        return default


def _llm_judge_link(entry_a: dict, entry_b: dict,
                    llm_call_fn: Optional[Callable] = None) -> dict:
    """2 entry 間の link 判定 (LLM mock 可能、error で graceful fallback)。"""
    if llm_call_fn is None:
        from core.llm import call_llm
        llm_call_fn = call_llm
    prompt = _build_link_prompt(entry_a, entry_b)
    try:
        response = llm_call_fn(prompt, max_tokens=200, temperature=0.2)
        return _parse_link_response(response)
    except Exception as e:
        print(f"  [memory_links] judge skip (error: {e})")
        return {"link_type": "none", "confidence": 0.0, "reason": ""}


def _build_link_entry(from_entry: dict, to_entry: dict, verdict: dict) -> dict:
    """link entry dict 生成 (storage 用)。11-A perspective 属性を付与。"""
    from core.perspective import default_self_perspective
    return {
        "id": f"link_{uuid.uuid4().hex[:12]}",
        "from_id": from_entry.get("id", ""),
        "to_id": to_entry.get("id", ""),
        "link_type": verdict.get("link_type", "none"),
        "confidence": float(verdict.get("confidence", 0.0)),
        "perspective": default_self_perspective(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason": verdict.get("reason", ""),
    }


def _append_link(link_entry: dict) -> None:
    """memory_links.jsonl に atomic append。"""
    fpath = _link_file()
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(link_entry, ensure_ascii=False) + "\n")


def list_links(limit: int = 200) -> list:
    """memory_links.jsonl から新しい順に limit 件読む。"""
    fpath = _link_file()
    if not fpath.exists():
        return []
    try:
        lines = fpath.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def generate_links_for(new_entry: dict, *,
                       top_k: int = LINK_GENERATION_TOP_K,
                       embed_fn: Optional[Callable] = None,
                       cosine_fn: Optional[Callable] = None,
                       llm_call_fn: Optional[Callable] = None,
                       confidence_threshold: float = LINK_CONFIDENCE_THRESHOLD,
                       candidate_limit: int = 50) -> list:
    """新 memory entry の近傍 top-K に対して link 生成 (memory_store 同期呼出想定)。

    PLAN §5 Phase 4 Step 4.3: memory_store 同期呼出で embedding 近傍 top-K を
    LLM judge、confidence >= threshold のみ memory_links.jsonl に append。

    Args:
        new_entry: 新規 memory entry
        top_k: 近傍数 (PLAN 推奨 5、cost / 密度バランス)
        embed_fn / cosine_fn: None で近傍取得 skip (= link 生成 skip)
        llm_call_fn: None で core.llm.call_llm 使用
        confidence_threshold: 閾値 (PLAN 推奨 0.7、smoke 後 tune 可能)
        candidate_limit: 同 network から走査する候補上限

    Returns:
        [link_entry, ...] 生成した link の list (smoke 分析用、実運用は副作用)
    """
    from core.memory import list_records

    network = new_entry.get("network", "")
    new_id = new_entry.get("id", "")
    if not network:
        return []

    # embed_fn 未指定なら近傍取得不可 → link 生成 skip
    if embed_fn is None or cosine_fn is None:
        return []

    all_records = list_records(network, limit=candidate_limit)
    candidates = [r for r in all_records if r.get("id") != new_id]
    if not candidates:
        return []

    new_content = new_entry.get("content", "")
    if not new_content:
        return []

    # embedding で近傍 top-K 取得
    try:
        vecs = embed_fn([new_content] + [c.get("content", "") for c in candidates])
    except Exception:
        return []
    if not vecs or len(vecs) != 1 + len(candidates):
        return []

    query_vec = vecs[0]
    sims = []
    for i, c in enumerate(candidates):
        try:
            sim = float(cosine_fn(query_vec, vecs[i + 1]))
        except Exception:
            continue
        sims.append((c, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    near = sims[:top_k]

    # 各近傍に対して LLM judge
    created = []
    for cand, _sim in near:
        verdict = _llm_judge_link(new_entry, cand, llm_call_fn=llm_call_fn)
        if verdict.get("link_type", "none") == "none":
            continue
        if verdict.get("confidence", 0.0) < confidence_threshold:
            continue
        link_entry = _build_link_entry(new_entry, cand, verdict)
        _append_link(link_entry)
        created.append(link_entry)
    return created


# ============================================================
# 段階11-C G-lite Phase 1: retrieval 拡張 (follow_links)
# ============================================================

LINK_TRAVERSAL_MAX_DEPTH_DEFAULT = 1   # G-lite 推奨 depth=1、smoke 後 tune 余地
LINK_TRAVERSAL_TOP_N_DEFAULT = 3       # depth 毎の上位 N 件 (cost / 密度 balance)


def follow_links(
    node_id: str,
    *,
    depth: int = LINK_TRAVERSAL_MAX_DEPTH_DEFAULT,
    link_types: Optional[tuple] = None,
    min_confidence: float = LINK_CONFIDENCE_THRESHOLD,
    top_n_per_depth: int = LINK_TRAVERSAL_TOP_N_DEFAULT,
    visited: Optional[set] = None,
) -> list:
    """指定 memory entry から link graph を traverse して近傍 entry を返す.

    段階11-C G-lite Phase 1: 既存 `memory_links.jsonl` (storage、Phase 4 実装)
    を retrieval 経路で活用、A-MEM の top-k similarity + link traversal 戦略。

    Args:
        node_id: 起点 memory entry の id
        depth: traversal 最大深さ (G-lite 推奨 1)
        link_types: 追従する link_type tuple (None で全 type)
        min_confidence: 未満の link は辿らない (G-lite 推奨 0.7 = 既存 storage 閾値)
        top_n_per_depth: 各 depth で confidence 上位 N 件のみ展開
        visited: 循環防止の visited memory id set (外部呼出は None)

    Returns:
        [{"memory_entry": dict, "via_link": dict,
          "depth": int, "strength_hint": float}, ...]

    API 契約 (full 見据え):
        strength_hint は G-lite では via_link.confidence を流用、
        11-D Phase 3 (Physarum) で strength field に差し替え可能。
        呼び手は「retrieval 順序付けに使う float」として扱うだけ、
        source を知らない設計で後付け拡張に壊れない。
    """
    if not node_id or depth <= 0:
        return []
    visited = set(visited) if visited else set()
    visited.add(node_id)
    all_links = list_links(limit=10000)
    return _traverse_depth(
        node_id, 1, depth, all_links, visited,
        link_types, min_confidence, top_n_per_depth,
    )


def _traverse_depth(
    node_id: str,
    current_depth: int,
    max_depth: int,
    all_links: list,
    visited: set,
    link_types: Optional[tuple],
    min_confidence: float,
    top_n_per_depth: int,
) -> list:
    """follow_links の再帰実装 (current_depth 1 始まり)."""
    if current_depth > max_depth:
        return []

    outgoing = [l for l in all_links if l.get("from_id") == node_id]
    filtered = []
    for l in outgoing:
        lt = l.get("link_type", "none")
        if lt == "none":
            continue
        if link_types is not None and lt not in link_types:
            continue
        if float(l.get("confidence", 0.0)) < min_confidence:
            continue
        filtered.append(l)
    filtered.sort(key=lambda l: float(l.get("confidence", 0.0)), reverse=True)
    near = filtered[:top_n_per_depth]

    results = []
    for link in near:
        to_id = link.get("to_id")
        if not to_id or to_id in visited:
            continue
        target_entry = _find_memory_entry_by_id(to_id)
        if target_entry is None:
            continue
        visited.add(to_id)
        results.append({
            "memory_entry": target_entry,
            "via_link": link,
            "depth": current_depth,
            "strength_hint": float(link.get("confidence", 0.0)),
        })
        if current_depth < max_depth:
            child = _traverse_depth(
                to_id, current_depth + 1, max_depth,
                all_links, visited,
                link_types, min_confidence, top_n_per_depth,
            )
            results.extend(child)
    return results


def _find_memory_entry_by_id(entry_id: str) -> Optional[dict]:
    """登録済 network を走査して entry_id を持つ memory entry を見つける.

    G-lite では coarse (全 tag 走査、network n 増加で線形)。
    11-D Phase 7 migration で索引化を検討可。
    """
    from core.memory import list_records
    from core.tag_registry import list_registered_tags
    for tag in list_registered_tags():
        try:
            recs = list_records(tag, limit=10000)
        except Exception:
            continue
        for r in recs:
            if r.get("id") == entry_id:
                return r
    return None
