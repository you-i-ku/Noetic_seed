"""段階11-D Phase 5 Step 5.1: cluster 推定 (posterior、永続化せず reflect 毎に再推定)。

11-D 4 本柱の 3 つ目「reflect → cluster 推定」の基盤。
PLAN §5 Phase 5 Step 5.1 確定値: 案 H (hybrid) 初期 — embedding pre-cluster + LLM label。

設計原則:
- cluster は posterior (= 観察を蓄積してから事後推定)、永続化しない
- 各 reflect 呼出で再推定 (label drift は許容、一貫性 metric は別途観察)
- LLM as brain: clustering 構造は数理 (K-means)、prompt 強制なし
- 依存最小化文化 (Phase 4 判断 K 整合): numpy 既使用 (embedding.py)、sklearn 不使用

graceful fallback:
- numpy / vector_ready が偽 → 1 cluster 全 memory (method=fallback_no_vector)
- embed 失敗 → 1 cluster 全 memory (method=fallback_embed_failed)
- memories 空 → []
- LLM 失敗 → label="" (cluster 自体は返す)
"""
import math
import uuid
from typing import Callable, Optional

try:
    import numpy as np
    _numpy_available = True
except ImportError:
    _numpy_available = False

from core.embedding import is_vector_ready, _embed_sync


def compute_default_n_clusters(memory_count: int) -> int:
    """memory 数から動的に cluster 数を派生 (sqrt(N)、floor 2)。

    PLAN §11-4「新規マジックナンバー 0 維持」整合 — n_clusters を固定値で持たず、
    内部状態 (memory 蓄積量) から派生する。iku の蓄積成長と cluster 粒度が
    連動する。

    floor=2 は数学的必然 (1 cluster は cluster と呼べない)。ceiling は設けず、
    sqrt の自然増加に任せる (smoke 4 段目で観察、必要なら導入判断)。
    """
    return max(2, int(math.sqrt(memory_count)))


def _kmeans_simple(vectors, n_clusters: int, max_iter: int = 10, seed: int = 42):
    """numpy 簡易 K-means (Lloyd 反復、cosine 系 random init)。

    bge-m3 出力は L2 正規化済前提なので cosine similarity = dot product。
    sklearn 不使用 (Phase 4 判断 K 同精神 — 用途に十分な近似で済ます)。

    Args:
        vectors: (N, D) ndarray、L2 正規化済前提
        n_clusters: cluster 数 (vectors 数より多いなら自動縮小)
        max_iter: Lloyd 反復上限
        seed: 再現性のための random seed

    Returns:
        labels: (N,) int ndarray、各 vector の cluster index
    """
    n = len(vectors)
    if n == 0:
        return np.array([], dtype=int)
    if n <= n_clusters:
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=n_clusters, replace=False)
    centroids = vectors[init_idx].astype(np.float32, copy=True)

    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iter):
        # cosine similarity (正規化済前提) → dot product 最大化と等価
        dots = vectors @ centroids.T  # (N, K)
        new_labels = np.argmax(dots, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # centroids 更新 = cluster 内 vector 平均 + 再正規化
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                mean = vectors[mask].mean(axis=0)
                norm = np.linalg.norm(mean)
                if norm > 0:
                    centroids[k] = mean / norm
    return labels


def _llm_label_for_cluster(memories: list, llm_call_fn: Optional[Callable]) -> str:
    """cluster 内 memory のサンプルから LLM が短い label を生成。

    cost 抑制のため代表 3 件まで、max_tokens=30。LLM 失敗時は空文字。
    """
    if not memories or llm_call_fn is None:
        return ""
    sample_size = min(3, len(memories))
    sample_text = "\n".join(
        f"- {(m.get('content') or '')[:120]}"
        for m in memories[:sample_size]
    )
    prompt = (
        "以下の memory 群に共通する主題を、短い 1 行 (15 字以内) で表現してください。\n"
        "ラベルのみ出力、説明や記号は不要。\n\n"
        f"{sample_text}\n\n主題:"
    )
    try:
        text = llm_call_fn(prompt, max_tokens=30, temperature=0.2)
    except Exception:
        return ""
    if not text:
        return ""
    lines = str(text).strip().splitlines()
    if not lines:
        return ""
    label = lines[0].strip().strip('"\'「」 .。:：-')
    return label[:30]


def estimate_clusters(
    memories: list,
    method: str = "hybrid",
    n_clusters: Optional[int] = None,
    llm_call_fn: Optional[Callable] = None,
) -> list[dict]:
    """memory list を cluster 化、label と membership を返す。

    PLAN §5 Phase 5 Step 5.1: cluster は posterior (永続化せず reflect 毎に再推定)。

    Args:
        memories: memory entry list (各 entry に "id", "content")
        method: "embedding" / "llm" / "hybrid" (default)
        n_clusters: K-means の cluster 数。None で動的派生 (sqrt(N) 整合)。
            明示指定で override 可 (smoke / test 用)。
        llm_call_fn: hybrid / llm method 用の LLM 呼出関数

    Returns:
        [{"cluster_id": str, "label": str, "memory_ids": [...], "method": str}, ...]
        memories 空 → []。vector / embed 失敗 → 1 cluster fallback。
    """
    if not memories:
        return []

    if n_clusters is None:
        n_clusters = compute_default_n_clusters(len(memories))

    if not _numpy_available or not is_vector_ready():
        return [{
            "cluster_id": uuid.uuid4().hex[:8],
            "label": "",
            "memory_ids": [m.get("id", "") for m in memories],
            "method": "fallback_no_vector",
        }]

    contents = [(m.get("content") or "") for m in memories]
    vecs = _embed_sync(contents)
    if not vecs:
        return [{
            "cluster_id": uuid.uuid4().hex[:8],
            "label": "",
            "memory_ids": [m.get("id", "") for m in memories],
            "method": "fallback_embed_failed",
        }]

    vectors = np.array(vecs, dtype=np.float32)
    # Q2 観察 log (ゆう 2026-04-26): cap ライン実証判断のための snapshot。
    # smoke 後に raw_log を grep して N と nbytes の関係を確認、必要なら cap 導入。
    print(f"  [cluster_estimation] N={len(memories)} vectors.nbytes={vectors.nbytes}")
    effective_k = min(n_clusters, len(memories))
    labels = _kmeans_simple(vectors, effective_k)

    clusters = []
    for k in range(effective_k):
        member_indices = [i for i, lbl in enumerate(labels) if int(lbl) == k]
        if not member_indices:
            continue
        member_memories = [memories[i] for i in member_indices]
        member_ids = [m.get("id", "") for m in member_memories]

        label = ""
        if method in ("hybrid", "llm") and llm_call_fn is not None:
            label = _llm_label_for_cluster(member_memories, llm_call_fn)

        clusters.append({
            "cluster_id": uuid.uuid4().hex[:8],
            "label": label,
            "memory_ids": member_ids,
            "method": method,
        })

    return clusters
