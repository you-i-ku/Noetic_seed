"""段階11-D Phase 5 Step 5.1: estimate_clusters tests.

成功条件:
  - empty memories → []
  - numpy / vector_ready 偽 → fallback (1 cluster 全 memory)
  - embed 失敗 → fallback (1 cluster 全 memory)
  - _kmeans_simple: 単一 vector / n_clusters > samples / 明確な 2 cluster 分離
  - estimate_clusters: embedding method (LLM 呼ばない、label 空)
  - estimate_clusters: hybrid method (LLM が cluster 数だけ呼ばれ label 付与)
  - LLM label の clip / 記号剥がし
  - LLM 例外 → label 空 (cluster 自体は返る)

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_reflect_cluster_estimation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import core.cluster_estimation as ce
from core.cluster_estimation import (
    estimate_clusters,
    _kmeans_simple,
    _llm_label_for_cluster,
    compute_default_n_clusters,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return bool(cond)


# ============================================================
# Section A: _kmeans_simple (純粋計算)
# ============================================================

def test_kmeans_empty():
    print("== _kmeans_simple: 空入力 → 空配列 ==")
    labels = _kmeans_simple(np.array([], dtype=np.float32).reshape(0, 2), n_clusters=3)
    return _assert(len(labels) == 0, "空 vectors → 空 labels")


def test_kmeans_single_vector():
    print("== _kmeans_simple: 1 vector + n_clusters=3 → 1 cluster ==")
    vecs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    labels = _kmeans_simple(vecs, n_clusters=3)
    return all([
        _assert(len(labels) == 1, "labels 長 1"),
        _assert(labels[0] == 0, "唯一の vector が cluster 0"),
    ])


def test_kmeans_n_greater_than_samples():
    print("== _kmeans_simple: n_clusters > サンプル数 → 各 sample 独立 cluster ==")
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = _kmeans_simple(vecs, n_clusters=5)
    return all([
        _assert(len(labels) == 2, "labels 長 2"),
        _assert(set(int(x) for x in labels) == {0, 1}, "各 sample 独立 cluster"),
    ])


def test_kmeans_separates_clear_clusters():
    print("== _kmeans_simple: 直交 2 cluster を分離 ==")
    vecs = np.array([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
        [0.01, 0.99],
    ], dtype=np.float32)
    # bge-m3 出力は L2 正規化済前提なので、ここでも正規化
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    labels = _kmeans_simple(vecs, n_clusters=2, seed=42)
    return all([
        _assert(labels[0] == labels[1], "近接ペア (0,1) 同 cluster"),
        _assert(labels[2] == labels[3], "近接ペア (2,3) 同 cluster"),
        _assert(labels[0] != labels[2], "直交ペアは別 cluster"),
    ])


# ============================================================
# Section B: estimate_clusters (graceful fallback / method 切替)
# ============================================================

def test_empty_memories():
    print("== estimate_clusters: 空 memories → [] ==")
    return _assert(estimate_clusters([]) == [], "空入力で空リスト返す")


def test_fallback_no_vector():
    print("== estimate_clusters: vector_ready 偽 → fallback (1 cluster) ==")
    orig = ce.is_vector_ready
    ce.is_vector_ready = lambda: False
    try:
        memories = [
            {"id": "m1", "content": "猫"},
            {"id": "m2", "content": "犬"},
        ]
        result = estimate_clusters(memories)
        return all([
            _assert(len(result) == 1, "fallback で 1 cluster"),
            _assert(result[0]["method"] == "fallback_no_vector", "method=fallback_no_vector"),
            _assert(set(result[0]["memory_ids"]) == {"m1", "m2"}, "全 memory が含まれる"),
            _assert(result[0]["label"] == "", "fallback の label は空"),
        ])
    finally:
        ce.is_vector_ready = orig


def test_fallback_embed_failed():
    print("== estimate_clusters: _embed_sync が None → fallback ==")
    orig_ready = ce.is_vector_ready
    orig_embed = ce._embed_sync
    ce.is_vector_ready = lambda: True
    ce._embed_sync = lambda texts: None
    try:
        memories = [{"id": "m1", "content": "猫"}]
        result = estimate_clusters(memories)
        return all([
            _assert(len(result) == 1, "embed 失敗で 1 cluster"),
            _assert(result[0]["method"] == "fallback_embed_failed", "method=fallback_embed_failed"),
            _assert(result[0]["memory_ids"] == ["m1"], "memory id 保持"),
        ])
    finally:
        ce.is_vector_ready = orig_ready
        ce._embed_sync = orig_embed


def test_estimate_embedding_method_no_llm_call():
    """embedding method は LLM を呼ばず、label が空のままで cluster 返す。"""
    print("== estimate_clusters method=embedding: LLM 不呼出、label 空 ==")
    orig_ready = ce.is_vector_ready
    orig_embed = ce._embed_sync
    ce.is_vector_ready = lambda: True
    ce._embed_sync = lambda texts: [
        [1.0, 0.0] if "猫" in t else [0.0, 1.0]
        for t in texts
    ]
    llm_calls = {"n": 0}

    def fake_llm(prompt, **kw):
        llm_calls["n"] += 1
        return "should_not_be_called"

    try:
        memories = [
            {"id": "m1", "content": "猫が好き"},
            {"id": "m2", "content": "猫はかわいい"},
            {"id": "m3", "content": "犬が好き"},
            {"id": "m4", "content": "犬はかわいい"},
        ]
        result = estimate_clusters(
            memories, method="embedding", n_clusters=2, llm_call_fn=fake_llm,
        )
        total = sum(len(c["memory_ids"]) for c in result)
        all_labels_empty = all(c["label"] == "" for c in result)
        all_method_emb = all(c["method"] == "embedding" for c in result)
        return all([
            _assert(len(result) == 2, "2 cluster 生成"),
            _assert(total == 4, "全 4 memory が割り当てられる"),
            _assert(llm_calls["n"] == 0, "LLM は 1 度も呼ばれない"),
            _assert(all_labels_empty, "embedding method の label は空"),
            _assert(all_method_emb, "method 文字列が embedding"),
        ])
    finally:
        ce.is_vector_ready = orig_ready
        ce._embed_sync = orig_embed


def test_estimate_hybrid_calls_llm_per_cluster():
    """hybrid method は cluster ごとに 1 LLM call し label 付与。"""
    print("== estimate_clusters method=hybrid: cluster 数だけ LLM call ==")
    orig_ready = ce.is_vector_ready
    orig_embed = ce._embed_sync
    ce.is_vector_ready = lambda: True
    ce._embed_sync = lambda texts: [
        [1.0, 0.0] if "猫" in t else [0.0, 1.0]
        for t in texts
    ]
    llm_calls = {"n": 0}

    def fake_llm(prompt, **kw):
        llm_calls["n"] += 1
        return "動物観察"

    try:
        memories = [
            {"id": "m1", "content": "猫が好き"},
            {"id": "m2", "content": "犬が好き"},
        ]
        result = estimate_clusters(
            memories, method="hybrid", n_clusters=2, llm_call_fn=fake_llm,
        )
        return all([
            _assert(len(result) == 2, "2 cluster 生成"),
            _assert(llm_calls["n"] == 2, "cluster 数だけ LLM call (=2)"),
            _assert(all(c["label"] == "動物観察" for c in result), "label 付与"),
            _assert(all(c["method"] == "hybrid" for c in result), "method=hybrid"),
        ])
    finally:
        ce.is_vector_ready = orig_ready
        ce._embed_sync = orig_embed


# ============================================================
# Section C: _llm_label_for_cluster
# ============================================================

def test_label_strip_quotes_and_clip():
    print("== _llm_label_for_cluster: quote 剥がし + 30 字 clip ==")
    long_label = "「ほんとに長すぎる動物観察に関する考察についてさらにいろいろなこと」"

    def fake_llm(prompt, **kw):
        return long_label

    label = _llm_label_for_cluster([{"id": "m1", "content": "猫"}], fake_llm)
    return all([
        _assert("「" not in label, "「 剥がれてる"),
        _assert("」" not in label, "」 剥がれてる"),
        _assert(len(label) <= 30, f"30 字以内 (got: {len(label)})"),
    ])


def test_label_empty_on_exception():
    print("== _llm_label_for_cluster: LLM 例外 → 空文字 ==")

    def fail_llm(prompt, **kw):
        raise RuntimeError("LLM down")

    label = _llm_label_for_cluster([{"id": "m1", "content": "猫"}], fail_llm)
    return _assert(label == "", "例外時は空文字")


def test_label_empty_on_no_fn():
    print("== _llm_label_for_cluster: llm_call_fn=None → 空文字 ==")
    label = _llm_label_for_cluster([{"id": "m1", "content": "猫"}], None)
    return _assert(label == "", "llm_call_fn=None で空文字")


# ============================================================
# Section D: 動的 n_clusters 派生 (PLAN §11-4 マジックナンバー 0 整合)
# ============================================================

def test_compute_default_n_clusters_floor():
    print("== compute_default_n_clusters: floor=2 (N=0/1/2/4) ==")
    return all([
        _assert(compute_default_n_clusters(0) == 2, "N=0 → 2 (floor)"),
        _assert(compute_default_n_clusters(1) == 2, "N=1 → 2 (floor)"),
        _assert(compute_default_n_clusters(2) == 2, "N=2 → 2 (sqrt=1.4 但し floor)"),
        _assert(compute_default_n_clusters(4) == 2, "N=4 → 2 (sqrt(4)=2)"),
    ])


def test_compute_default_n_clusters_sqrt():
    print("== compute_default_n_clusters: sqrt 自然増加 ==")
    return all([
        _assert(compute_default_n_clusters(10) == 3, "N=10 -> 3 (sqrt(10)~3.16 -> int=3)"),
        _assert(compute_default_n_clusters(50) == 7, "N=50 -> 7 (sqrt(50)~7.07 -> int=7)"),
        _assert(compute_default_n_clusters(100) == 10, "N=100 -> 10"),
        _assert(compute_default_n_clusters(200) == 14, "N=200 -> 14"),
    ])


def test_estimate_clusters_uses_dynamic_n_when_none():
    """n_clusters=None で動的派生 (sqrt) が走り、cluster 数が派生値に従う。"""
    print("== estimate_clusters: n_clusters=None で動的派生 ==")
    orig_ready = ce.is_vector_ready
    orig_embed = ce._embed_sync
    ce.is_vector_ready = lambda: True
    # 4 memory → sqrt(4)=2 cluster になるはず
    ce._embed_sync = lambda texts: [
        [1.0, 0.0] if "猫" in t else [0.0, 1.0]
        for t in texts
    ]
    try:
        memories = [
            {"id": f"m{i}", "content": "猫" if i < 2 else "犬"}
            for i in range(4)
        ]
        result = estimate_clusters(memories, method="embedding", n_clusters=None)
        # N=4 → sqrt(4)=2 → 2 cluster (memories 4 件、effective_k=min(2,4)=2)
        return _assert(len(result) == 2, f"N=4 で 2 cluster (got: {len(result)})")
    finally:
        ce.is_vector_ready = orig_ready
        ce._embed_sync = orig_embed


def test_estimate_clusters_n_explicit_overrides_dynamic():
    """n_clusters 明示指定で動的派生を override (smoke / test 用)。

    N=16、動的派生なら sqrt(16)=4 だが、明示 n=2 で override → 2 cluster になる。
    検証は override の方向性 (動的とは異なる値が反映されること)。
    """
    print("== estimate_clusters: n_clusters 明示で override ==")
    orig_ready = ce.is_vector_ready
    orig_embed = ce._embed_sync
    ce.is_vector_ready = lambda: True
    ce._embed_sync = lambda texts: [
        [1.0, 0.0] if "猫" in t else [0.0, 1.0]
        for t in texts
    ]
    try:
        memories = [
            {"id": f"m{i}", "content": "猫" if i < 8 else "犬"}
            for i in range(16)
        ]
        result = estimate_clusters(memories, method="embedding", n_clusters=2)
        return _assert(
            len(result) == 2,
            f"明示 n=2 で 2 cluster (動的派生 sqrt(16)=4 を override) (got: {len(result)})",
        )
    finally:
        ce.is_vector_ready = orig_ready
        ce._embed_sync = orig_embed


# ============================================================
# Runner
# ============================================================

def run_all():
    print("=" * 60)
    print("test_reflect_cluster_estimation.py (段階11-D Phase 5 Step 5.1)")
    print("=" * 60)
    results = [
        test_kmeans_empty(),
        test_kmeans_single_vector(),
        test_kmeans_n_greater_than_samples(),
        test_kmeans_separates_clear_clusters(),
        test_empty_memories(),
        test_fallback_no_vector(),
        test_fallback_embed_failed(),
        test_estimate_embedding_method_no_llm_call(),
        test_estimate_hybrid_calls_llm_per_cluster(),
        test_label_strip_quotes_and_clip(),
        test_label_empty_on_exception(),
        test_label_empty_on_no_fn(),
        test_compute_default_n_clusters_floor(),
        test_compute_default_n_clusters_sqrt(),
        test_estimate_clusters_uses_dynamic_n_when_none(),
        test_estimate_clusters_n_explicit_overrides_dynamic(),
    ]
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"結果: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
