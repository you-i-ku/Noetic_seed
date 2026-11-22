"""bge-m3 ONNX埋め込み・ベクトル類似度"""
import math
import re
from collections import OrderedDict
from typing import Optional

try:
    import numpy as np
    _numpy_available = True
except ImportError:
    _numpy_available = False

# === bge-m3 ONNX ===
_onnx_session = None
_onnx_tokenizer = None
_onnx_tried = False

# === V07.5 commit 1 (Codex AXIS 4 Medium): (text) → vector cache ===
# bge-m3 5x encode latency 対策、LRU cache cap 1000 entries。
# jepa_runtime の policy_embedding (5 候補 / cycle) で同じ tool+intent 文字列が
# 反復出現する想定 (例: 同じ tool を別 cycle で再度評価)、cache hit 時 ONNX 呼出 skip。
# thread-safe: Noetic は basically single-thread (event_emitter subscriber 同期、
# controller cycle serial)、OrderedDict GIL 下で atomic。
_EMBEDDING_CACHE: "OrderedDict[str, list]" = OrderedDict()
_EMBEDDING_CACHE_MAX = 1000

def _load_bge_m3():
    """bge-m3 ONNXモデルを遅延初期化で取得（HuggingFaceから自動ダウンロード）"""
    global _onnx_session, _onnx_tokenizer, _onnx_tried
    if _onnx_tried:
        return _onnx_session is not None
    _onnx_tried = True
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        import onnxruntime as ort

        model_path = hf_hub_download("BAAI/bge-m3", "onnx/model.onnx")
        hf_hub_download("BAAI/bge-m3", "onnx/model.onnx_data")
        tok_path = hf_hub_download("BAAI/bge-m3", "onnx/tokenizer.json")

        _onnx_tokenizer = Tokenizer.from_file(tok_path)
        _onnx_tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        _onnx_tokenizer.enable_truncation(max_length=512)

        _onnx_session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        return True
    except ImportError:
        return False
    except Exception:
        return False

def _embed_sync(texts: list) -> list | None:
    """bge-m3 ONNX（CPU同期）でembedding取得"""
    if not _numpy_available or not _load_bge_m3():
        return None
    try:
        encoded = _onnx_tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        outputs = _onnx_session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )

        embeddings = outputs[0]
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        pooled = (embeddings * mask).sum(axis=1) / mask.sum(axis=1)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / norms

        return [vec.tolist() for vec in pooled]
    except Exception:
        return None

def _embed_with_cache(text: str) -> Optional[list]:
    """単一 text の embedding を cache 経由で取得。

    V07.5 commit 1 (Codex AXIS 4 Medium): bge-m3 5x encode latency 対策。
    cache hit 時 ONNX 呼出 skip、miss 時 _embed_sync で取得 + cache 追加 (LRU)。

    Returns:
        1024D vector (list[float]) on success / cache hit、None on failure。
    """
    if text in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE.move_to_end(text)
        return _EMBEDDING_CACHE[text]
    result = _embed_sync([text])
    if result is None or len(result) != 1:
        return None
    vec = result[0]
    _EMBEDDING_CACHE[text] = vec
    if len(_EMBEDDING_CACHE) > _EMBEDDING_CACHE_MAX:
        _EMBEDDING_CACHE.popitem(last=False)
    return vec


def _embed_batch_with_cache(texts: list) -> Optional[list]:
    """複数 text の embedding を cache 経由で取得 (batch mode)。

    V07.5 commit 1 (Codex AXIS 4 Medium): cache miss だけ _embed_sync で 1 回 batch
    呼出、cache hit は scan で取得。結果は input texts 順に整列。

    Args:
        texts: encode 対象 text list (重複あり可)。

    Returns:
        各 text の 1024D vector (list[list[float]]) on success、None on failure。
        empty input は [] を返す (no-op success)。
    """
    if not texts:
        return []
    cached: dict = {}
    miss_texts: list = []
    for t in texts:
        if t in _EMBEDDING_CACHE:
            _EMBEDDING_CACHE.move_to_end(t)
            cached[t] = _EMBEDDING_CACHE[t]
        elif t not in cached:
            miss_texts.append(t)
    if miss_texts:
        miss_results = _embed_sync(miss_texts)
        if miss_results is None or len(miss_results) != len(miss_texts):
            return None
        for t, vec in zip(miss_texts, miss_results):
            cached[t] = vec
            _EMBEDDING_CACHE[t] = vec
            if len(_EMBEDDING_CACHE) > _EMBEDDING_CACHE_MAX:
                _EMBEDDING_CACHE.popitem(last=False)
    return [cached[t] for t in texts]


def cosine_similarity(a: list, b: list) -> float:
    """Pure Python cosine similarity"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# === ベクトル初期化状態 ===
_vector_ready = False


def is_vector_ready() -> bool:
    """bge-m3 ONNX 初期化済みか返す (関数経由で呼出時の最新値を返す)。

    段階11-C hotfix (2026-04-24、犯人 C): 他モジュールで
    `from core.embedding import _vector_ready` していたため、Python の
    import スナップショット仕様で import 時の False が固定化されていた。
    `_init_vector()` が後で True に更新しても import 先には反映されず、
    memory.py の link 生成経路 (`if _vector_ready:`) が永遠に False 分岐
    → generate_links_for に embed_fn=None で渡る → early return で
    memory_links.jsonl が 11-B Phase 4 以降一度も生成されなかった根本原因。
    関数経由で参照することで呼出時の最新値を取得、スナップショット封じ。
    """
    return _vector_ready


def _init_vector():
    """bge-m3 ONNX埋め込みを初期化"""
    global _vector_ready
    try:
        test = _embed_sync(["test"])
        if test:
            _vector_ready = True
            print("  (ベクトル類似度: bge-m3 ONNX/CPU)")
    except Exception as e:
        print(f"  (ベクトル初期化失敗、キーワード比較にフォールバック: {e})")

def _compare_expect_result(expect: str, result: str) -> str:
    """expectとresultを比較。ベクトル類似度優先、フォールバックでキーワード比較"""
    if not expect or not result:
        return ""

    if _vector_ready:
        try:
            vecs = _embed_sync([expect, result])
            if vecs and len(vecs) == 2:
                sim = cosine_similarity(vecs[0], vecs[1])
                sim_pct = round(sim * 100)
                if "エラー" in result:
                    return f"失敗({sim_pct}%)"
                return f"{sim_pct}%"
        except Exception:
            pass

    # フォールバック: キーワード一致
    expect_tokens = set(re.findall(r'\w+', expect.lower()))
    result_tokens = set(re.findall(r'\w+', result.lower()))
    if not expect_tokens:
        return "不明"
    overlap = expect_tokens & result_tokens
    ratio = len(overlap) / len(expect_tokens)
    if "エラー" in result:
        return "失敗"
    if ratio > 0.3:
        return "一致"
    elif ratio > 0.1:
        return "部分一致"
    else:
        return "不一致"
