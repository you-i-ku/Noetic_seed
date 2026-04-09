"""bge-m3 ONNX埋め込み・ベクトル類似度"""
import math
import re

try:
    import numpy as np
    _numpy_available = True
except ImportError:
    _numpy_available = False

# === bge-m3 ONNX ===
_onnx_session = None
_onnx_tokenizer = None
_onnx_tried = False

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
