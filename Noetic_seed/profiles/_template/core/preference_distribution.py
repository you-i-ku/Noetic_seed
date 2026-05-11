"""Slice 6.5 Step 1: preference distribution C (案 ④ identity-anchored).

state.self string 群 (可変層、NAME_KEY 除く) を embedding 周辺分布 C に変換する
module。Active Inference の preference distribution p*(o | self) を構成する核 — 案 ④
identity-anchored dynamic (Phase A 着手前、ゆう確定) の入口。

設計原則 (memory feedback literal):
  - llm_as_brain / freedom_to_die: identity は LLM 内部参照ではなく構造的 distribution
  - drift_is_not_developer_error: 「現在 identity に沿う」(homeostatic ではない)
  - no_biological_mimicry: 数学構造 (Gaussian mixture) のみ参照、生物模倣なし

Self-Prior paper (arXiv:2504.11075v2) 借用範囲 (Phase A 6 軸 literal 検証):
  - ② EFE 数式骨格 (本 module は EFE の C 入力側)
  - ④ density machinery (Step 2 で NSF 採用、Step 1 では mean/variance + 上界 entropy)
  - ①③⑤⑥ は Noetic 独自 (o^E 削除 / identity 文 source / 空 → 均一 sentinel / 現在 identity)

PLAN reference: WORLD_MODEL_DESIGN/INFO_GAIN_EFE_REDESIGN_PLAN.md §5.1
"""
import math
from typing import Optional

from core.embedding import _embed_sync, is_vector_ready


NAME_KEY = "name"
DEFAULT_CONFIDENCE = 0.7
EPS = 1e-3
UNIFORM_VARIANCE = 1.0 / EPS

_DEFAULT_DIM = 1024


def compute_C_from_self(
    state_self: dict,
    self_confidence: Optional[dict] = None,
    source_keys: Optional[list] = None,
) -> dict:
    """state.self → preference distribution C (案 ④ identity-anchored)。

    Args:
        state_self: state["self"] dict (key → value string)。NAME_KEY は自動除外。
        self_confidence: state["_efe_self_confidence"] dict。None なら全 key default 0.7。
        source_keys: source とする可変層 key list。None なら NAME_KEY 以外全 key。

    Returns:
        {
            "components": list[dict],     # 各 key の {key, mean, variance, weight}
            "source_keys": list[str],     # 使用 key (NAME_KEY 除外 + source_keys filter 後)
            "per_key_confidence": dict,   # key → confidence (NAME_KEY 含まず)
            "per_key_variance": dict,     # key → variance = 1/(conf+ε)
            "C_entropy": float,           # Shannon entropy (nat) 上界近似
            "n_components": int,          # 可変層 key 数
        }

    Bootstrap (§5.3、Step 7 で test 強化):
        state_self={} or 全 key 除外 → components=[]、C_entropy = uniform sentinel
        (= isotropic Gaussian variance=1/ε, dim=1024 の H_k で「均一を示す大値」)
        update_self 1 回で単峰 C、複数 key で多峰 C へ遷移。

    Note:
        per-key variance:
            variance_k = 1 / (confidence_k + ε)   (ε = 1e-3)
            高 conf → 小 variance (鋭い peak、preference 強)
            低 conf → 大 variance (広い peak、preference 弱)

        Step 1 段階の C_entropy は Gaussian mixture entropy の **上界近似**:
            H[Σ w_k N_k] ≤ -Σ w_k log w_k + Σ w_k H[N_k]
        component 間 mean 距離 (同義 keys 等) は捕捉しないため、Step 2 で NSF 採用後に
        log density 経由で真の entropy に置換予定 (§6.4 case 4 同義 keys assert は Step 2 で activate)。

        embedding 取得失敗 (bge-m3 未起動 / _embed_sync 例外) でも instantiate 失敗させない:
            components[*]["mean"] = None、ただし per_key_variance / weight は構築継続。
            Noetic 哲学: 起動失敗で固まらない (Step 7 bootstrap と同じ思想)。
    """
    if self_confidence is None:
        self_confidence = {}

    filtered = {k: v for k, v in (state_self or {}).items() if k != NAME_KEY}
    if source_keys is not None:
        filtered = {k: v for k, v in filtered.items() if k in source_keys}

    keys_used = list(filtered.keys())

    if not keys_used:
        return {
            "components": [],
            "source_keys": [],
            "per_key_confidence": {},
            "per_key_variance": {},
            "C_entropy": _gaussian_entropy(UNIFORM_VARIANCE, _DEFAULT_DIM),
            "n_components": 0,
        }

    per_key_confidence: dict = {}
    per_key_variance: dict = {}
    for k in keys_used:
        conf = float(self_confidence.get(k, DEFAULT_CONFIDENCE))
        conf = max(0.0, min(1.0, conf))
        per_key_confidence[k] = conf
        per_key_variance[k] = 1.0 / (conf + EPS)

    values = [str(filtered[k]) for k in keys_used]
    try:
        embeddings = _embed_sync(values) if is_vector_ready() else None
    except Exception:
        embeddings = None

    n = len(keys_used)
    weight = 1.0 / n
    components: list = []
    if embeddings is not None and len(embeddings) == n:
        for k, emb in zip(keys_used, embeddings):
            components.append({
                "key": k,
                "mean": emb,
                "variance": per_key_variance[k],
                "weight": weight,
            })
    else:
        for k in keys_used:
            components.append({
                "key": k,
                "mean": None,
                "variance": per_key_variance[k],
                "weight": weight,
            })

    return {
        "components": components,
        "source_keys": keys_used,
        "per_key_confidence": per_key_confidence,
        "per_key_variance": per_key_variance,
        "C_entropy": _mixture_entropy_upper_bound(components, _DEFAULT_DIM),
        "n_components": n,
    }


def _gaussian_entropy(variance: float, dim: int) -> float:
    """Isotropic Gaussian N(μ, σ^2 I_d) の Shannon entropy (nat)。

    H[N] = (d/2) (1 + log(2π σ^2))
    """
    return 0.5 * dim * (1.0 + math.log(2.0 * math.pi * variance))


def _mixture_entropy_upper_bound(components: list, dim: int) -> float:
    """Gaussian mixture H[Σ w_k N_k] の上界近似 (nat)。

    H[mixture] ≤ -Σ w_k log w_k + Σ w_k H[N_k]
    (Jensen 不等式由来、component 重複 / 重なり距離は捕捉しない緩い上界)

    Step 2 で NSF 採用後、log density 経由で真の entropy に置換予定。
    """
    if not components:
        return 0.0

    weight_term = 0.0
    component_term = 0.0
    for c in components:
        w = float(c["weight"])
        var = float(c["variance"])
        if w > 0:
            weight_term -= w * math.log(w)
        component_term += w * _gaussian_entropy(var, dim)

    return weight_term + component_term
