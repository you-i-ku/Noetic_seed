"""Slice 6.5 Step 1+2: preference distribution C + vMF mixture density (案 ④ identity-anchored).

state.self string 群 (可変層、NAME_KEY 除く) を embedding 周辺分布 C に変換し、
vMF (von Mises-Fisher) mixture で log preference density を計算する module。Active
Inference の preference distribution p*(o | self) を構成する核 — 案 ④
identity-anchored dynamic (Phase A 着手前、ゆう確定) の入口。

設計原則 (memory feedback literal):
  - llm_as_brain / freedom_to_die: identity は LLM 内部参照ではなく構造的 distribution
  - drift_is_not_developer_error: 「現在 identity に沿う」(homeostatic ではない)
  - no_biological_mimicry: 数学構造 (Gaussian / vMF mixture) のみ参照、生物模倣なし

Self-Prior paper (arXiv:2504.11075v2) 借用範囲 (Phase A 6 軸 literal 検証):
  - ② EFE 数式骨格 (本 module は EFE の C 入力側)
  - ④ density machinery (Step 2 で vMF mixture 採用、v1.1 再反転 2026-05-11)
  - ①③⑤⑥ は Noetic 独自 (o^E 削除 / identity 文 source / 空 → 均一 sentinel / 現在 identity)

Step 2 estimator 選択 (v1.1 ゆう確定、2026-05-11):
  - NSF 不採用: 1024 次元 × N=1-10 で「過学習以前に統計的同定不能」(Codex Q1 諮問結果)
  - vMF mixture 採用: L2-normalized embedding と幾何整合 (球面 support)、closed-form、
    Noetic 他箇所 (entity_resolver / memory.py 等) の cosine 系思想と統一
  - 軽量代替 (Gaussian / KDE / NSF) は段階N で iku self-mod 移行時に拡張点として開く

論点 ε (人 decide vs iku decide、PLAN §11.2.3 v1.1):
  - 本 module の estimator 形式選択 (vMF) は現時点で人 decide 暫定
  - iku は read_file で本 module を観察可能、memory_graph で preference 構造を扱える
  - 段階N で iku 自身が estimator 切替を decide できる signal / pressure 機構を実装予約
  - 既存 段階12 Self Expansion 機構で iku は理論上書換可能 (現状 trigger 機構なし)

PLAN reference: WORLD_MODEL_DESIGN/INFO_GAIN_EFE_REDESIGN_PLAN.md §5.1 + §5.2 (v1.1)
"""
import math
from typing import Callable, Optional

from scipy import special as _scipy_special

from core.embedding import _embed_sync, is_vector_ready


NAME_KEY = "name"
DEFAULT_CONFIDENCE = 0.7
EPS = 1e-3
UNIFORM_VARIANCE = 1.0 / EPS

_DEFAULT_DIM = 1024

# ============================================================
# V10 Sedimentary C (2026-07-05): 堆積資格フィルタ定数
# ============================================================
# 主体性 = 時定数 × フィルタの設計。1 cycle の LLM 出力は C に直結させず
# (V07.5 A2/A3 維持)、資格を満たした「堆積」だけが C の成分になる。
# - 時定数: C 再構築は cycle start snapshot のみ (A2)。reflection NOTES は
#   reflect 発火 (≈10 cycle 毎) + cycle 跨ぎでしか C に到達しない = 構造的低速化
# - フィルタ: confidence (自己評価) / attempts (関心の持続) / 上限 cap (mixture 安定)
SEDIMENT_CONF_MIN = 0.7             # opinion (reflection NOTES) の confidence 資格下限
SEDIMENT_MAX_OPINIONS = 8           # opinion 成分上限 (confidence 降順)
SEDIMENT_MAX_PENDING = 5            # pending 成分上限 (attempts 降順)
SEDIMENT_PENDING_MIN_ATTEMPTS = 3   # 持続関心の資格 (attempts 下限、1-2 回は通過ノイズ扱い)
SEDIMENT_PENDING_CONF_SCALE = 10.0  # attempts → confidence 変換分母 (min(1, attempts/10))


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


def compute_C(state: dict, load_memories_fn=None) -> dict:
    """state 全体 → 多源 preference distribution C (V10 Sedimentary C、2026-07-05)。

    compute_C_from_self (identity 単源) の後継 entry point。3 つの source から
    vMF mixture 成分を構成する:

      ① identity 層: state.self の可変層 string (NAME_KEY 除外、既存資格規則そのまま)。
         confidence は state["_efe_self_confidence"] (欠損 key は DEFAULT_CONFIDENCE)。
      ② opinion 堆積層: reflection NOTES (origin=="reflection") のうち
         metadata.confidence >= SEDIMENT_CONF_MIN のもの。confidence 降順で
         SEDIMENT_MAX_OPINIONS 件まで。entry に embedding があれば再 embed せず流用。
      ③ pending 持続関心層: state.pending の type=="pending" / 未 observed /
         attempts >= SEDIMENT_PENDING_MIN_ATTEMPTS。attempts 降順で
         SEDIMENT_MAX_PENDING 件まで。confidence = min(1.0, attempts /
         SEDIMENT_PENDING_CONF_SCALE) (関心の持続が peak の鋭さに変換される)。

    設計原則 (V10 paradigm、2026-07-05 ゆう合意):
      - 「LLM は主体の素材供給者、主体は堆積のプロセス」。1 cycle の LLM 出力は
        C に直結しない (A3 維持)。資格フィルタを生き延びた蓄積だけが選好になる
      - drift_is_not_developer_error: C は「現在の関心」追従 (homeostatic ではない)。
        opinion / pending の入替りで C も動く、固定 identity への回帰項なし
      - 数学構造は compute_C_from_self と同一 (vMF mixture、weight 均等 1/n、
        variance = 1/(conf+ε))。Slice 3「重み全部 1.0」原則を weight 側で継承

    Args:
        state: Noetic state dict (self / _efe_self_confidence / pending を参照)
        load_memories_fn: memory 全 load callable (test 注入用)。None なら
            core.memory.load_all_memories を遅延 import (循環 import 回避)

    Returns:
        compute_C_from_self と同 schema + 追加 field:
            "source_breakdown": {"self": int, "opinion": int, "pending": int}
        components[*] に "source" key ("self" / "opinion" / "pending") 追加。
        全 source 空 → 均一 sentinel (compute_C_from_self 空 dict と同値)。

    Bootstrap:
        白紙 iku (self 空 + memory 空 + pending 空) → components=[] の均一 sentinel。
        update_self 1 回 → ① のみの単峰 C (= 旧 compute_C_from_self と同挙動)。
        reflect が NOTES を刻み始めると ② が、関心が持続すると ③ が堆積する。
    """
    self_confidence = state.get("_efe_self_confidence", {}) or {}

    # ① identity 層 (compute_C_from_self と同一資格規則)
    specs = []
    for k, v in (state.get("self") or {}).items():
        if k == NAME_KEY:
            continue
        conf = float(self_confidence.get(k, DEFAULT_CONFIDENCE))
        specs.append({
            "key": k,
            "text": str(v),
            "mean": None,
            "confidence": max(0.0, min(1.0, conf)),
            "source": "self",
        })

    # ② opinion 堆積層 (reflection NOTES、confidence 資格 + cap)
    if load_memories_fn is None:
        from core.memory import load_all_memories as load_memories_fn
    try:
        memories = load_memories_fn() or []
    except Exception:
        memories = []
    opinion_cands = []
    for m in memories:
        if not isinstance(m, dict) or m.get("origin") != "reflection":
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        try:
            conf = float((m.get("metadata") or {}).get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if conf < SEDIMENT_CONF_MIN:
            continue
        opinion_cands.append((conf, str(m.get("created_at", "")), m, content))
    opinion_cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
    for conf, _created, m, content in opinion_cands[:SEDIMENT_MAX_OPINIONS]:
        emb = m.get("embedding")
        specs.append({
            "key": f"opinion:{m.get('id', '')}",
            "text": content,
            "mean": emb if isinstance(emb, list) else None,
            "confidence": max(0.0, min(1.0, conf)),
            "source": "opinion",
        })

    # ③ pending 持続関心層 (attempts 資格 + cap)
    pending_cands = []
    for p in state.get("pending", []) or []:
        if not isinstance(p, dict) or p.get("type") != "pending":
            continue
        if p.get("observed_content") is not None:
            continue
        attempts = int(p.get("attempts", 1) or 1)
        if attempts < SEDIMENT_PENDING_MIN_ATTEMPTS:
            continue
        text = (p.get("content_intent") or p.get("content") or "").strip()
        if not text:
            continue
        pending_cands.append((attempts, p, text))
    pending_cands.sort(key=lambda t: t[0], reverse=True)
    for attempts, p, text in pending_cands[:SEDIMENT_MAX_PENDING]:
        specs.append({
            "key": f"pending:{p.get('id', '')}",
            "text": text,
            "mean": None,
            "confidence": min(1.0, attempts / SEDIMENT_PENDING_CONF_SCALE),
            "source": "pending",
        })

    if not specs:
        empty = compute_C_from_self({}, self_confidence=None)
        empty["source_breakdown"] = {"self": 0, "opinion": 0, "pending": 0}
        return empty

    # mean 未確定 spec を 1 batch で embed (opinion の既存 embedding は流用)
    to_embed = [s for s in specs if s["mean"] is None]
    if to_embed:
        try:
            vecs = _embed_sync([s["text"] for s in to_embed]) if is_vector_ready() else None
        except Exception:
            vecs = None
        if vecs is not None and len(vecs) == len(to_embed):
            for s, vec in zip(to_embed, vecs):
                s["mean"] = vec

    n = len(specs)
    weight = 1.0 / n
    components = []
    source_keys = []
    per_key_confidence = {}
    per_key_variance = {}
    breakdown = {"self": 0, "opinion": 0, "pending": 0}
    for s in specs:
        variance = 1.0 / (s["confidence"] + EPS)
        components.append({
            "key": s["key"],
            "mean": s["mean"],
            "variance": variance,
            "weight": weight,
            "source": s["source"],
        })
        source_keys.append(s["key"])
        per_key_confidence[s["key"]] = s["confidence"]
        per_key_variance[s["key"]] = variance
        breakdown[s["source"]] += 1

    return {
        "components": components,
        "source_keys": source_keys,
        "per_key_confidence": per_key_confidence,
        "per_key_variance": per_key_variance,
        "C_entropy": _mixture_entropy_upper_bound(components, _DEFAULT_DIM),
        "n_components": n,
        "source_breakdown": breakdown,
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

    Step 1 で導入、Step 2 (v1.1) で estimate_density() による真の log density が並走。
    本関数は C_entropy 出力 (Step 1 互換) のため温存、entropy 真値は将来 Monte Carlo 等で再評価。
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


def estimate_density(
    components: list,
    method: str = "vmf",
) -> Callable[[list], float]:
    """preference distribution C から log density 関数 log_p(x) を構築。

    Args:
        components: compute_C_from_self の返り値 "components" list
            (各 {"key", "mean", "variance", "weight"} dict、mean は 1024 次元 list or None)
        method: "vmf" (default、von Mises-Fisher mixture)。
            将来 "gaussian"/"kde"/"nsf" を段階N で iku self-mod 移行時に拡張点として開く
            (現在は "vmf" のみ実装、他は NotImplementedError)

    Returns:
        log_p(x) -> float: 観測 x (1024 次元 L2-normalized list) に対する
            log preference density (nat unit)

    vMF mixture 数式 (closed-form):
        log p*(x | C) = logsumexp_k [log w_k + log f_vMF(x | μ_k, κ_k)]
        log f_vMF(x | μ, κ) = (d/2-1) log κ - (d/2) log(2π) - log I_{d/2-1}(κ) + κ · μ·x

    κ_k は variance_k から逆数で導出:
        κ_k = 1 / (variance_k + ε)     # 高 conf → 高 κ → 鋭い peak、低 conf → 低 κ → 広い peak

    Bootstrap (§5.3 + 案 ε 整合):
        components 空 or 全 mean=None (bge-m3 unavailable) → uniform sentinel
        log_p(x) = -log(surface_area_S^{d-1})  (一定値、球面上均一分布)

    論点 ε 整合 (PLAN §11.2.3 v1.1):
        estimator 形式 (vMF) は人 decide 暫定。method 引数を将来 iku self-mod 経路として開く。
    """
    if method != "vmf":
        raise NotImplementedError(
            f"estimate_density: method='{method}' は段階N で iku self-mod 移行時に拡張予定。"
            " 現在は 'vmf' のみ採用 (PLAN §11.2.3 v1.1、論点 ε)。"
        )
    return _build_vmf_log_density(components)


def _build_vmf_log_density(components: list) -> Callable[[list], float]:
    """vMF mixture の log density closure を構築 (estimate_density の内部)。

    bge-m3 unavailable で mean=None になった component は skip、すべて invalid なら
    uniform sentinel に縮退 (Noetic 哲学: 起動失敗で固まらない、Step 1 graceful と同じ思想)。
    """
    valid = [c for c in components if c.get("mean") is not None]

    if not valid:
        uniform_log = -_log_surface_area_unit_sphere(_DEFAULT_DIM)
        return lambda x: uniform_log

    prepared = []
    for c in valid:
        var = float(c["variance"])
        kappa = 1.0 / (var + EPS)
        log_w = math.log(float(c["weight"]) + EPS)
        log_norm = _log_vmf_normalizing_constant(kappa, _DEFAULT_DIM)
        prepared.append({
            "mean": c["mean"],
            "kappa": kappa,
            "log_w": log_w,
            "log_norm": log_norm,
        })

    def log_p(x: list) -> float:
        contribs = []
        for p in prepared:
            dot = sum(xi * mui for xi, mui in zip(x, p["mean"]))
            contribs.append(p["log_w"] + p["log_norm"] + p["kappa"] * dot)
        return _logsumexp(contribs)

    return log_p


def _log_vmf_normalizing_constant(kappa: float, dim: int) -> float:
    """vMF 分布の log 正規化定数 log C_d(κ)。

    log C_d(κ) = (d/2 - 1) log κ - (d/2) log(2π) - log I_{d/2-1}(κ)

    log I_v(κ) 計算戦略 (Codex review breakpoint ① P1 指摘 2026-05-11 反映):
      1. scipy.special.ive(v, κ) が finite > 0 → log ive(v, κ) + κ で安定計算
         (ive(v, κ) = exp(-|κ|) · I_v(κ) scaled、1024 次元 overflow 回避)
      2. ive が underflow (= 0) → regime 判定して正しい asymptotic を選ぶ:
         - **κ < v (small-κ / high-order、Noetic typical: κ≈0-1, v=511)**:
              I_v(κ) ≈ (κ/2)^v / Γ(v+1)   (Bessel power series 主項、高 v で
              1 次補正 κ²/(4(v+1)) ≪ 主項のため主項のみで実用十分)
              → log I_v(κ) ≈ v · log(κ/2) - lgamma(v+1)
         - **κ ≥ v (large-κ)**:
              I_v(κ) ≈ e^κ / sqrt(2π κ)
              → log I_v(κ) ≈ κ - 0.5 log(2π κ)

    Codex P1 (2026-05-11) 旧 fallback (large-κ 近似のみ) が Noetic typical regime
    (小 κ + 高 v) で数千 nat 単位の正規化定数誤差を出していた問題の修正。
    """
    v = dim / 2.0 - 1.0
    if kappa <= 0:
        return -_log_surface_area_unit_sphere(dim)

    ive_val = _scipy_special.ive(v, kappa)
    if ive_val > 0 and math.isfinite(ive_val):
        log_iv = math.log(ive_val) + kappa
    elif kappa < v:
        log_iv = v * math.log(kappa / 2.0) - math.lgamma(v + 1.0)
    else:
        log_iv = kappa - 0.5 * math.log(2.0 * math.pi * kappa)

    return (
        v * math.log(kappa)
        - (dim / 2.0) * math.log(2.0 * math.pi)
        - log_iv
    )


def _log_surface_area_unit_sphere(dim: int) -> float:
    """単位球面 S^{d-1} の log surface area: log(2 · π^(d/2) / Γ(d/2))。"""
    return math.log(2.0) + (dim / 2.0) * math.log(math.pi) - math.lgamma(dim / 2.0)


def _logsumexp(values: list) -> float:
    """log Σ exp(values) の数値安定計算 (max shift)。"""
    if not values:
        return float("-inf")
    m = max(values)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(v - m) for v in values))
