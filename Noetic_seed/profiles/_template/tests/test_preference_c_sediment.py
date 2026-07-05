"""V10 Sedimentary C (compute_C 多源化) の識別力 test。

対象: core/preference_distribution.py compute_C (2026-07-05)
PLAN reference: WORLD_MODEL_DESIGN/V10_SEDIMENTARY_C_PLAN.md (後書き正典)

docstring contract ↔ test 対応 (CLAUDE.md §6 literal):
  ① identity 層 (NAME_KEY 除外、self_confidence)      → case A / I
  ② opinion 堆積層 (origin + confidence 資格 + cap)    → case B / D / F
  ③ pending 持続関心層 (attempts 資格 + conf 変換)     → case C
  weight 均等 1/n (Slice 3 原則継承)                   → case G
  全 source 空 → 均一 sentinel                          → case E
  estimate_density 互換 (下流 _compute_log_p_now)      → case H

識別力 fixture (CLAUDE.md §5 literal): 各資格条件に「資格を満たさない decoy」を
併置し、フィルタを無視する誤実装 (全件取込 / 片側条件無視) が必ず fail する構成。
edge 4 種 (空入力 / 全一致 / 部分一致 / 完全不一致) は case E / B / B,C / I が対応。

使い方:
  cd Noetic_seed/profiles/_template
  .venv python -m pytest tests/test_preference_c_sediment.py -q
"""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import preference_distribution as pd_mod
from core.preference_distribution import (
    DEFAULT_CONFIDENCE,
    EPS,
    SEDIMENT_CONF_MIN,
    SEDIMENT_MAX_OPINIONS,
    SEDIMENT_PENDING_MIN_ATTEMPTS,
    compute_C,
    compute_C_from_self,
    estimate_density,
)


@pytest.fixture
def deterministic_embed():
    """bge-m3 を deterministic onehot 単位 vector に差し替え (既存 fixture pattern 踏襲)。"""
    def fake_embed(texts):
        out = []
        for t in texts:
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % 1024
            vec = [0.0] * 1024
            vec[idx] = 1.0
            out.append(vec)
        return out

    orig_embed = pd_mod._embed_sync
    orig_ready = pd_mod.is_vector_ready
    pd_mod._embed_sync = fake_embed
    pd_mod.is_vector_ready = lambda: True
    try:
        yield fake_embed
    finally:
        pd_mod._embed_sync = orig_embed
        pd_mod.is_vector_ready = orig_ready


def _mem(mid, origin, content, confidence, embedding=None, created_at=""):
    return {
        "id": mid,
        "origin": origin,
        "content": content,
        "metadata": {"confidence": confidence},
        "embedding": embedding,
        "created_at": created_at,
    }


def _pend(pid, attempts, content_intent, observed=None, p_type="pending"):
    return {
        "id": pid,
        "type": p_type,
        "attempts": attempts,
        "content_intent": content_intent,
        "observed_content": observed,
    }


def test_case_a_self_only_matches_legacy_behavior(deterministic_embed):
    """case A (§6 ①): self のみの state では compute_C_from_self と同値の C。

    識別力: NAME_KEY 混入誤実装なら source_keys に "name" が入り fail。
    """
    state = {
        "self": {"name": "iku", "identity": "I observe", "style": "quiet"},
        "_efe_self_confidence": {"identity": 0.9},
        "pending": [],
    }
    c = compute_C(state, load_memories_fn=lambda: [])
    legacy = compute_C_from_self(state["self"], self_confidence=state["_efe_self_confidence"])

    assert sorted(c["source_keys"]) == sorted(legacy["source_keys"]) == ["identity", "style"]
    assert "name" not in c["source_keys"]
    assert c["per_key_confidence"]["identity"] == 0.9
    assert c["per_key_confidence"]["style"] == DEFAULT_CONFIDENCE
    assert c["source_breakdown"] == {"self": 2, "opinion": 0, "pending": 0}
    assert c["n_components"] == 2


def test_case_b_opinion_qualification_filters(deterministic_embed):
    """case B (§6 ②): confidence >= 0.7 かつ origin=="reflection" のみ堆積。

    識別力 decoy 3 種:
      - conf 0.5 (資格未満) → 除外。閾値無視誤実装なら混入し fail
      - origin="tool" で conf 0.9 → 除外。origin 無視誤実装なら混入し fail
      - content 空 → 除外
    """
    memories = [
        _mem("m1", "reflection", "リンクの発見が続くと理解が深まる", 0.9),
        _mem("m2", "reflection", "低確信の思いつき", 0.5),          # decoy: conf 未満
        _mem("m3", "tool", "tool 由来の高 conf 記録", 0.9),          # decoy: origin 違い
        _mem("m4", "reflection", "", 0.9),                           # decoy: content 空
        _mem("m5", "reflection", "観察には周期性がある", 0.75),
    ]
    state = {"self": {}, "_efe_self_confidence": {}, "pending": []}
    c = compute_C(state, load_memories_fn=lambda: memories)

    assert c["source_breakdown"] == {"self": 0, "opinion": 2, "pending": 0}
    assert set(c["source_keys"]) == {"opinion:m1", "opinion:m5"}
    assert c["per_key_confidence"]["opinion:m1"] == 0.9
    assert c["per_key_confidence"]["opinion:m5"] == 0.75


def test_case_c_pending_qualification_and_conf_mapping(deterministic_embed):
    """case C (§6 ③): attempts >= 3 / 未 observed / type=="pending" のみ堆積、
    confidence = min(1.0, attempts/10)。

    識別力 decoy 3 種:
      - attempts=1 (資格未満) → 除外
      - observed 済 (解消済) → 除外。未 observed 条件無視誤実装なら混入し fail
      - type="input" → 除外
    conf 変換: attempts=5 → 0.5 / attempts=20 → 1.0 clamp (変換省略誤実装なら fail)。
    """
    pending = [
        _pend("p1", 5, "claude channel の応答リズムを掴みたい"),
        _pend("p2", 1, "一度きりの通過ノイズ"),                       # decoy: attempts 未満
        _pend("p3", 8, "解消済のこだわり", observed="done"),          # decoy: observed 済
        _pend("p4", 9, "外部入力", p_type="input"),                   # decoy: type 違い
        _pend("p5", 20, "長期のこだわり"),
    ]
    state = {"self": {}, "_efe_self_confidence": {}, "pending": pending}
    c = compute_C(state, load_memories_fn=lambda: [])

    assert c["source_breakdown"] == {"self": 0, "opinion": 0, "pending": 2}
    assert set(c["source_keys"]) == {"pending:p1", "pending:p5"}
    assert c["per_key_confidence"]["pending:p1"] == pytest.approx(0.5)
    assert c["per_key_confidence"]["pending:p5"] == pytest.approx(1.0)  # clamp
    # variance = 1/(conf+ε): 高 attempts → 鋭い peak
    assert c["per_key_variance"]["pending:p5"] < c["per_key_variance"]["pending:p1"]


def test_case_d_opinion_cap_keeps_highest_confidence(deterministic_embed):
    """case D (§6 ②): 資格通過が SEDIMENT_MAX_OPINIONS 超なら confidence 降順 cap。

    識別力: 10 件全部 conf 資格通過、conf を 0.99..0.90 で識別可能に振り、
    cap されるのは最低 conf の 2 件 (m8/m9) であることを content で検証。
    「先頭 8 件」等の順序依存誤実装なら fail (入力順は conf 昇順で与える)。
    """
    memories = [
        _mem(f"m{i}", "reflection", f"気づき {i}", 0.99 - i * 0.01)
        for i in range(9, -1, -1)  # conf 昇順 (0.90 → 0.99) で与える
    ]
    state = {"self": {}, "_efe_self_confidence": {}, "pending": []}
    c = compute_C(state, load_memories_fn=lambda: memories)

    assert c["n_components"] == SEDIMENT_MAX_OPINIONS
    kept = set(c["source_keys"])
    assert "opinion:m8" not in kept and "opinion:m9" not in kept  # 最低 conf 2 件が cap
    assert "opinion:m0" in kept and "opinion:m7" in kept


def test_case_e_all_empty_returns_uniform_sentinel(deterministic_embed):
    """case E (§5 空入力 edge): 全 source 空 → compute_C_from_self({}) と同じ均一 sentinel。"""
    state = {"self": {}, "_efe_self_confidence": {}, "pending": []}
    c = compute_C(state, load_memories_fn=lambda: [])
    legacy_empty = compute_C_from_self({})

    assert c["components"] == []
    assert c["n_components"] == 0
    assert c["C_entropy"] == legacy_empty["C_entropy"]
    assert c["source_breakdown"] == {"self": 0, "opinion": 0, "pending": 0}


def test_case_f_opinion_embedding_reused_without_bge(deterministic_embed):
    """case F (§6 ②): memory entry の既存 embedding を再 embed せず流用。

    識別力: is_vector_ready を False に落とす。再 embed 実装なら opinion の
    mean も None になり fail。流用実装なら entry embedding がそのまま mean。
    """
    stored_vec = [0.0] * 1024
    stored_vec[7] = 1.0
    memories = [_mem("m1", "reflection", "埋め込み済の気づき", 0.9, embedding=stored_vec)]
    state = {"self": {"identity": "X"}, "_efe_self_confidence": {}, "pending": []}

    pd_mod.is_vector_ready = lambda: False  # fixture teardown で復元される
    c = compute_C(state, load_memories_fn=lambda: memories)

    by_key = {comp["key"]: comp for comp in c["components"]}
    assert by_key["opinion:m1"]["mean"] == stored_vec       # 流用
    assert by_key["identity"]["mean"] is None               # bge 不在で embed 不能


def test_case_g_equal_weights_across_sources(deterministic_embed):
    """case G (§6 weight): 混在 source でも weight は均等 1/n、Σ=1。

    識別力: source 別に重み差を付ける誤実装 (self を 2 倍等) なら fail。
    """
    memories = [_mem("m1", "reflection", "気づき", 0.9)]
    pending = [_pend("p1", 5, "持続する関心")]
    state = {"self": {"identity": "X"}, "_efe_self_confidence": {}, "pending": pending}
    c = compute_C(state, load_memories_fn=lambda: memories)

    assert c["n_components"] == 3
    weights = [comp["weight"] for comp in c["components"]]
    assert all(w == pytest.approx(1.0 / 3) for w in weights)
    assert sum(weights) == pytest.approx(1.0)


def test_case_h_downstream_estimate_density_compat(deterministic_embed):
    """case H (§6 互換): compute_C 出力が estimate_density / _compute_log_p_now
    下流でそのまま使える (components schema 互換)。

    識別力: components schema 変更 (mean/variance/weight 欠落) 誤実装なら
    estimate_density 内で KeyError / 非 finite になり fail。
    """
    import math
    memories = [_mem("m1", "reflection", "気づき", 0.9)]
    state = {"self": {"identity": "X"}, "_efe_self_confidence": {}, "pending": []}
    c = compute_C(state, load_memories_fn=lambda: memories)

    log_p = estimate_density(c["components"], method="vmf")
    probe = [0.0] * 1024
    probe[3] = 1.0
    val = log_p(probe)
    assert isinstance(val, float) and math.isfinite(val)


def test_case_i_all_disqualified_falls_back_to_self(deterministic_embed):
    """case I (§5 完全不一致 edge): memory / pending が全部資格外なら self のみの C。

    識別力: 資格フィルタ全無視の誤実装なら n_components が 3 になり fail。
    """
    memories = [_mem("m1", "reflection", "低確信", 0.3)]
    pending = [_pend("p1", SEDIMENT_PENDING_MIN_ATTEMPTS - 1, "浅い関心")]
    state = {"self": {"identity": "X"}, "_efe_self_confidence": {}, "pending": pending}
    c = compute_C(state, load_memories_fn=lambda: memories)

    assert c["source_keys"] == ["identity"]
    assert c["source_breakdown"] == {"self": 1, "opinion": 0, "pending": 0}


def test_case_j_load_failure_graceful(deterministic_embed):
    """case J (graceful): load_memories_fn 例外でも self のみで C 構築継続。

    Noetic 哲学「起動失敗で固まらない」literal (compute_C_from_self bootstrap と同思想)。
    """
    def broken_loader():
        raise OSError("memory dir unavailable")

    state = {"self": {"identity": "X"}, "_efe_self_confidence": {}, "pending": []}
    c = compute_C(state, load_memories_fn=broken_loader)

    assert c["source_keys"] == ["identity"]
    assert c["source_breakdown"]["opinion"] == 0
