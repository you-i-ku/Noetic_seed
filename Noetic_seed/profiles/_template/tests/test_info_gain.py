"""Slice 3 info_gain test (orchestration §3 P1 #3 + §5 Slice 3)。

CLAUDE.md §5 識別力 + §6 docstring 同期 遵守。各 helper / 統合 / snapshot に
discriminating fixture を仕込んで、誤実装が fail する設計。

数学的裏付け literal は core/info_gain.py docstring 参照 (情報理論 / Bayesian 推論 /
Active Inference 数学フレームワーク / CIG 数式 / A-MEM zettelkasten link 数式)。
脳のモデルとしての参照は不採用 (`feedback_no_biological_mimicry` literal、
BP-1 hotfix 2026-05-09)。

検証範囲:

A. compute_info_gain_components 統合:
   A1. 全 6 項目 + info_gain + model_resolution_gain が dict に含まれる
   A2. info_gain = novelty + effective_change + memory_link + resolution
                  + capability - redundancy_penalty (literal 加算式)
   A3. model_resolution_gain == world_model_resolution_gain (独立 metric の同値)

B. _novelty_gain (newest first 入力前提、Codex review 2026-05-09 P2 fix 反映):
   B1. 新 entry なし (count 増えてない) → 0.0
   B2. 新 entry が既存と完全同一 embedding → 0.0
       (識別力: cosine 比較しない実装で常に 1.0 → fail)
   B3. 新 entry が既存と直交 → ~= 1.0
       (識別力: 旧 oldest first 実装 (embs[prev_count:] で末尾取得) で
       新と既存を取り違えて 0 → fail、Codex 指摘 P2 #1 回帰防止)
   B4. 初 cycle (existing 0、prev_count 0) → 0.0

C. _effective_change_gain (Codex review 2026-05-09 P2 #2 fix 反映):
   C1. last_e1 上昇 → 加点 (= diff)
   C2. last_e1 不変 → 0.0
   C3. last_e1 下降 → 0.0
       (識別力: clamp 忘れ実装で負値返す → fail)
   C4. last_e1 0.0 保持 (legitimate 0.0 を default 0.5 で潰さない)
       (識別力: ``or 0.5`` 旧実装で 0.5-0.5=0 返す → fail、本 fixture は
       state=0.5/prev=0.0 で旧 0、新 0.5 を区別)

D. _memory_link_gain:
   D1. strength 合計上昇 → diff 加点
   D2. strength 合計不変 → 0.0
   D3. strength 合計下降 (decay) → 0.0
       (識別力: clamp + 忘れ実装で負値 → fail)

E. _world_model_resolution_gain (BP-1 hotfix 2026-05-09 で /100 線形正規化追加):
   E1. pe 下降 + density 上昇 → 両方加点 (合計、fixture 0-100 実値域)
   E2. pe 不変 + density 上昇 → density のみ加点 (fixture 0-100 実値域)
       (識別力: density を読まない実装で 0 → fail)
   E3. pe None (初 cycle) → density のみで OK、pe 部分は 0 fallback
   E4. fog_now=None → 0 fallback (defensive、fixture 0-100 実値域)
   E5. 大 pe drop (prev=80, now=10) → /100 後 0.7 で saturate (cycle 19 spike bug 同型再現)
       (識別力: PE_NORMALIZATION_FACTOR 落とした旧無正規化実装で 70.0 → fail)
   E6. 値域上限 (prev=100, now=0) → 1.0、上限突破しない
       (識別力: 正規化なし実装で 100.0 → fail)

F. _capability_gain:
   F1. variance 縮小 + tool 多様性 +1 → 両方加点
   F2. tool 多様性のみ +1 → diversity 1.0 加点
       (識別力: variance のみ実装で 0 → fail)
   F3. predictor_confidence 空 → 0.0

G. _redundancy_penalty (newest first 入力前提):
   G1. 異なる tool 5 連発 + 多様 emb → 0
   G2. 同 tool 5 連発 + 多様 emb → 4 (5-1) penalty
   G3. 直近 emb 完全一致 (中心固着) → centroid_stuck > 0
       (識別力: 閾値 0 固定実装で常に 0 → fail)
   G4. action_ledger 空 → 0
   G5. newest first で先頭 (=直近) 5 件を見る (Codex review P2 #3 回帰防止)
       (識別力: 旧 `[-N:]` 末尾実装で古い stuck cluster を見て fail)

H. snapshot_for_next_cycle:
   H1. 必須 field 全て揃う (last_e1 / pe / entries_count / strength_total /
       predictor_var / tool_count / fog_density)
   H2. predictor_confidence 空でも snapshot 生成 (defensive、var=0/count=0)
   H3. legitimate 0.0 last_e1 が snapshot に保持される (Codex P2 #2 回帰防止)

I. round-trip integration:
   I1. cycle_N で snapshot → cycle_N+1 で compute、subjective_count diff が
       novelty_gain の input になる (既存→新と同期して認識される)

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_info_gain.py
"""
import sys
from pathlib import Path

import pytest

# Slice 6.5 Step 3a (PLAN §5.12.2) 2026-05-11: info_gain.py の helper 数式を Lv3 (nat
# 単位 log ratio) に書換に伴い、旧 raw / clamp 値域期待の test 群 8 件を Step 5 大規模置換
# まで一時 skip。skip 対象は @pytest.mark.skip で marker、Step 5 で完全置換予定。
_SLICE65_STEP5_DEFERRED = pytest.mark.skip(
    reason="Slice 6.5 Step 5 (PLAN §5.12.2) で大規模置換予定、"
    "Step 3a 以降 helper Lv3 log scale 移行で旧 raw/clamp 数値期待は意味失う"
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.info_gain as cig


def _assert(cond, label):
    """assert helper。pytest 互換のため失敗時は AssertionError を raise する。

    Codex review 2026-05-09 P2 #4 fix: `def test_*(): return False` だと
    pytest が PytestReturnNotNoneWarning を出すだけで test pass 扱いになる。
    AssertionError raise で正しく fail させる (standalone runner は run_all
    内で try/except で集計)。
    """
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        raise AssertionError(label)
    return cond


def _emb(*vals):
    """test 用 embedding (短縮、L2 正規化済を仮定)。"""
    import math
    norm = math.sqrt(sum(v * v for v in vals))
    if norm < 1e-9:
        return list(vals)
    return [v / norm for v in vals]


# ============================================================
# A. compute_info_gain_components 統合
# ============================================================

def test_a1_all_keys_present():
    """A1: 全 6 項目 + info_gain + model_resolution_gain が dict に含まれる。"""
    print("== A1: all keys present ==")
    # fixture 0-100 値域 (BP-1 hotfix 2026-05-09 識別力強化)
    state = {"last_e1": 0.5, "last_prediction_error": 30.0,
             "predictor_confidence": {}, "action_ledger": []}
    prev = {}
    result = cig.compute_info_gain_components(state, prev, [], [])
    expected_keys = {
        "info_gain", "model_resolution_gain",
        "novelty_gain", "effective_change_gain", "memory_link_gain",
        "world_model_resolution_gain", "capability_gain", "redundancy_penalty",
    }
    return _assert(set(result.keys()) == expected_keys,
                   f"all 8 keys: {set(result.keys())}")


def test_a2_info_gain_literal_formula():
    """A2: info_gain = sum 5 positive - redundancy_penalty (literal 加算式)。"""
    print("== A2: info_gain literal formula ==")
    state = {
        "last_e1": 0.7,
        "last_prediction_error": 20.0,  # 実 state 0-100 値域 (BP-1 hotfix 2026-05-09)
        "predictor_confidence": {
            "tool_a": {"success": 5, "fail": 0},
            "tool_b": {"success": 3, "fail": 2},
        },
        "action_ledger": [{"tool": "tool_a"}],  # 単発、penalty 0
    }
    prev = {
        "last_e1": 0.5,  # +0.2 effective_change
        "last_prediction_error": 40.0,  # pe_drop = (40-20)/100 = 0.2
        "subjective_count": 0,
        "memory_links_strength_total": 0.0,
        "predictor_success_rate_var": 0.5,
        "predictor_tool_count": 0,
        "fog_local_density_mean": None,
    }
    result = cig.compute_info_gain_components(state, prev, [], [])
    expected = (
        result["novelty_gain"]
        + result["effective_change_gain"]
        + result["memory_link_gain"]
        + result["world_model_resolution_gain"]
        + result["capability_gain"]
        - result["redundancy_penalty"]
    )
    return _assert(
        abs(result["info_gain"] - round(expected, 6)) < 1e-5,
        f"literal sum: {result['info_gain']} ~= {expected}",
    )


def test_a3_model_resolution_equals_wm_res():
    """A3: model_resolution_gain == world_model_resolution_gain (独立 metric の同値性)。"""
    print("== A3: model_resolution_gain alias ==")
    state = {"last_e1": 0.5, "last_prediction_error": 10.0,
             "predictor_confidence": {}, "action_ledger": []}
    prev = {"last_prediction_error": 50.0,  # pe_drop = (50-10)/100 = 0.4
            "fog_local_density_mean": 0.3}
    fog = {"local_density_mean": 0.5}  # density_gain = 0.2
    result = cig.compute_info_gain_components(state, prev, [], [], fog_now=fog)
    return _assert(
        result["model_resolution_gain"] == result["world_model_resolution_gain"],
        f"alias equal: {result['model_resolution_gain']} == {result['world_model_resolution_gain']}",
    )


# ============================================================
# B. _novelty_gain (CIG N の embedding 近似)
# ============================================================

def test_b1_no_new_entries():
    """B1: 新 entry なし (entries_count 増えてない) → 0.0。"""
    print("== B1: no new entries ==")
    entries = [{"embedding": _emb(1, 0, 0)}, {"embedding": _emb(0, 1, 0)}]
    prev = {"entries_count": 2}
    result = cig._novelty_gain(entries, prev)
    return _assert(result == 0.0, f"novelty=0 (no new): {result}")


def test_b2_new_identical_to_existing():
    """B2: 新 entry が既存と完全同一 embedding → 0.0 (max sim 1.0)。

    入力順序: newest first (新 entry が list 先頭、既存が後ろ)。
    識別力: cosine 比較しない実装は常に高 novelty 返す → fail。
    """
    print("== B2: new identical -> novelty 0 ==")
    entries = [
        {"embedding": _emb(1, 0, 0)},  # 新 (newest first 先頭)、既存と同一
        {"embedding": _emb(1, 0, 0)},  # 既存
    ]
    prev = {"entries_count": 1}
    result = cig._novelty_gain(entries, prev)
    return _assert(result < 1e-5, f"novelty ~= 0: {result}")


def test_b3_new_orthogonal_to_existing():
    """B3: 新 entry が既存と直交 → ~= 1.0 (max sim 0)。

    入力順序: newest first (新 entry 先頭)。
    識別力 (Codex review P2 #1 回帰防止): 旧 oldest first 実装
    (`embs[prev_count:]` で末尾取得) なら新と既存を取り違えて
    novelty = 0 を返す → fail。本 fixture は newest first 仮定の
    実装でのみ ~= 1.0 を返す。
    """
    print("== B3: new orthogonal -> novelty ~= 1 ==")
    entries = [
        {"embedding": _emb(0, 1, 0)},  # 新 (newest first 先頭)、直交
        {"embedding": _emb(1, 0, 0)},  # 既存
    ]
    prev = {"entries_count": 1}
    result = cig._novelty_gain(entries, prev)
    return _assert(result > 0.99, f"novelty ~= 1: {result}")


def test_b4_initial_cycle_no_existing():
    """B4: 初 cycle (existing 0、prev_count 0) → 0.0 (比較不可)。"""
    print("== B4: initial cycle -> 0 ==")
    entries = [{"embedding": _emb(1, 0, 0)}]
    prev = {}  # entries_count 不在 → default 0、existing なし
    result = cig._novelty_gain(entries, prev)
    return _assert(result == 0.0, f"initial cycle: {result}")


# ============================================================
# C. _effective_change_gain (pragmatic value)
# ============================================================

@_SLICE65_STEP5_DEFERRED
def test_c1_e1_increase():
    """C1: last_e1 上昇 → 加点 (= diff)。"""
    print("== C1: e1 increase → gain ==")
    state = {"last_e1": 0.7}
    prev = {"last_e1": 0.5}
    result = cig._effective_change_gain(state, prev)
    return _assert(abs(result - 0.2) < 1e-5, f"diff 0.2: {result}")


@_SLICE65_STEP5_DEFERRED
def test_c2_e1_unchanged():
    """C2: last_e1 不変 → 0.0。"""
    print("== C2: e1 unchanged → 0 ==")
    state = {"last_e1": 0.5}
    prev = {"last_e1": 0.5}
    result = cig._effective_change_gain(state, prev)
    return _assert(result == 0.0, f"unchanged: {result}")


@_SLICE65_STEP5_DEFERRED
def test_c3_e1_decrease_clamped():
    """C3: last_e1 下降 → 0.0 (clamp +)。

    識別力: clamp 忘れ実装で負値返す → fail。
    """
    print("== C3: e1 decrease -> 0 (clamp) ==")
    state = {"last_e1": 0.3}
    prev = {"last_e1": 0.7}
    result = cig._effective_change_gain(state, prev)
    return _assert(result == 0.0, f"clamp +: {result}")


@_SLICE65_STEP5_DEFERRED
def test_c4_e1_zero_preserved():
    """C4: last_e1 = 0.0 を default 0.5 で潰さない (Codex review P2 #2 回帰防止)。

    fixture: state.last_e1 = 0.5, prev.last_e1 = 0.0
    期待: max(0, 0.5 - 0.0) = 0.5
    旧実装 (`or 0.5`): 0.5 or 0.5 = 0.5、0.0 or 0.5 = 0.5、diff = 0 → fail
    新実装: legitimate 0.0 を保持して 0.5 を返す
    """
    print("== C4: e1 zero preserved ==")
    state = {"last_e1": 0.5}
    prev = {"last_e1": 0.0}
    result = cig._effective_change_gain(state, prev)
    return _assert(abs(result - 0.5) < 1e-5, f"0.0 preserved -> 0.5 gain: {result}")


# ============================================================
# D. _memory_link_gain (A-MEM)
# ============================================================

@_SLICE65_STEP5_DEFERRED
def test_d1_strength_total_increase():
    """D1: strength 合計上昇 → diff 加点。"""
    print("== D1: strength sum increase ==")
    links = [{"strength": 0.5}, {"strength": 0.7}]  # sum = 1.2
    prev = {"memory_links_strength_total": 0.8}
    result = cig._memory_link_gain(links, prev)
    return _assert(abs(result - 0.4) < 1e-5, f"diff 0.4: {result}")


def test_d2_strength_total_unchanged():
    """D2: strength 合計不変 → 0.0。"""
    print("== D2: strength sum unchanged ==")
    links = [{"strength": 0.5}]
    prev = {"memory_links_strength_total": 0.5}
    result = cig._memory_link_gain(links, prev)
    return _assert(result == 0.0, f"unchanged: {result}")


@_SLICE65_STEP5_DEFERRED
def test_d3_strength_total_decrease_clamped():
    """D3: strength 合計下降 (decay) → 0.0 (clamp +)。

    識別力: clamp 忘れ実装で負値 → fail。
    """
    print("== D3: strength sum decay → 0 ==")
    links = [{"strength": 0.3}]
    prev = {"memory_links_strength_total": 0.8}
    result = cig._memory_link_gain(links, prev)
    return _assert(result == 0.0, f"clamp +: {result}")


# ============================================================
# E. _world_model_resolution_gain (Bayesian posterior precision + 情報理論的 entropy)
#
# fixture 値域は **実 state 0-100 スケール** (BP-1 hotfix 2026-05-09):
#   旧 fixture は 0-1 値域で書かれてて、実 state 0-100 と乖離 → cycle 19 spike
#   bug を test 上で検出できなかった (Codex 5 周 review 見逃し真因)。
#   識別力強化のため全 fixture を 0-100 実値域に揃える (CLAUDE.md §5 literal)。
# ============================================================

@_SLICE65_STEP5_DEFERRED
def test_e1_both_pe_drop_and_density_gain():
    """E1: pe 下降 + density 上昇 → 両方加点 (合計)。

    fixture 0-100 値域: pe 20 vs 50 → pe_drop_raw=30 → /100 = 0.3
    """
    print("== E1: pe drop + density gain ==")
    state = {"last_prediction_error": 20.0}
    prev = {"last_prediction_error": 50.0, "fog_local_density_mean": 0.3}
    fog = {"local_density_mean": 0.5}
    result = cig._world_model_resolution_gain(state, prev, fog)
    expected = 0.3 + 0.2  # pe_drop (30/100) + density_gain (0.5-0.3)
    return _assert(abs(result - expected) < 1e-5, f"sum {expected}: {result}")


@_SLICE65_STEP5_DEFERRED
def test_e2_density_only_pe_unchanged():
    """E2: pe 不変 + density 上昇 → density のみ加点。

    fixture 0-100 値域: pe 40 vs 40 → pe_drop_raw=0 → /100 = 0
    識別力: density を読まない実装で 0 → fail。
    """
    print("== E2: density-only gain ==")
    state = {"last_prediction_error": 40.0}
    prev = {"last_prediction_error": 40.0, "fog_local_density_mean": 0.3}
    fog = {"local_density_mean": 0.6}
    result = cig._world_model_resolution_gain(state, prev, fog)
    return _assert(abs(result - 0.3) < 1e-5, f"density only: {result}")


@_SLICE65_STEP5_DEFERRED
def test_e3_pe_none_initial_cycle():
    """E3: pe None (初 cycle) → density のみで OK。"""
    print("== E3: pe None → density only ==")
    state = {"last_prediction_error": None}
    prev = {"fog_local_density_mean": 0.2}
    fog = {"local_density_mean": 0.5}
    result = cig._world_model_resolution_gain(state, prev, fog)
    return _assert(abs(result - 0.3) < 1e-5, f"pe None defensive: {result}")


@_SLICE65_STEP5_DEFERRED
def test_e4_fog_none_defensive():
    """E4: fog_now None → density 部分 0 fallback。

    fixture 0-100 値域: pe 10 vs 30 → pe_drop_raw=20 → /100 = 0.2
    """
    print("== E4: fog None defensive ==")
    state = {"last_prediction_error": 10.0}
    prev = {"last_prediction_error": 30.0}
    result = cig._world_model_resolution_gain(state, prev, None)
    return _assert(abs(result - 0.2) < 1e-5, f"pe drop only: {result}")


@_SLICE65_STEP5_DEFERRED
def test_e5_large_pe_drop_saturates_to_unit_range():
    """E5: 大 pe drop で値域 [0, ~1] に saturate (BP-1 hotfix 2026-05-09 の核 test)。

    cycle 19 spike bug の同型再現: prev=80, now=10 → pe_drop_raw=70。
    旧無正規化実装は 70.0 を返す = info_gain 全体を支配する spike 発生。
    新実装は /100 後 0.7 = [0, 1] 範囲内で saturate、6 成分加算式の単位整合維持。

    識別力: PE_NORMALIZATION_FACTOR を落とす実装 (生差分のまま) で 70.0 → fail。
    BP-1 cycle 19 で観察された world_model_resolution_gain=34.0003 の同型 case。
    """
    print("== E5: large pe drop saturates ==")
    state = {"last_prediction_error": 10.0}
    prev = {"last_prediction_error": 80.0}
    result = cig._world_model_resolution_gain(state, prev, None)
    return _assert(
        0.0 <= result <= 1.0 and abs(result - 0.7) < 1e-5,
        f"saturate to [0,1]: {result} (expected 0.7)",
    )


@_SLICE65_STEP5_DEFERRED
def test_e6_max_pe_drop_at_value_range_boundary():
    """E6: 値域上限 (pe 100 → 0) で結果が 1.0、上限突破しない。

    識別力: 正規化なし実装で 100.0 → fail。
    """
    print("== E6: max pe drop boundary ==")
    state = {"last_prediction_error": 0.0}
    prev = {"last_prediction_error": 100.0}
    result = cig._world_model_resolution_gain(state, prev, None)
    return _assert(
        abs(result - 1.0) < 1e-5,
        f"upper bound 1.0: {result}",
    )


# ============================================================
# F. _capability_gain (CIG C)
# ============================================================

@_SLICE65_STEP5_DEFERRED
def test_f1_variance_drop_and_diversity():
    """F1: variance 縮小 + tool 多様性 +1 → 両方加点。"""
    print("== F1: variance + diversity ==")
    state = {
        "predictor_confidence": {
            "a": {"success": 5, "fail": 0},  # rate=1.0
            "b": {"success": 4, "fail": 1},  # rate=0.8
            "c": {"success": 3, "fail": 2},  # rate=0.6 (新規 tool)
        },
    }
    # rates = [1.0, 0.8, 0.6], var = ~0.04
    prev = {"predictor_success_rate_var": 0.5, "predictor_tool_count": 2}
    result = cig._capability_gain(state, prev)
    # competence_gain = 0.5 - 0.04 ~= 0.46
    # tool_diversity_delta = 3 - 2 = 1
    return _assert(result > 1.4 and result < 1.5, f"variance + diversity: {result}")


@_SLICE65_STEP5_DEFERRED
def test_f2_diversity_only():
    """F2: tool 多様性のみ +1 → diversity 1.0 加点。

    識別力: variance のみ実装で 0 → fail。
    """
    print("== F2: diversity only ==")
    state = {"predictor_confidence": {"a": {"success": 1, "fail": 0}}}
    # 1 tool で variance 計算不可 → var_now = 0
    prev = {"predictor_success_rate_var": 0.0, "predictor_tool_count": 0}
    result = cig._capability_gain(state, prev)
    return _assert(result == 1.0, f"diversity 1: {result}")


def test_f3_empty_predictor():
    """F3: predictor_confidence 空 → 0.0。"""
    print("== F3: empty predictor ==")
    state = {"predictor_confidence": {}}
    prev = {}
    result = cig._capability_gain(state, prev)
    return _assert(result == 0.0, f"empty: {result}")


# ============================================================
# G. _redundancy_penalty (CIG L)
# ============================================================

def test_g1_diverse_tools_diverse_embs():
    """G1: 異なる tool 5 連発 + 多様 emb → 0。"""
    print("== G1: diverse → 0 ==")
    state = {"action_ledger": [
        {"tool": "a"}, {"tool": "b"}, {"tool": "c"},
        {"tool": "d"}, {"tool": "e"},
    ]}
    entries = [
        {"embedding": _emb(1, 0, 0)}, {"embedding": _emb(0, 1, 0)},
        {"embedding": _emb(0, 0, 1)},
    ]
    result = cig._redundancy_penalty(state, entries)
    return _assert(result == 0.0, f"diverse: {result}")


def test_g2_same_tool_5_consecutive():
    """G2: 同 tool 5 連発 + 観察データ不足 (entries 2 件) → 4 (5-1) penalty のみ。

    centroid_stuck は entries 5 件未満で 0 (実コード側で gate)、起動直後 cycle
    で偶然境界値判定が出る罠の回帰防止 fixture。consecutive_penalty 純粋検証。
    """
    print("== G2: same tool 5x + entries < 5 -> 4 penalty ==")
    state = {"action_ledger": [{"tool": "x"}] * 5}
    entries = [
        {"embedding": _emb(1, 0, 0)},
        {"embedding": _emb(0, 1, 0)},
    ]
    result = cig._redundancy_penalty(state, entries)
    return _assert(abs(result - 4.0) < 1e-5, f"5-1 penalty: {result}")


def test_g3_centroid_stuck():
    """G3: 直近 emb 完全一致 → centroid_stuck > 0。

    識別力: 閾値 0 固定実装で常に 0 → fail。
    """
    print("== G3: centroid stuck ==")
    state = {"action_ledger": []}
    entries = [{"embedding": _emb(1, 0, 0)}] * 5  # 5 件全部同一
    result = cig._redundancy_penalty(state, entries)
    return _assert(result > 0.9, f"stuck > 0.9: {result}")


def test_g4_empty_ledger():
    """G4: action_ledger 空 → 0 (consecutive_run 0)。"""
    print("== G4: empty ledger -> 0 ==")
    state = {"action_ledger": []}
    entries = []
    result = cig._redundancy_penalty(state, entries)
    return _assert(result == 0.0, f"empty: {result}")


def test_g5_newest_first_centroid_window():
    """G5: newest first 順序で先頭 (=直近) 5 件を見る (Codex review P2 #3 回帰防止)。

    fixture: [新 5 件 (多様、直交ベクトル群)] + [古 5 件 (全部同一、stuck)] (newest first)
    新実装 (`[:RECENT_EMB_WINDOW]`): 先頭 5 = 新 (多様) → stuck = 0
    旧実装 (`[-RECENT_EMB_WINDOW:]`): 末尾 5 = 古 (全部同一) → stuck > 0 → fail

    識別力: redundancy 計算が「直近の繰り返し」を見るのか「過去の cluster」を
    見るのかを区別する。realtime な行動固着検知のためには直近の方を見る必要。
    """
    print("== G5: newest first centroid window (recent diverse) ==")
    new_diverse = [
        {"embedding": _emb(1, 0, 0)},
        {"embedding": _emb(0, 1, 0)},
        {"embedding": _emb(0, 0, 1)},
        {"embedding": _emb(1, 1, 0)},
        {"embedding": _emb(1, 0, 1)},
    ]
    old_stuck = [{"embedding": _emb(0.5, 0.5, 0.5)}] * 5
    entries = new_diverse + old_stuck  # newest first
    state = {"action_ledger": []}
    result = cig._redundancy_penalty(state, entries)
    # 新実装: 直近 5 件 (多様) → stuck = 0、ledger 空なので consecutive 0
    # 旧実装: 末尾 5 件 (古 stuck) → centroid_stuck > 0
    return _assert(result == 0.0, f"recent diverse window: {result}")


# ============================================================
# H. snapshot_for_next_cycle
# ============================================================

@_SLICE65_STEP5_DEFERRED
def test_h1_snapshot_all_fields():
    """H1: 必須 field 全て揃う。"""
    print("== H1: snapshot all fields ==")
    state = {
        "last_e1": 0.6,
        "last_prediction_error": 25.0,  # 実 state 0-100 値域
        "predictor_confidence": {
            "a": {"success": 5, "fail": 0},
            "b": {"success": 3, "fail": 2},
        },
    }
    entries = [{"embedding": _emb(1, 0)}, {"embedding": _emb(0, 1)}]
    links = [{"strength": 0.5}, {"strength": 0.3}]
    fog = {"local_density_mean": 0.4}
    snap = cig.snapshot_for_next_cycle(state, entries, links, fog)
    expected_keys = {
        "last_e1", "last_prediction_error", "entries_count",
        "memory_links_strength_total", "predictor_success_rate_var",
        "predictor_tool_count", "fog_local_density_mean",
    }
    ok = set(snap.keys()) == expected_keys
    ok &= snap["last_e1"] == 0.6
    ok &= snap["entries_count"] == 2
    ok &= abs(snap["memory_links_strength_total"] - 0.8) < 1e-5
    ok &= snap["predictor_tool_count"] == 2
    ok &= snap["fog_local_density_mean"] == 0.4
    return _assert(ok, f"snapshot: {snap}")


def test_h2_snapshot_defensive_empty():
    """H2: predictor_confidence 空でも snapshot 生成 (defensive)。"""
    print("== H2: defensive empty ==")
    state = {"last_e1": 0.5, "last_prediction_error": None,
             "predictor_confidence": {}}
    snap = cig.snapshot_for_next_cycle(state, [], [], None)
    ok = (
        snap["predictor_success_rate_var"] == 0.0
        and snap["predictor_tool_count"] == 0
        and snap["last_prediction_error"] is None
        and snap["fog_local_density_mean"] is None
        and snap["memory_links_strength_total"] == 0.0
    )
    return _assert(ok, f"defensive: {snap}")


def test_h3_snapshot_preserves_zero_e1():
    """H3: state["last_e1"] = 0.0 を snapshot で保持 (Codex P2 #2 回帰防止)。

    識別力: ``or 0.5`` 旧実装で 0.0 が 0.5 に化ける → fail。
    """
    print("== H3: snapshot preserves 0.0 e1 ==")
    state = {"last_e1": 0.0, "last_prediction_error": 0.0,
             "predictor_confidence": {}}
    snap = cig.snapshot_for_next_cycle(state, [], [], None)
    ok = snap["last_e1"] == 0.0 and snap["last_prediction_error"] == 0.0
    return _assert(ok, f"0.0 preserved: e1={snap['last_e1']}, pe={snap['last_prediction_error']}")


# ============================================================
# I. round-trip integration
# ============================================================

def test_i1_round_trip():
    """I1: cycle_N で snapshot → cycle_N+1 で compute、新 entry が novelty 計算に入る。

    入力順序: newest first (新 entry を list 先頭に挿入する load_all_* の順序契約)。
    """
    print("== I1: round-trip cycle N -> N+1 ==")
    # Cycle N: 1 entry のみ
    state_n = {
        "last_e1": 0.5,
        "last_prediction_error": 40.0,  # 実 state 0-100 値域
        "predictor_confidence": {},
        "action_ledger": [],
    }
    entries_n = [{"embedding": _emb(1, 0, 0)}]
    snap_n = cig.snapshot_for_next_cycle(state_n, entries_n, [], None)

    # Cycle N+1: 新 entry 追加 (直交)、newest first なので先頭に prepend
    state_np1 = {
        "last_e1": 0.5,
        "last_prediction_error": 40.0,  # pe 不変 (cycle N と同値、pe_drop=0)
        "predictor_confidence": {},
        "action_ledger": [],
    }
    entries_np1 = [{"embedding": _emb(0, 1, 0)}] + entries_n  # newest first
    result = cig.compute_info_gain_components(state_np1, snap_n, entries_np1, [])
    # novelty_gain ~= 1.0 (新 entry が直交)
    return _assert(
        result["novelty_gain"] > 0.99,
        f"round-trip novelty ~= 1: {result['novelty_gain']}",
    )


# ============================================================
# J. metrics.py 経由統合 (実コード機能確認)
# ============================================================

def test_j1_metrics_event_includes_info_gain():
    """J1: build_cycle_metrics_event の戻り値 dict に info_gain field が含まれる。

    識別力: build に info_gain 配線してない実装 (Slice 2 のままの metrics.py) で fail。
    実コードが metrics_events.jsonl に Slice 3 の info_gain を実際に書く保証。
    """
    print("== J1: metrics event includes info_gain ==")
    import core.metrics as cm
    state = {
        "cycle_id": 1,
        "run_id": "test-run",
        "session_id": "test-sess",
        "last_e1": 0.6,
        "last_e2": 0.5,
        "last_e3": 0.5,
        "last_e4": 0.5,
        "last_prediction_error": 20.0,  # 実 state 0-100 値域
        "energy": 50.0,
        "entropy": 0.65,
        "pressure": 0.0,
        "predictor_confidence": {},
        "action_ledger": [],
        "_info_gain_prev": {},
        "voluntary_memory_store_count": 0,
        "tool_level": 0,
        "files_written": [],
        "files_read": [],
    }
    settings = {"model": "test"}
    pref = {}
    event = cm.build_cycle_metrics_event(
        state, settings, pref,
        entries_with_embedding=[],
        links=[],
        code_version="test",
    )
    ok = "info_gain" in event
    if ok:
        ig = event["info_gain"]
        ok &= isinstance(ig, dict)
        ok &= "info_gain" in ig and "model_resolution_gain" in ig
        ok &= all(k in ig for k in (
            "novelty_gain", "effective_change_gain", "memory_link_gain",
            "world_model_resolution_gain", "capability_gain", "redundancy_penalty",
        ))
    return _assert(ok, f"info_gain field present: keys={list(event.get('info_gain', {}).keys())}")


def test_j2_emit_updates_prev_snapshot():
    """J2: emit_cycle_metrics 後に state[CYCLE_KEY] が新 snapshot で更新される。

    識別力: snapshot 更新を落とした実装 (build のみで終わる) で fail。
    cycle 跨ぎで diff 計算が回る保証 (起動再起動で state.json 経由で persist)。
    """
    print("== J2: emit updates prev snapshot ==")
    import core.metrics as cm
    import core.info_gain as cig_mod
    import shutil
    import tempfile
    from pathlib import Path

    # tmp MEMORY_DIR を立てて emit 実行 (jsonl 書き込み副作用を隔離)
    tmpdir = tempfile.mkdtemp(prefix="info_gain_j2_")
    try:
        from core import config as cfg
        original_mem_dir = cfg.MEMORY_DIR
        cfg.MEMORY_DIR = Path(tmpdir)

        state = {
            "cycle_id": 1,
            "run_id": "test-run",
            "session_id": "test-sess",
            "last_e1": 0.5,
            "last_prediction_error": 30.0,  # 実 state 0-100 値域
            "predictor_confidence": {},
            "action_ledger": [],
            "_info_gain_prev": {},  # 初期 empty
        }
        settings = {"model": "test"}
        pref = {}

        # 取得関数を mock (実 jsonl を読まないため)
        import core.memory as cm_mem
        import core.memory_links as cm_links
        original_subj = cm_mem.load_all_subjective_entries
        original_mem = cm_mem.load_all_memories
        original_links = cm_links.list_links
        cm_mem.load_all_subjective_entries = lambda: [{"embedding": _emb(1, 0, 0)}]
        cm_mem.load_all_memories = lambda: []
        cm_links.list_links = lambda limit=500: [{"strength": 0.5}]

        try:
            cm.emit_cycle_metrics(state, settings, pref)
        finally:
            cm_mem.load_all_subjective_entries = original_subj
            cm_mem.load_all_memories = original_mem
            cm_links.list_links = original_links
            cfg.MEMORY_DIR = original_mem_dir

        # state[CYCLE_KEY] が更新された (空でない)
        snap = state.get(cig_mod.CYCLE_KEY, {})
        ok = isinstance(snap, dict) and len(snap) > 0
        ok &= snap.get("entries_count") == 1  # mock fixture 1 件
        ok &= abs(snap.get("memory_links_strength_total", 0) - 0.5) < 1e-5
        return _assert(ok, f"snapshot updated: {snap}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# Runner
# ============================================================

def run_all():
    tests = [
        test_a1_all_keys_present,
        test_a2_info_gain_literal_formula,
        test_a3_model_resolution_equals_wm_res,
        test_b1_no_new_entries,
        test_b2_new_identical_to_existing,
        test_b3_new_orthogonal_to_existing,
        test_b4_initial_cycle_no_existing,
        test_c1_e1_increase,
        test_c2_e1_unchanged,
        test_c3_e1_decrease_clamped,
        test_d1_strength_total_increase,
        test_d2_strength_total_unchanged,
        test_d3_strength_total_decrease_clamped,
        test_e1_both_pe_drop_and_density_gain,
        test_e2_density_only_pe_unchanged,
        test_e3_pe_none_initial_cycle,
        test_e4_fog_none_defensive,
        test_e5_large_pe_drop_saturates_to_unit_range,
        test_e6_max_pe_drop_at_value_range_boundary,
        test_f1_variance_drop_and_diversity,
        test_f2_diversity_only,
        test_f3_empty_predictor,
        test_g1_diverse_tools_diverse_embs,
        test_g2_same_tool_5_consecutive,
        test_g3_centroid_stuck,
        test_g4_empty_ledger,
        test_g5_newest_first_centroid_window,
        test_c4_e1_zero_preserved,
        test_h1_snapshot_all_fields,
        test_h2_snapshot_defensive_empty,
        test_h3_snapshot_preserves_zero_e1,
        test_i1_round_trip,
        test_j1_metrics_event_includes_info_gain,
        test_j2_emit_updates_prev_snapshot,
    ]
    # _assert は AssertionError raise する設計 (pytest 互換)。standalone runner
    # では try/except で集計、pytest 経由では各 test 関数が直接 fail する。
    results = []
    for t in tests:
        try:
            t()
            results.append(True)
        except AssertionError as e:
            results.append(False)
    passed = sum(1 for r in results if r)
    print(f"\n{'='*50}")
    print(f"Result: {passed}/{len(results)} pass")
    print('='*50)
    return passed == len(results)


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
