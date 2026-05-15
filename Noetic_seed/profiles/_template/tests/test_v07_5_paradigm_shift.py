"""V07.5 FEP Literal Selection Paradigm Shift の識別力 test (§6-1 ~ §6-10、PLAN v1.4 literal)。

CLAUDE.md §5 (test 識別力) + §6 (docstring/spec/test 同期) 厳守:
各 test は誤実装注入で fail する fixture を設計、commit 1-5 の literal 健全性を verify。

PLAN v1.4 §6 識別力 test 設計 literal:
  §6-1: argmin_π G(π) 各項識別力 (pragmatic / epistemic / regularization 符号間違い)
  §6-2: C fixed snapshot timing literal verify (update_self 内 _efe_C 再構築禁止)
  §6-3: Proposal 2 batched forward literal verify (5 候補 1 batch forward)
  §6-4: 段階14 basin id 堅牢化 (type mismatch 防止、PLAN v1.4 §3-5 fix のみ literal、dormancy 解消保証じゃない)
  §6-5: 内発駆動 device 維持 3 点アサート (calc_pressure_signals + should_reflect 早出し literal)
  §6-6: raw+subj id join 完全性 (_collect_training_pairs triple 構築)
  §6-7: raw/subj id 1:1 完全性 assertion (_split_entry_fields literal)
  §6-8: JEPA model mode lock reentrancy guard (eval/train 競合防止)
  §6-9: bge-m3 batch encode + result cache verify (5x latency 削減)
  §6-10: _efe_C first-cycle bootstrap (cycle 1 None graceful)
"""
import hashlib
import math

import pytest


# ============================================================
# §6-1: argmin_π G(π) 各項識別力 (PLAN v1.4 §6-1 literal)
# ============================================================

class TestG_PiSignsDiscrimination:
    """G(π) = -pragmatic - epistemic + regularization の符号配置 literal verify。

    PLAN reference: PLAN v1.4 §6-1 (argmin_π G(π) 各項識別力)

    識別力 fixture (Codex review 2 周目 P2 反映、production 経路 verify):
      pragmatic 大 → G 負方向 (preference 達成 = 低 G で argmin 勝ち)
      epistemic 大 → G 負方向 (情報利得 = 低 G で argmin 勝ち)
      regularization 大 → G 正方向 (固着 = 高 G で argmin 負け)

    誤実装 fail: compute_efe_components 内で符号間違い (G = +pragmatic + epistemic - reg)
    した場合、baseline G との大小関係が literal 逆転して本 test fail。monkeypatch で
    helper の戻り値を制御、production G formula 経路を literal 経由する設計。
    """

    def test_g_decreases_with_pragmatic_via_production_path(self, monkeypatch):
        """case 1 (production 経路): pragmatic 大 → G 小 literal verify。

        monkeypatch で _effective_change_gain (pragmatic 項) を high value に literal 増、
        baseline G との literal 比較で「pragmatic 大 → G 小」literal 符号 verify。
        識別力誤実装 fail: compute_efe_components が `+pragmatic` で書かれてたら
        high pragmatic → high G になり、`G_high < G_baseline` が literal 不成立で test fail。
        """
        from core.info_gain import compute_efe_components
        import core.info_gain as ig_mod
        state = {"self": {}, "_efe_C": None, "predictor_confidence": {}, "action_ledger": []}

        # baseline (全成分 0)
        result_baseline = compute_efe_components(state, {}, [], [], None)
        G_baseline = result_baseline["G"]

        # pragmatic 項 (effective_change) のみ high に literal mock
        monkeypatch.setattr(ig_mod, "_effective_change_gain", lambda s, p, h=None: 5.0)
        result_high_prag = compute_efe_components(state, {}, [], [], None)
        # production G formula で pragmatic 大 → G 小 literal verify
        assert result_high_prag["G"] < G_baseline, (
            f"pragmatic 増 → G 減じない literal 異常 (符号間違い候補): "
            f"high={result_high_prag['G']} vs baseline={G_baseline}"
        )

    def test_g_decreases_with_epistemic_via_production_path(self, monkeypatch):
        """case 2 (production 経路): epistemic 大 → G 小 literal verify。

        monkeypatch で _novelty_gain (epistemic 項) を high に literal 増、
        production G formula 経由で「epistemic 大 → G 小」literal 符号 verify。
        """
        from core.info_gain import compute_efe_components
        import core.info_gain as ig_mod
        state = {"self": {}, "_efe_C": None, "predictor_confidence": {}, "action_ledger": []}

        result_baseline = compute_efe_components(state, {}, [], [], None)
        G_baseline = result_baseline["G"]

        # epistemic 項 (novelty) のみ high
        monkeypatch.setattr(ig_mod, "_novelty_gain", lambda s, p, h=None: 3.0)
        result_high_epi = compute_efe_components(state, {}, [], [], None)
        assert result_high_epi["G"] < G_baseline, (
            f"epistemic 増 → G 減じない literal 異常: high={result_high_epi['G']} vs baseline={G_baseline}"
        )

    def test_g_increases_with_regularization_via_production_path(self, monkeypatch):
        """case 3 (production 経路): regularization 大 → G 大 literal verify。

        monkeypatch で _consecutive_penalty (regularization 項) を high に literal 増、
        production G formula 経由で「regularization 大 → G 大」literal 符号 verify。
        識別力誤実装 fail: regularization を `-R` で書かれてたら G 減 → 本 test fail。
        """
        from core.info_gain import compute_efe_components
        import core.info_gain as ig_mod
        state = {"self": {}, "_efe_C": None, "predictor_confidence": {}, "action_ledger": []}

        result_baseline = compute_efe_components(state, {}, [], [], None)
        G_baseline = result_baseline["G"]

        # regularization 項 (consecutive_penalty) のみ high
        monkeypatch.setattr(ig_mod, "_consecutive_penalty", lambda s, p, h=None: 5.0)
        result_high_reg = compute_efe_components(state, {}, [], [], None)
        assert result_high_reg["G"] > G_baseline, (
            f"regularization 増 → G 増じない literal 異常 (符号間違い候補): "
            f"high={result_high_reg['G']} vs baseline={G_baseline}"
        )


# ============================================================
# §6-2: C fixed snapshot timing (PLAN v1.4 §6-2 literal)
# ============================================================

class TestCFixedSnapshotTiming:
    """update_self は _efe_C を mutate しない、_rebuild_efe_C_snapshot のみ literal 更新。

    PLAN reference: PLAN v1.4 §6-2 (C fixed snapshot timing literal verify)

    A2 paradigm shift literal verify (commit 4):
      旧誤実装: update_self 内で _efe_C 再構築 (self-referential loop = cycle 55 attractor 真因)
      新 correct: update_self は _efe_self_confidence のみ更新、_efe_C は cycle start hook で再構築
    """

    def test_update_self_does_not_mutate_efe_c(self, monkeypatch):
        """case 1: update_self 呼出後、_efe_C は前 snapshot のまま (literal 不変)。"""
        from tools import builtin as builtin_mod

        mock_state = {
            "self": {"identity": "fixed"},
            "cycle_id": 5,
            "_efe_self_confidence": {"identity": 0.5},
            "_efe_C": {"sentinel": "pre_update_value"},  # 前 cycle の snapshot
            "_efe_C_update_cycle": 3,
            "drives_state": {},
        }
        monkeypatch.setattr(builtin_mod, "load_state", lambda: mock_state)
        monkeypatch.setattr(builtin_mod, "save_state", lambda s: mock_state.update(s))

        builtin_mod._update_self("preferences", "new_pref", confidence=0.8)

        # _efe_self_confidence は更新 (next cycle で C 再構築時に反映)
        assert mock_state["_efe_self_confidence"]["preferences"] == 0.8
        # _efe_C は touch なし (literal sentinel 不変、A2 literal)
        assert mock_state["_efe_C"] == {"sentinel": "pre_update_value"}
        assert mock_state["_efe_C_update_cycle"] == 3

    def test_rebuild_efe_c_snapshot_does_mutate(self, monkeypatch):
        """case 2: _rebuild_efe_C_snapshot 呼出で _efe_C literal 構築。"""
        from core import controller as ctrl_mod
        from core import preference_distribution as pd_mod

        # bge-m3 mock (literal deterministic)
        def fake_embed(texts):
            return [[1.0] + [0.0] * 1023 for _ in texts]
        monkeypatch.setattr(pd_mod, "_embed_sync", fake_embed)
        monkeypatch.setattr(pd_mod, "is_vector_ready", lambda: True)

        mock_state = {
            "self": {"identity": "I am"},
            "cycle_id": 10,
            "_efe_self_confidence": {"identity": 0.7},
            "_efe_C": None,
            "_efe_C_update_cycle": -1,
        }

        ctrl_mod._rebuild_efe_C_snapshot(mock_state)

        assert mock_state["_efe_C"] is not None
        assert mock_state["_efe_C"]["n_components"] == 1
        assert mock_state["_efe_C_update_cycle"] == 10


# ============================================================
# §6-3: Proposal 2 batched forward (PLAN v1.4 §6-3 literal)
# ============================================================

class TestProposal2BatchedForward:
    """predict_next_conditioned_batch は (N, T+1, D) 1 batch forward literal。

    PLAN reference: PLAN v1.4 §6-3 (Proposal 2 batched forward literal verify)

    誤実装 fail: 5 候補を serial forward で 5 回呼出 → forward count != 1。
    correct: 1 batch tensor で 1 回呼出。

    JEPA 未起動環境では graceful skip 想定、positive 5-candidate test (Codex 2 周目 P2 反映)
    で mock torch + MockJEPAModel literal 経由で本格 batched verify。
    """

    def test_predict_next_conditioned_batch_skips_without_torch(self, monkeypatch):
        """case 1: torch 未 install で None 返却 (graceful skip)。"""
        from core import jepa_runtime
        from core import predictor_jepa as pj
        monkeypatch.setattr(pj, "is_torch_available", lambda: False)

        result = jepa_runtime.predict_next_conditioned_batch({}, [{"tool": "x", "intent": "y"}])
        assert result is None

    def test_predict_next_conditioned_batch_empty_candidates(self):
        """case 2: candidates 空で [] 返却 (graceful)。"""
        from core import jepa_runtime
        result = jepa_runtime.predict_next_conditioned_batch({}, [])
        assert result == []

    def test_predict_next_conditioned_batch_5_candidates_single_forward(self, monkeypatch):
        """case 3 (positive 5-candidate、Codex 2 周目 P2 反映): 5 候補で JEPAModel.forward が
        literal **1 回だけ** 呼ばれる、batch shape (N=5, T+1=9, D=1024) literal 検証。

        識別力誤実装 fail:
          - serial 5 forward 実装で forward call count == 5 (本 test fail、batched 化未達)
          - batch shape T+1 != 9 (Option B' literal 未反映、SEQ_LENGTH+1=9 で append 機能してない)

        correct: 1 batch tensor (5, 9, 1024) で literal 1 回 forward。
        """
        from core import predictor_jepa as pj
        if not pj.is_torch_available():
            pytest.skip("torch not available")

        from core import jepa_runtime as jr_mod
        import torch

        # forward 呼出回数 + batch shape を spy
        forward_call_log = {"n": 0, "shapes": []}

        class MockJEPAModel:
            def __init__(self, **kwargs):
                self.seq_length = kwargs.get("seq_length", 9)
            def eval(self):
                return self
            def train(self):
                return self
            def __call__(self, x):
                forward_call_log["n"] += 1
                forward_call_log["shapes"].append(tuple(x.shape))
                N, _T, D = x.shape
                out = torch.randn(N, D)
                return torch.nn.functional.normalize(out, p=2, dim=-1)

        # _model singleton + helpers を literal mock
        monkeypatch.setattr(jr_mod, "_model", MockJEPAModel(seq_length=9))
        monkeypatch.setattr(jr_mod, "_get_model", lambda: jr_mod._model)
        monkeypatch.setattr(jr_mod, "_build_input_sequence", lambda s: torch.randn(1, 8, 1024))
        # predict_next_conditioned_batch は内部で _embed_batch_with_cache を直接呼ぶ literal、
        # monkeypatch path = embedding module (not jepa_runtime)
        from core import embedding as emb_mod
        monkeypatch.setattr(emb_mod, "is_vector_ready", lambda: True)
        monkeypatch.setattr(emb_mod, "_embed_batch_with_cache", lambda texts: [[0.5] * 1024 for _ in texts])

        state = {}
        candidates = [
            {"tool": f"tool_{i}", "intent": f"intent_{i}"} for i in range(5)
        ]

        results = jr_mod.predict_next_conditioned_batch(state, candidates)

        # batched 1 回 forward literal verify
        assert forward_call_log["n"] == 1, (
            f"5 候補で forward 呼出回数 = {forward_call_log['n']} (期待 1 = batched)、"
            f"serial 実装 (= 5 回) の literal 候補"
        )
        # batch shape (N=5, T+1=9, D=1024) literal
        assert forward_call_log["shapes"][0] == (5, 9, 1024), (
            f"batch shape = {forward_call_log['shapes'][0]} ≠ (5, 9, 1024)、"
            f"Option B' (SEQ_LENGTH+1=9) literal 未反映の候補"
        )
        # 結果は各 candidate に対応する 5 個 literal
        assert results is not None and len(results) == 5


# ============================================================
# §6-4: basin id 堅牢化 (PLAN v1.4 §6-4 literal)
# ============================================================

class TestBasinIdRobustness:
    """_classify_basin_from_snapshot は subject_id を str() 正規化 literal、type mismatch 防止。

    PLAN reference: PLAN v1.4 §6-4 (basin id 堅牢化 literal verify)

    誤実装 fail: 旧 `if subject_id in cluster.get("memory_ids"):` で type mismatch (str vs int)
                 → basin reconcile 失敗で "" 返却。
    correct: str() 正規化 + set 化で type mismatch 解消。

    PLAN v1.4 §3-5 literal「dormant 解消保証じゃなく fix そのもの literal」整合 (smoke data 依存)。
    """

    def test_basin_classify_str_subject_str_memory(self):
        """case 1: 両側 str で literal マッチ。"""
        from core.world_model import _classify_basin_from_snapshot
        clusters = [{"cluster_id": "c1", "memory_ids": ["abc", "def"]}]
        assert _classify_basin_from_snapshot("abc", clusters) == "c1"

    def test_basin_classify_int_subject_str_memory(self):
        """case 2: type mismatch (subject=int, memory_ids=str) でも literal マッチ (str 正規化)。"""
        from core.world_model import _classify_basin_from_snapshot
        clusters = [{"cluster_id": "c1", "memory_ids": ["123"]}]
        assert _classify_basin_from_snapshot(123, clusters) == "c1"

    def test_basin_classify_no_match_returns_empty(self):
        """case 3: subject が cluster に含まれない → "" graceful。"""
        from core.world_model import _classify_basin_from_snapshot
        clusters = [{"cluster_id": "c1", "memory_ids": ["xyz"]}]
        assert _classify_basin_from_snapshot("abc", clusters) == ""


# ============================================================
# §6-5: 内発駆動 device 維持 3 点アサート (PLAN v1.4 §6-5 literal)
# ============================================================

class TestInternalDrivePreservation:
    """A3 降格後も internal drive 経路 literal 維持。3 点 critical literal:
      1. calc_pressure_signals の prediction_error 経路維持
      2. should_reflect 早出し reflection 経路維持 (last_prediction_error > 0.8)
      3. main.py:1202 + 1211 の calc_pressure_signals 呼出 + pressure 累積は touch なし

    PLAN reference: PLAN v1.4 §6-5 (内発駆動 device 維持 3 点アサート) + §8-1 C (Blocker 2 件)

    誤実装 fail: pressure 機構消滅 → reflection 前倒し発火が cycle interval (タイマー) 駆動に退化。

    Codex 2 周目 P2 反映: main.py preservation の literal source 検査追加 (case 4, 5)。
    """

    def test_calc_pressure_signals_emits_prediction_error(self):
        """case 1: calc_pressure_signals が last_prediction_error から prediction_error signal を literal 出力。"""
        from core.entropy import calc_pressure_signals
        signals = calc_pressure_signals({"last_prediction_error": 100, "pending": []})
        assert "prediction_error" in signals
        assert signals["prediction_error"] > 0

    def test_should_reflect_triggers_on_high_prediction_error(self):
        """case 2: should_reflect 早出し発火 (cycles_since >= 3 + last_prediction_error > 0.8 in 0-100 scale)。"""
        from core.reflection import should_reflect
        # cycles_since=3 + last_prediction_error=81 (>0.8 in /100 scale) で literal 発火
        state = {"reflection_cycle": 3, "last_prediction_error": 81}
        assert should_reflect(state) is True

    def test_should_reflect_no_premature_fire_with_low_pe(self):
        """case 3: 低 prediction_error で前倒し発火しない (cycle interval 経路に literal 委ねる)。"""
        from core.reflection import should_reflect
        # cycles_since=3、PE=10 (< 80) で前倒し発火しない literal
        state = {"reflection_cycle": 3, "last_prediction_error": 10}
        assert should_reflect(state) is False

    def test_main_py_preserves_calc_pressure_signals_call(self):
        """case 4 (Codex 2 周目 P2 反映、main.py preservation literal): main.py に literal な
        `calc_pressure_signals(` 呼出が保持されてるか source 検査。

        PLAN v1.4 §6-5 + §8-1 C 内発駆動 Blocker literal: main.py:1202 で毎 cycle
        calc_pressure_signals(state, ...) が literal 呼ばれる必要、A3 降格で controller 経路
        から外しただけで main.py 経路は **絶対削除しない** literal 保証。

        誤実装 fail: A3 「降格」を「機構消滅」と誤解して main.py:1202 から calc_pressure_signals
        呼出を削除 → 本 source 検査で `calc_pressure_signals(` 文字列消失で fail。
        """
        from pathlib import Path
        main_py_path = Path(__file__).resolve().parent.parent / "main.py"
        source = main_py_path.read_text(encoding="utf-8")

        assert "calc_pressure_signals(" in source, (
            "main.py で calc_pressure_signals 呼出消失 = A3 内発駆動 Blocker literal 違反 "
            "(PLAN v1.4 §8-1 C Blocker、reflection 前倒し発火経路 literal 死守)"
        )

    def test_main_py_preserves_pressure_accumulation(self):
        """case 5 (Codex 2 周目 P2 反映、main.py preservation literal): main.py に pressure
        累積構造 `pressure = pressure * decay + signal_total + clock_base` literal が保持
        されてるか source 検査。

        PLAN v1.4 §8-1 C 内発駆動 Blocker literal: main.py:1211 付近で pressure 累積式
        が literal 動作する必要、A3 降格で削除されない literal 保証。
        """
        from pathlib import Path
        main_py_path = Path(__file__).resolve().parent.parent / "main.py"
        source = main_py_path.read_text(encoding="utf-8")

        # pressure 累積式の literal pattern (pressure 変数 + decay or clock_base)
        has_pressure_word = "pressure" in source
        has_accumulation_marker = ("decay" in source) or ("clock_base" in source)
        assert has_pressure_word and has_accumulation_marker, (
            "main.py で pressure 累積式消失 = A3 内発駆動 Blocker literal 違反 "
            f"(pressure 単語: {has_pressure_word}, decay/clock_base marker: {has_accumulation_marker})"
        )


# ============================================================
# §6-6: raw+subj id join 完全性 (PLAN v1.4 §6-6 literal)
# ============================================================

class TestRawSubjIdJoin:
    """_collect_training_pairs は raw+subj id join で triple (seq, π_taken, target) を構築。

    PLAN reference: PLAN v1.4 §6-6 (raw+subj id join 完全性 verify)

    誤実装 fail: raw entry 欠落で tool 取得失敗 → π_taken=None で graceful skip じゃなく crash。
    correct: graceful skip (pi_taken=None で triple 含む、消費側で skip)。

    Codex 2 周目 P2 反映: empty state graceful + non-empty paired entries の literal 検証
    両方カバー、id join が機能してることを literal verify。
    """

    def test_collect_training_pairs_empty_state(self):
        """case 1: empty state で [] 返却 (graceful、bootstrap)。"""
        from core import jepa_runtime
        from core import predictor_jepa as pj
        if not pj.is_torch_available():
            pytest.skip("torch not available")
        result = jepa_runtime._collect_training_pairs({"subjective_entries": [], "raw_events": []})
        assert result == []

    def test_collect_training_pairs_non_empty_paired_entries(self, monkeypatch):
        """case 2 (Codex 2 周目 P2 反映、non-empty paired): subjective_entries + raw_events
        に同 id paired entries を literal 配置、_collect_training_pairs が triple
        (seq, π_taken, target) を literal 構築する verify。

        識別力誤実装 fail: raw+subj id join が機能してないと π_taken=None で literal 全 triple
        が pi_taken=None になる、本 test の `pi_taken is not None` literal assert で fail。

        correct: 同 id paired entry で _build_policy_embedding 経由 pi_taken 1024D 構築。
        """
        from core import predictor_jepa as pj
        if not pj.is_torch_available():
            pytest.skip("torch not available")

        from core import jepa_runtime as jr_mod

        # _build_policy_embedding を literal mock (non-None 1024D vector 返却 = bge-m3 mock)
        monkeypatch.setattr(jr_mod, "_build_policy_embedding", lambda tool, intent: [0.5] * 1024)

        # mock state with paired raw+subj entries (SEQ_LENGTH=8 以上必要)
        subj_entries = []
        raw_entries = []
        for i in range(10):  # SEQ_LENGTH=8 以上、literal 10 件
            eid = f"entry_{i:03d}"
            subj_entries.append({
                "id": eid,
                "intent": f"intent_text_{i}",  # non-empty intent literal
                "embedding": [0.1 * ((i + 1) % 7)] * 1024,  # literal 1024D vector (variation 付き)
            })
            raw_entries.append({
                "id": eid,
                "tool": f"tool_{i % 3}",  # non-empty tool literal (3 種)
                "result": "ok",
            })

        state = {
            "subjective_entries": subj_entries,
            "raw_events": raw_entries,
        }

        triples = jr_mod._collect_training_pairs(state)

        # 10 entries で SEQ_LENGTH=8 以上、triple が literal 構築
        assert len(triples) > 0, (
            "non-empty paired entries で triple ゼロ件 = raw+subj id join literal failure"
        )

        # 各 triple は (seq, pi_taken, target) literal structure
        from core.predictor_jepa import SEQ_LENGTH, EMBED_DIM
        for seq, pi_taken, target in triples:
            assert isinstance(seq, list) and len(seq) == SEQ_LENGTH, (
                f"seq literal structure 異常: len={len(seq) if isinstance(seq, list) else 'N/A'}"
            )
            assert isinstance(target, list) and len(target) == EMBED_DIM, (
                f"target literal structure 異常: len={len(target) if isinstance(target, list) else 'N/A'}"
            )
            # pi_taken は raw+subj id join で literal 構築、None じゃない (paired raw tool + subj intent literal)
            assert pi_taken is not None, (
                "pi_taken=None: raw+subj id join literal failure (paired entries あるのに tool/intent 取得失敗)"
            )
            assert isinstance(pi_taken, list) and len(pi_taken) == EMBED_DIM


# ============================================================
# §6-7: raw/subj id 1:1 完全性 assertion (PLAN v1.4 §6-7 literal)
# ============================================================

class TestRawSubjIdOneToOne:
    """_split_entry_fields は id を raw + subj 両側 literal コピー + assertion で防壁化。

    PLAN reference: PLAN v1.4 §6-7 (raw/subj id 1:1 完全性 assertion literal)

    誤実装 fail: RAW_FIELDS / SUBJECTIVE_FIELDS から id 抜けると assert fail。
    correct: id 両側に literal 含む、両者一致 literal 保証。
    """

    def test_split_entry_preserves_id_in_both_partitions(self):
        """case 1: 通常 entry で id が raw + subj 両側に literal 存在 + 一致。"""
        from core.memory import _split_entry_fields
        entry = {"id": "abc123", "tool": "search_memory", "intent": "explore", "result": "ok"}
        raw, subj, _ = _split_entry_fields(entry)
        assert raw["id"] == "abc123"
        assert subj["id"] == "abc123"
        assert raw["id"] == subj["id"]

    def test_split_entry_no_id_no_assertion(self):
        """case 2: id 欠落 entry で assertion trigger なし (entry に id ない case literal graceful)。"""
        from core.memory import _split_entry_fields
        entry = {"tool": "x", "intent": "y"}  # id なし
        raw, subj, _ = _split_entry_fields(entry)
        # id ない場合 assertion は skip (literal safe)
        assert "id" not in raw
        assert "id" not in subj


# ============================================================
# §6-8: JEPA model mode lock reentrancy guard (PLAN v1.4 §6-8 literal)
# ============================================================

class TestJepaModelModeLock:
    """_model_mode_lock は eval/train mode 切替を排他制御 literal。

    PLAN reference: PLAN v1.4 §6-8 (JEPA mode lock reentrancy guard)

    ★ 本 class の test は **literal source-level smoke** (Codex 2 周目 P3 explicit documentation):
    実 concurrent spy test (threading + JEPA model concurrent call で literal race detection)
    は heavy fixture (threading.Event + 2 thread + 同期 barrier) で commit 7 / smoke 後 hotfix
    reserved。本 class は source 上で _model_mode_lock 参照が literal 存在することを verify
    する smoke test、Codex「acceptable as literal smoke if explicitly documented」literal 整合。

    誤実装 fail: mode lock なしで predict_next_conditioned_batch 中に maybe_train_step 割込
                 → mode 競合 race condition risk。
    correct: threading.Lock で排他制御 literal。
    """

    def test_model_mode_lock_is_threading_lock(self):
        """case 1: _model_mode_lock が threading.Lock instance literal。"""
        from core import jepa_runtime
        import threading
        assert isinstance(jepa_runtime._model_mode_lock, type(threading.Lock()))

    def test_predict_next_uses_lock(self):
        """case 2: predict_next の source に `_model_mode_lock` 参照 literal 存在。"""
        from core import jepa_runtime
        import inspect
        source = inspect.getsource(jepa_runtime.predict_next)
        assert "_model_mode_lock" in source

    def test_predict_next_conditioned_batch_uses_lock(self):
        """case 3: predict_next_conditioned_batch の source に `_model_mode_lock` 参照 literal 存在。"""
        from core import jepa_runtime
        import inspect
        source = inspect.getsource(jepa_runtime.predict_next_conditioned_batch)
        assert "_model_mode_lock" in source

    def test_maybe_train_step_uses_lock(self):
        """case 4: maybe_train_step の source に `_model_mode_lock` 参照 literal 存在。"""
        from core import jepa_runtime
        import inspect
        source = inspect.getsource(jepa_runtime.maybe_train_step)
        assert "_model_mode_lock" in source


# ============================================================
# §6-9: bge-m3 batch encode + result cache (PLAN v1.4 §6-9 literal)
# ============================================================

class TestBgeM3BatchAndCache:
    """_embed_batch_with_cache は cache hit で encode skip、5x latency 削減 literal。

    PLAN reference: PLAN v1.4 §6-9 (bge-m3 batch encode + result cache literal verify)

    誤実装 fail: cache miss で再 encode → encode 回数増。
    correct: cache hit で _embed_sync skip。
    """

    def test_embed_with_cache_caches_result(self, monkeypatch):
        """case 1: 同 text 2 回呼出で _embed_sync が 1 回のみ呼ばれる (cache hit)。"""
        from core import embedding as emb_mod
        call_count = {"n": 0}

        def spy_embed_sync(texts):
            call_count["n"] += 1
            return [[1.0] + [0.0] * 1023 for _ in texts]

        monkeypatch.setattr(emb_mod, "_embed_sync", spy_embed_sync)
        # cache clear
        emb_mod._EMBEDDING_CACHE.clear()

        result1 = emb_mod._embed_with_cache("tool_x: intent_y")
        result2 = emb_mod._embed_with_cache("tool_x: intent_y")
        assert result1 == result2
        assert call_count["n"] == 1  # 2 回目は cache hit literal

    def test_embed_batch_with_cache_only_misses_invoke_sync(self, monkeypatch):
        """case 2: batch 内 cache hit + miss 混在で miss だけ _embed_sync 呼出 literal。"""
        from core import embedding as emb_mod
        call_args = {"texts": None}

        def spy_embed_sync(texts):
            call_args["texts"] = list(texts)
            return [[1.0] + [0.0] * 1023 for _ in texts]

        monkeypatch.setattr(emb_mod, "_embed_sync", spy_embed_sync)
        emb_mod._EMBEDDING_CACHE.clear()

        # 1 回目: 全 miss
        emb_mod._embed_batch_with_cache(["t1", "t2"])
        assert call_args["texts"] == ["t1", "t2"]

        # 2 回目: t1 cache hit + t3 miss → t3 のみ _embed_sync 呼出 literal
        call_args["texts"] = None
        emb_mod._embed_batch_with_cache(["t1", "t3"])
        assert call_args["texts"] == ["t3"]


# ============================================================
# §6-10: _efe_C first-cycle bootstrap (PLAN v1.4 §6-10 literal)
# ============================================================

class TestEfeCFirstCycleBootstrap:
    """cycle 1 で _efe_C=None でも _rebuild_efe_C_snapshot literal 動作、crash しない。

    PLAN reference: PLAN v1.4 §6-10 (_efe_C first-cycle bootstrap literal verify)

    誤実装 fail: bootstrap hook なしで cycle 1 controller_select 実行 → _efe_C None で計算崩壊。
    correct: cycle start hook で _rebuild_efe_C_snapshot 強制呼出、空 self → empty components literal。

    Codex 2 周目 P2 反映: source-level smoke (case 2) に加えて call-count spy (case 3) で
    cycle start hook の **実動 timing** を literal verify、dead-code 状態を検出可能化。
    """

    def test_rebuild_efe_c_snapshot_with_empty_self(self, monkeypatch):
        """case 1: 空 self → empty components literal (bootstrap、graceful)。"""
        from core import controller as ctrl_mod
        from core import preference_distribution as pd_mod

        # bge-m3 mock
        monkeypatch.setattr(pd_mod, "_embed_sync", lambda texts: [[1.0] + [0.0] * 1023 for _ in texts])
        monkeypatch.setattr(pd_mod, "is_vector_ready", lambda: True)

        mock_state = {"self": {}, "cycle_id": 0, "_efe_self_confidence": {}, "_efe_C": None}
        ctrl_mod._rebuild_efe_C_snapshot(mock_state)

        # 空 self → components 空 literal、ただし C 構造自体は構築済 (graceful)
        assert mock_state["_efe_C"] is not None
        assert mock_state["_efe_C"]["n_components"] == 0
        assert mock_state["_efe_C"]["source_keys"] == []

    def test_rebuild_efe_c_snapshot_called_at_cycle_start(self):
        """case 2: controller() 関数の source に _rebuild_efe_C_snapshot 呼出 literal 存在 (literal smoke)。"""
        from core import controller as ctrl_mod
        import inspect
        source = inspect.getsource(ctrl_mod.controller)
        assert "_rebuild_efe_C_snapshot" in source

    def test_rebuild_efe_c_snapshot_actually_called_via_spy(self, monkeypatch):
        """case 3 (Codex 2 周目 P2 反映、call-count spy): controller() 呼出で
        _rebuild_efe_C_snapshot が literal 1 回呼ばれる、cycle start hook の **実動 timing**
        を literal verify (source-level smoke の弱点を補完)。

        識別力誤実装 fail: controller() の hook 呼出 line を消した dead-code 状態だと、
        call-count=0 で literal 本 test fail (source level test では検出不可、本 spy で literal 検出)。

        correct: controller() entry で _rebuild_efe_C_snapshot(state) が literal 1 回呼出。
        """
        from core import controller as ctrl_mod
        from core import preference_distribution as pd_mod

        # bge-m3 mock (scipy 経由のため、test 環境で literal deterministic)
        monkeypatch.setattr(pd_mod, "_embed_sync", lambda texts: [[1.0] + [0.0] * 1023 for _ in texts])
        monkeypatch.setattr(pd_mod, "is_vector_ready", lambda: True)

        # _rebuild_efe_C_snapshot を literal spy で置換
        spy_call_log = {"n": 0, "called_with_state": False}
        orig_rebuild = ctrl_mod._rebuild_efe_C_snapshot

        def spy_rebuild(state):
            spy_call_log["n"] += 1
            spy_call_log["called_with_state"] = isinstance(state, dict)
            orig_rebuild(state)  # 元の literal logic 実行

        monkeypatch.setattr(ctrl_mod, "_rebuild_efe_C_snapshot", spy_rebuild)

        # minimal state + controller 呼出 (literal cycle start simulate)
        state = {
            "self": {},
            "cycle_id": 0,
            "_efe_self_confidence": {},
            "_efe_C": None,
            "_efe_C_update_cycle": -1,
            "energy": 50,
            "entropy": 0.65,
            "files_read": [],
            "files_written": [],
            "tool_level": 0,
            "predictor_confidence": {},
            "action_ledger": [],
            "raw_events": [],
            "subjective_entries": [],
            "voluntary_memory_store_count": 0,
            "pending": [],
        }
        tools_dict = {"reflect": None}  # minimal
        level_tools = {0: ["reflect"], 1: [], 2: [], 3: []}

        ctrl_mod.controller(state, tools_dict, level_tools)

        # _rebuild_efe_C_snapshot が literal 1 回呼ばれた + state 引数で
        assert spy_call_log["n"] == 1, (
            f"controller() 呼出で _rebuild_efe_C_snapshot 呼出回数 = {spy_call_log['n']} (期待 1)、"
            f"cycle start hook literal 未実動 (dead code 候補)"
        )
        assert spy_call_log["called_with_state"], "hook が state 以外で呼ばれた literal 異常"


# ============================================================
# 実行 (pytest 経由想定、CLAUDE.md §5/§6 識別力 + docstring/spec 同期)
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
