"""段階13 Phase 2 (Layer C JEPA 流次状態予測) — 検証 test。

検証対象 (commit 508478e):
  - core.predictor_jepa.JEPAModel: causal transformer 4 layer / 8 head /
    1024 dim / mlp_ratio=2 / seq=8、出力 L2 normalize 強制
  - core.predictor_jepa.hybrid_loss: 0.8·(1-cos) + 0.2·MSE 数値正確性
  - core.predictor_jepa.cosine_distance: 0.0-1.0 clamp
  - core.predictor_jepa.is_torch_available: optional import 判定
  - core.jepa_runtime._build_input_sequence: SEQ_LENGTH 検査 + Phase 1 helper
    non_empty_subjective_entries 流用 + 形状ズレ entry 除外
  - core.jepa_runtime.measure_prediction_error: graceful skip (None / 形状ズレ)
  - core.jepa_runtime._append_pred_error_history: cap PRED_ERROR_HISTORY_CAP FIFO
  - core.jepa_runtime.maybe_train_step: counter trigger + reset
  - core.jepa_runtime.jepa_observe_entry: graceful skip + state 操作順序契約

CLAUDE.md §5 (識別力) + §6 (docstring 同期) literal 適用。

想定誤実装 12 件カバー:
  D1. JEPAModel が学習しない (predict 出力が学習前後で同じ、optimizer.step 抜け)
  D2. graceful skip 効かない (torch import 失敗時に NameError)
  D3. cosine_distance が 0.0-1.0 クランプ外す
  D4. JEPAModel forward が L2 normalize 忘れ (出力 norm != 1.0)
  D5. _build_input_sequence が SEQ_LENGTH 未満で None 返さない
  D6. _build_input_sequence が non_empty_subjective_entries 経由してない (blank 混入)
  D7. _build_input_sequence が形状ズレ entry 除外しない
  D8. measure_prediction_error が形状ズレで None 返さない
  D9. _append_pred_error_history が cap 超え (古いの削除し忘れ)
  D10. maybe_train_step counter が trigger 時 reset しない
  D11. jepa_observe_entry が embedding 欠落 entry で crash (state 汚染)
  D12. hybrid_loss の重み係数取り違え (0.8/0.2 逆等)

識別力 fixture 4 種混在 (CLAUDE.md §5):
  - subj entry あり embedding あり 1024D (full)
  - subj entry あり embedding 欠落 (Phase 0.1.A 二層化で blank subj が日常)
  - subj entry あり embedding 形状ズレ (512D 等)
  - subj entry 不在 / raw event のみ

torch 不在時運用: 各 test 冒頭で is_torch_available() で SKIP 判定。本実装の
graceful skip 設計 (Phase 0/1 影響ゼロ保証) を test runner level でも維持。
torch 在時は 本格 identifying。Phase 7 結合 smoke で .venv に torch install 後
本格識別力検証 (Phase 1 commit 2 と同型: bge-m3 mock pattern 流儀継承)。

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_predictor_jepa.py
"""
import sys
from pathlib import Path

# standalone 直接実行時 (Windows cp932 console) でも Unicode 文字 (≈, →, etc.)
# を出力できるよう UTF-8 reconfigure (Noetic_seed/run_tests.py 冒頭と同 pattern)。
# run_tests.py 経由 (subprocess capture_output=encoding="utf-8") では不要だが、
# standalone debug 経路でも動かせるよう defensive。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, OSError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import predictor_jepa as _pj
from core import jepa_runtime as _jr


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _make_subj_entries(n: int) -> list:
    """有効 subj entry n 件を一意 embedding で返す (一意性で形状検証可能)。"""
    out = []
    for i in range(n):
        emb = [0.0] * _pj.EMBED_DIM
        emb[i % _pj.EMBED_DIM] = 1.0
        out.append({
            "id": f"sub_make_{i}",
            "intent": f"intent {i}",
            "expect": f"expect {i}",
            "embedding": emb,
        })
    return out


# ============================================================
# predictor_jepa.py (model / loss / cosine_distance / is_torch_available)
# ============================================================

def test_is_torch_available_matches_module_flag():
    """D2 予防: is_torch_available() == _pj._TORCH_AVAILABLE 一致。

    api と internal flag がズレると graceful skip 経路の judging が壊れる。
    """
    print("== is_torch_available() == _TORCH_AVAILABLE 一致 ==")
    return _assert(
        _pj.is_torch_available() == _pj._TORCH_AVAILABLE,
        f"flag 一致 (api={_pj.is_torch_available()}, internal={_pj._TORCH_AVAILABLE})",
    )


def test_hybrid_loss_numeric_correctness():
    """D12 予防: hybrid_loss = 0.8·(1-cos) + 0.2·MSE 数値検証。

    完全一致 case で loss ≈ 0、直交 case で loss ≈ 0.8 + 微少 MSE 寄与。
    重み係数取り違え (0.2/0.8 逆) でこの両 case の値が崩れる識別力。
    """
    if not _pj.is_torch_available():
        return _assert(True, "torch 不在 skip (Phase 7 smoke で本格検証)")
    import torch
    print("== hybrid_loss: 完全一致 ≈0 / 直交 ≈0.8 (重み検証) ==")
    a = torch.zeros(1, _pj.EMBED_DIM)
    a[0, 0] = 1.0
    same = a.clone()
    orth = torch.zeros(1, _pj.EMBED_DIM)
    orth[0, 1] = 1.0
    same_loss = float(_pj.hybrid_loss(a, same))
    orth_loss = float(_pj.hybrid_loss(a, orth))
    ok = True
    ok &= _assert(same_loss < 1e-5, f"完全一致 loss ≈ 0 (actual: {same_loss:.6f})")
    # 直交: cos=0 → 0.8·1.0 = 0.8、MSE = (1²+1²)/EMBED_DIM/B ≈ 2/1024 ≈ 0.002
    # → 0.8 + 0.2·0.002 ≈ 0.8004。識別力: 0.7 < orth_loss < 0.85 で 0.8 重み確認
    ok &= _assert(0.7 < orth_loss < 0.85,
                  f"直交 loss ≈ 0.8 (重み 0.8/0.2 確認、actual: {orth_loss:.6f})")
    return ok


def test_cosine_distance_clamp():
    """D3 予防: cosine_distance を 0.0-1.0 にクランプ。

    完全一致 → 0.0、直交 → 1.0、反対 (cos=-1 → 1.0 clamp) の 3 識別力 case。
    clamp 忘れ誤実装は反対 case で 2.0 を返す。
    """
    if not _pj.is_torch_available():
        return _assert(True, "torch 不在 skip")
    import torch
    print("== cosine_distance: 完全一致 0 / 直交 1 / 反対 clamp 1 ==")
    a = torch.zeros(_pj.EMBED_DIM); a[0] = 1.0
    same = a.clone()
    orth = torch.zeros(_pj.EMBED_DIM); orth[1] = 1.0
    opposite = torch.zeros(_pj.EMBED_DIM); opposite[0] = -1.0
    ok = True
    ok &= _assert(_pj.cosine_distance(a, same) < 1e-6, "完全一致 → 0")
    ok &= _assert(abs(_pj.cosine_distance(a, orth) - 1.0) < 1e-6, "直交 → 1")
    ok &= _assert(_pj.cosine_distance(a, opposite) == 1.0,
                  f"反対 → clamp 1.0 (actual: {_pj.cosine_distance(a, opposite)})")
    return ok


def test_jepa_model_forward_shape_and_l2_norm():
    """D4 予防: JEPAModel forward 出力形状 (B, EMBED_DIM) + L2 norm ≈ 1.0。

    L2 normalize 忘れ誤実装は norm != 1.0 で識別。bge-m3 と同空間整合のため
    出力は必ず L2 normalized でないといけない。
    """
    if not _pj.is_torch_available():
        return _assert(True, "torch 不在 skip")
    import torch
    print("== JEPAModel forward: shape (B, D) + L2 norm ≈ 1.0 ==")
    model = _pj.JEPAModel()
    seq = torch.randn(2, _pj.SEQ_LENGTH, _pj.EMBED_DIM)
    with torch.inference_mode():
        out = model(seq)
    norms = out.norm(p=2, dim=-1)
    ok = True
    ok &= _assert(out.shape == (2, _pj.EMBED_DIM),
                  f"形状 (2, {_pj.EMBED_DIM}) (actual: {tuple(out.shape)})")
    ok &= _assert(bool(((norms - 1.0).abs() < 1e-5).all()),
                  f"L2 norm ≈ 1.0 (actual: {norms.tolist()})")
    return ok


def test_jepa_model_learns_in_one_step():
    """D1 予防: 1 step 学習で predict 出力が変化。

    random seq に対し target を decide → hybrid_loss → backward → step
    → 同 seq の predict 出力が学習前と異なる。
    識別力: optimizer.step() 抜き誤実装で no-change 検出。
    """
    if not _pj.is_torch_available():
        return _assert(True, "torch 不在 skip")
    import torch
    print("== JEPAModel: 1 step 学習で predict 出力変化 ==")
    torch.manual_seed(42)
    model = _pj.JEPAModel()
    seq = torch.randn(2, _pj.SEQ_LENGTH, _pj.EMBED_DIM)
    target = torch.zeros(2, _pj.EMBED_DIM); target[:, 0] = 1.0

    # Codex review P2-1 反映: TransformerEncoderLayer は default で dropout 付き、
    # torch.inference_mode() は grad 切るだけで dropout は止めない。model.eval()
    # を before/after 両方で呼ばないと、異なる dropout mask の noise だけで diff
    # > 1e-4 になり、optimizer.step 抜き誤実装でも pass する偽 green 状態になる
    # (D1 識別力 ZERO リスク)。eval mode 強制で dropout を止めて純粋な学習効果
    # のみで diff 検証する。
    model.eval()
    with torch.inference_mode():
        before = model(seq).clone()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pred = model(seq)
    loss = _pj.hybrid_loss(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        after = model(seq)

    diff = float((before - after).abs().max())
    return _assert(diff > 1e-4, f"学習で predict 変化 (max diff: {diff:.6f})")


# ============================================================
# jepa_runtime.py (sequence build / measure / history / train trigger / observer)
# ============================================================

def test_build_input_sequence_seq_length_check():
    """D5 予防: SEQ_LENGTH 未満で None、ちょうどで tensor 返す。

    識別力: 不足チェック忘れ誤実装は (1, 7, 1024) 等の不正形状 tensor を返す。
    """
    if not _pj.is_torch_available():
        return _assert(True, "torch 不在 skip")
    print("== _build_input_sequence: SEQ_LENGTH-1 → None / SEQ_LENGTH → tensor ==")
    state_short = {"subjective_entries": _make_subj_entries(_pj.SEQ_LENGTH - 1)}
    state_exact = {"subjective_entries": _make_subj_entries(_pj.SEQ_LENGTH)}
    out_short = _jr._build_input_sequence(state_short)
    out_exact = _jr._build_input_sequence(state_exact)
    ok = True
    ok &= _assert(out_short is None,
                  f"{_pj.SEQ_LENGTH - 1} entry → None (actual: {out_short})")
    ok &= _assert(
        out_exact is not None and tuple(out_exact.shape) == (1, _pj.SEQ_LENGTH, _pj.EMBED_DIM),
        f"{_pj.SEQ_LENGTH} entry → (1, {_pj.SEQ_LENGTH}, {_pj.EMBED_DIM}) "
        f"(actual: {tuple(out_exact.shape) if out_exact is not None else None})",
    )
    return ok


def test_build_input_sequence_4_kind_filter():
    """D6+D7 予防: 4 種混在 (CLAUDE.md §5)、blank + 形状ズレ + 正常で正常のみ抽出。

    blank entry: intent/expect 両方空 (Phase 0.1.A raw 起源、Phase 1 enrich skip)
    形状ズレ entry: embedding 512D (1024 != 512)
    正常 entry: full subj + 1024D embedding

    識別力: non_empty filter 忘れ → blank 混入で seq 形状 OK でも内容汚染。
    形状チェック忘れ → 512D が tensor 構築時 RuntimeError or 不正形状。
    """
    if not _pj.is_torch_available():
        return _assert(True, "torch 不在 skip")
    print("== _build_input_sequence: 4 種混在 (blank + wrong-dim + valid) → valid のみ ==")
    entries = []
    # blank (Phase 1 enrich skip pattern)
    entries.append({"id": "sub_blank", "intent": "", "expect": ""})
    # 形状ズレ (512D)
    entries.append({
        "id": "sub_wrong_dim",
        "intent": "wrong dim text",
        "expect": "wrong dim expect",
        "embedding": [0.1] * 512,
    })
    # 正常 SEQ_LENGTH 件
    entries.extend(_make_subj_entries(_pj.SEQ_LENGTH))
    state = {"subjective_entries": entries}
    out = _jr._build_input_sequence(state)
    ok = True
    ok &= _assert(out is not None and tuple(out.shape) == (1, _pj.SEQ_LENGTH, _pj.EMBED_DIM),
                  f"形状 (1, 8, 1024) (actual: {tuple(out.shape) if out is not None else None})")
    # 識別力 content 検証: 1 entry 目の embedding は make_subj_entries[0]
    # = one-hot vector with [0]=1.0 → blank/形状ズレが含まれてないことを排他証明
    if out is not None:
        ok &= _assert(abs(float(out[0, 0, 0]) - 1.0) < 1e-6,
                      f"1 entry 目の embedding[0] = 1.0 (排他、blank/wrong 含まず、"
                      f"actual: {float(out[0, 0, 0]):.6f})")
    return ok


def test_measure_prediction_error_graceful_skip_4_kinds():
    """D8 予防: 4 種混在 (None / not list / 1024D / 512D) で graceful skip。

    識別力: 形状チェック忘れ誤実装は 512D 入力で例外 or 不正値を返す。
    """
    if not _pj.is_torch_available():
        return _assert(True, "torch 不在 skip")
    print("== measure_prediction_error: 4 種混在で graceful skip / 正常 case で float ==")
    valid = [0.0] * _pj.EMBED_DIM
    valid[0] = 1.0
    ok = True
    ok &= _assert(_jr.measure_prediction_error(None, valid) is None, "predicted=None → None")
    ok &= _assert(_jr.measure_prediction_error(valid, None) is None, "actual=None → None")
    ok &= _assert(_jr.measure_prediction_error("not list", valid) is None,
                  "predicted=str → None")
    ok &= _assert(_jr.measure_prediction_error([0.0] * 512, valid) is None,
                  "predicted=512D → None")
    ok &= _assert(_jr.measure_prediction_error(valid, [0.0] * 512) is None,
                  "actual=512D → None")
    # 正常 case で float (完全一致 → ≈ 0)
    err = _jr.measure_prediction_error(valid, valid)
    ok &= _assert(isinstance(err, float) and err < 1e-6,
                  f"正常 case (完全一致) → float ≈ 0 (actual: {err})")
    return ok


def test_append_pred_error_history_cap_fifo():
    """D9 予防: cap = PRED_ERROR_HISTORY_CAP (100) 超で FIFO 削除。

    PRED_ERROR_HISTORY_CAP + 5 件投入で len = CAP、最古 = 5 (0-4 が drop)。
    識別力: cap なし誤実装は len = CAP+5 で fail。
    """
    print("== _append_pred_error_history: cap 100 超で FIFO ==")
    state = {}
    for i in range(_jr.PRED_ERROR_HISTORY_CAP + 5):
        _jr._append_pred_error_history(state, float(i))
    history = state[_jr.STATE_FIELD_PRED_ERROR_HISTORY]
    ok = True
    ok &= _assert(len(history) == _jr.PRED_ERROR_HISTORY_CAP,
                  f"len = cap {_jr.PRED_ERROR_HISTORY_CAP} (actual: {len(history)})")
    ok &= _assert(history[0] == 5.0, f"最古 = 5 (actual: {history[0]})")
    ok &= _assert(history[-1] == float(_jr.PRED_ERROR_HISTORY_CAP + 4),
                  f"最新 = {_jr.PRED_ERROR_HISTORY_CAP + 4} (actual: {history[-1]})")
    return ok


def test_maybe_train_step_counter_trigger_and_reset():
    """D10 予防: counter < TRIGGER_EVERY で skip、達したら reset。

    counter = TRIGGER_EVERY-2 から開始 → 1 回呼出で +1 (未達)、
    2 回目で TRIGGER 達 → reset (subj 不足で train 自体は skip → False)。
    識別力: trigger 後 reset 忘れ誤実装は counter が永続 trigger 状態に留まる。
    """
    if not _pj.is_torch_available():
        return _assert(True, "torch 不在 skip")
    print("== maybe_train_step: counter +1、trigger 達で reset ==")
    state = {"subjective_entries": []}
    state[_jr.STATE_FIELD_STEP_COUNTER] = _jr.TRAIN_TRIGGER_EVERY - 2

    triggered_1 = _jr.maybe_train_step(state)
    counter_1 = state[_jr.STATE_FIELD_STEP_COUNTER]

    triggered_2 = _jr.maybe_train_step(state)
    counter_2 = state[_jr.STATE_FIELD_STEP_COUNTER]

    ok = True
    ok &= _assert(triggered_1 is False, f"trigger 未達 → False (actual: {triggered_1})")
    ok &= _assert(counter_1 == _jr.TRAIN_TRIGGER_EVERY - 1,
                  f"counter +1 (actual: {counter_1})")
    ok &= _assert(triggered_2 is False,
                  f"pairs 不足 → False (actual: {triggered_2})")
    ok &= _assert(counter_2 == 0, f"trigger 後 counter reset (actual: {counter_2})")
    return ok


def test_jepa_observe_entry_graceful_on_missing_or_wrong_embedding():
    """D11 予防: embedding 欠落 / 形状ズレ entry で全 no-op (state 不変)。

    識別力: graceful skip 効かない誤実装は state 汚染 (history に何か追加 or
    LAST_PREDICTED に不正値) で fail。Phase 0/1 影響ゼロ保証の核心。
    """
    print("== jepa_observe_entry: embedding 欠落 / 形状ズレ で state 不変 ==")
    state_no_emb = {"subjective_entries": []}
    state_wrong = {"subjective_entries": []}
    _jr.jepa_observe_entry(state_no_emb, {"id": "raw_only", "tool": "x", "result": "y"})
    _jr.jepa_observe_entry(state_wrong, {
        "id": "wrong_dim", "intent": "x", "expect": "y", "embedding": [0.1] * 512,
    })
    ok = True
    ok &= _assert(state_no_emb.get(_jr.STATE_FIELD_PRED_ERROR_HISTORY) is None,
                  "embedding なし → history 追加なし")
    ok &= _assert(state_no_emb.get(_jr.STATE_FIELD_LAST_PREDICTED) is None,
                  "embedding なし → LAST_PREDICTED 設定なし")
    ok &= _assert(state_wrong.get(_jr.STATE_FIELD_PRED_ERROR_HISTORY) is None,
                  "形状ズレ → history 追加なし")
    ok &= _assert(state_wrong.get(_jr.STATE_FIELD_LAST_PREDICTED) is None,
                  "形状ズレ → LAST_PREDICTED 設定なし")
    return ok


def test_jepa_observe_entry_normal_flow_state_order():
    """D11 予防 (順序契約): 連続 entry で state 操作順序検証。

    bootstrap 期 (subj < SEQ_LENGTH): predict_next None → state 更新なし、
    history 追加なし。
    SEQ_LENGTH 達時: predict_next 動く → LAST_PREDICTED 設定、ただし
    prev_predicted なかったので history 追加なし。
    SEQ_LENGTH+1 時: prev_predicted 存在 → measure → history に 1 件追加 +
    LAST_PREDICTED 更新。

    識別力: history 追加と predict_next の順序逆実装は bootstrap 期 (1 回目)
    から history に何か追加してしまう (or LAST_PREDICTED が actual で上書き)。
    観測 subscriber は post-write 経路で entry mutate なし (判断 5 案 b 整合)。
    """
    if not _pj.is_torch_available():
        return _assert(True, "torch 不在 skip")
    _jr.reset_singletons()
    print("== jepa_observe_entry: bootstrap → SEQ 達 → SEQ+1 で state 操作順序 ==")
    entries = _make_subj_entries(_pj.SEQ_LENGTH + 2)
    state = {"subjective_entries": []}

    # 1 回目 (bootstrap): SEQ 未達 → 全 None 維持
    state["subjective_entries"].append(entries[0])
    _jr.jepa_observe_entry(state, entries[0])
    boot_pred = state.get(_jr.STATE_FIELD_LAST_PREDICTED)
    boot_hist = state.get(_jr.STATE_FIELD_PRED_ERROR_HISTORY)

    # 2-7 回目 (SEQ 未達期): 同様
    for i in range(1, _pj.SEQ_LENGTH - 1):
        state["subjective_entries"].append(entries[i])
        _jr.jepa_observe_entry(state, entries[i])

    # SEQ_LENGTH 回目: SEQ 達 → predict_next 動 → LAST_PREDICTED 設定 (prev なし → history 追加なし)
    state["subjective_entries"].append(entries[_pj.SEQ_LENGTH - 1])
    _jr.jepa_observe_entry(state, entries[_pj.SEQ_LENGTH - 1])
    seq_reach_pred = state.get(_jr.STATE_FIELD_LAST_PREDICTED)
    seq_reach_hist = state.get(_jr.STATE_FIELD_PRED_ERROR_HISTORY)

    # SEQ_LENGTH+1 回目: prev_predicted 存在 → history 追加 + LAST_PREDICTED 更新
    state["subjective_entries"].append(entries[_pj.SEQ_LENGTH])
    _jr.jepa_observe_entry(state, entries[_pj.SEQ_LENGTH])
    plus1_pred = state.get(_jr.STATE_FIELD_LAST_PREDICTED)
    plus1_hist = state.get(_jr.STATE_FIELD_PRED_ERROR_HISTORY)

    ok = True
    ok &= _assert(boot_pred is None and boot_hist is None,
                  f"bootstrap: 全 None (actual pred={boot_pred is not None}, hist={boot_hist})")
    ok &= _assert(seq_reach_pred is not None and isinstance(seq_reach_pred, list)
                  and len(seq_reach_pred) == _pj.EMBED_DIM,
                  f"SEQ 達: LAST_PREDICTED 設定 (1024 list)")
    ok &= _assert(seq_reach_hist is None or seq_reach_hist == [],
                  f"SEQ 達: history 追加なし (prev なかった、actual: {seq_reach_hist})")
    ok &= _assert(plus1_hist is not None and len(plus1_hist) == 1
                  and isinstance(plus1_hist[0], float),
                  f"SEQ+1: history 1 件追加 (actual: {plus1_hist})")
    ok &= _assert(plus1_pred is not None and plus1_pred != seq_reach_pred,
                  "SEQ+1: LAST_PREDICTED 更新 (前回と異なる新 predict)")
    return ok


# ============================================================
# main runner (Phase 1 test_layer_b_embedding pattern 継承、run_tests.py の
# subprocess 起動経路で各 test 関数を順次実行 + exit code 判定)
# ============================================================

def main():
    results = [
        test_is_torch_available_matches_module_flag(),
        test_hybrid_loss_numeric_correctness(),
        test_cosine_distance_clamp(),
        test_jepa_model_forward_shape_and_l2_norm(),
        test_jepa_model_learns_in_one_step(),
        test_build_input_sequence_seq_length_check(),
        test_build_input_sequence_4_kind_filter(),
        test_measure_prediction_error_graceful_skip_4_kinds(),
        test_append_pred_error_history_cap_fifo(),
        test_maybe_train_step_counter_trigger_and_reset(),
        test_jepa_observe_entry_graceful_on_missing_or_wrong_embedding(),
        test_jepa_observe_entry_normal_flow_state_order(),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} test groups passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
