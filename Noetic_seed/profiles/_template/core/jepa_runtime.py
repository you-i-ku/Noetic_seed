"""段階13 Phase 2: Layer C JEPA runtime (predict / train / 配線 helper)。

PLAN §22-5 接続点 (a)-(d) literal の実装層: model arch + 学習データ + 誤差 modulator
+ 依存管理。

設計:
- 経路 A (cycle 内 predict、event_emitter subscriber 経由): subj entry 永続化後に
  「前回 predicted vs 今回 actual」cosine 距離 = prediction_error を計算、
  state history に蓄積 (Phase 3 入力素材)。次 entry の predict_next を計算して
  state に保存 (次回比較用)。
- 経路 B (background train、同 subscriber から起動): N=TRAIN_TRIGGER_EVERY entry
  蓄積で 1 batch (TRAIN_STEPS_PER_TRIGGER step)、最新 TRAIN_WINDOW_RECENT 件
  sliding window、batch TRAIN_BATCH_SIZE、AdamW lr TRAIN_LR。Phase 1 helper
  non_empty_subjective_entries 流用で blank 除外 (judging 7 同型契約継承)。

判断 8 案 a deferred 整合: prediction_error → memory_links.update_link_strength_used
への link 接続は Phase 3 (TE + Bayesian) で「どの link を update するか」を
graph 全 link の TE evaluate という自然構造で確定させる。Phase 2 では state
history (STATE_FIELD_PRED_ERROR_HISTORY) に蓄積して Phase 3 の Bayesian
posterior 入力素材として待機 (PLAN §4-3 「lag-1 TE + Beta posterior + 既存
prediction error modulator の 3 段重ね」着地点)。

graceful skip: predictor_jepa.is_torch_available() = False なら全 op no-op
(Phase 0/1 影響ゼロ保証、判断 4 案 Y boundary)。
"""
from __future__ import annotations

import random
from typing import Optional, List, Tuple

from core import predictor_jepa as _pj


# ============================================================
# 学習設定 constants (web 調査 §4 reference + Noetic CPU 制約)
# ============================================================

TRAIN_TRIGGER_EVERY = 32        # N entry 蓄積で 1 trigger
TRAIN_BATCH_SIZE = 8            # CPU 学習で realistic、Noetic 蓄積速度に合う
TRAIN_STEPS_PER_TRIGGER = 5     # 1 trigger で 5 step (online learning)
TRAIN_WINDOW_RECENT = 256       # sliding window: 最新 N entry を学習対象
TRAIN_LR = 1e-4                 # small transformer 標準
TRAIN_WEIGHT_DECAY = 0.01       # AdamW 標準


# ============================================================
# state field 命名 (Noetic 慣習: predictor_confidence /
# prediction_error_history_e2 系統と整合、判断 6)
# ============================================================

STATE_FIELD_LAST_PREDICTED = "jepa_last_predicted_embedding"
STATE_FIELD_STEP_COUNTER = "jepa_step_counter"
STATE_FIELD_PRED_ERROR_HISTORY = "jepa_prediction_error_history"
PRED_ERROR_HISTORY_CAP = 100    # predictor.HISTORY_CAP と同値、PC メモリ現実解


# ============================================================
# singleton model + optimizer (lazy init、torch 不在時は None)
# ============================================================

_model = None
_optimizer = None


def _get_model():
    """JEPAModel singleton を lazy init で取得。torch 未 install なら None。"""
    global _model
    if not _pj.is_torch_available():
        return None
    if _model is None:
        _model = _pj.JEPAModel()
    return _model


def _get_optimizer():
    """AdamW optimizer singleton を lazy init で取得。"""
    global _optimizer
    if not _pj.is_torch_available():
        return None
    if _optimizer is None:
        model = _get_model()
        if model is None:
            return None
        import torch
        _optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAIN_LR,
            weight_decay=TRAIN_WEIGHT_DECAY,
        )
    return _optimizer


def reset_singletons() -> None:
    """test 用 cleanup。production では呼ばない (event_emitter.reset_subscribers
    と同パターン、判断 4 案 Y boundary 維持)。"""
    global _model, _optimizer
    _model = None
    _optimizer = None


# ============================================================
# 経路 A: predict + 誤差計算 (subscriber 経由で呼ばれる)
# ============================================================

def _build_input_sequence(state: dict):
    """state["subjective_entries"] から直近 SEQ_LENGTH entry の embedding を取得し
    (1, SEQ_LENGTH, EMBED_DIM) tensor を返す。

    entry 不足 / embedding 欠落時は None (graceful skip)。Phase 1 helper
    non_empty_subjective_entries 流用で blank 除外 (filter 順序契約継承)。
    """
    if not _pj.is_torch_available():
        return None
    from core.memory import non_empty_subjective_entries
    import torch
    entries = non_empty_subjective_entries(state.get("subjective_entries", []))
    embedded = [
        e.get("embedding") for e in entries
        if isinstance(e.get("embedding"), list) and len(e.get("embedding")) == _pj.EMBED_DIM
    ]
    if len(embedded) < _pj.SEQ_LENGTH:
        return None
    seq = embedded[-_pj.SEQ_LENGTH:]
    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, T, D)


def predict_next(state: dict) -> Optional[list]:
    """直近 sequence から次 entry の embedding を predict。Python list で返す。

    torch 未 install / sequence 不足 / 例外時は None (graceful skip、Phase 0/1
    無影響保証)。
    """
    model = _get_model()
    if model is None:
        return None
    seq = _build_input_sequence(state)
    if seq is None:
        return None
    try:
        import torch
        with torch.inference_mode():
            model.eval()
            pred = model(seq)  # (1, D)
        return pred.squeeze(0).tolist()
    except Exception as exc:
        print(f"  [jepa_runtime] predict_next skip (error: {exc})")
        return None


def measure_prediction_error(predicted: list, actual: list) -> Optional[float]:
    """前回 predicted (list) と actual (list) の cosine 距離を 0.0-1.0 で返す。

    両者とも L2 normalized 1024D 想定 (predict_next 出力 / bge-m3 出力)。
    None / 形状ズレ / torch 未 install は None (graceful skip)。
    """
    if not _pj.is_torch_available():
        return None
    if not (isinstance(predicted, list) and isinstance(actual, list)):
        return None
    if len(predicted) != _pj.EMBED_DIM or len(actual) != _pj.EMBED_DIM:
        return None
    try:
        import torch
        p = torch.tensor(predicted, dtype=torch.float32)
        a = torch.tensor(actual, dtype=torch.float32)
        return _pj.cosine_distance(p, a)
    except Exception as exc:
        print(f"  [jepa_runtime] measure_prediction_error skip (error: {exc})")
        return None


def _append_pred_error_history(state: dict, err: float) -> None:
    """prediction_error history に append、cap PRED_ERROR_HISTORY_CAP。

    Phase 3 の Bayesian posterior 入力素材として蓄積 (judging 8 案 a deferred、
    PLAN §4-3 「3 段重ね現実解」着地点)。predictor.py の _append_history と
    同 pattern (FIFO、abs() 不要 ── 既に 0-1 clamp 済)。
    """
    history = state.setdefault(STATE_FIELD_PRED_ERROR_HISTORY, [])
    history.append(float(err))
    if len(history) > PRED_ERROR_HISTORY_CAP:
        del history[0]


# ============================================================
# 経路 B: background train (同 subscriber から trigger 起動)
# ============================================================

def _collect_training_pairs(state: dict) -> List[Tuple[List[list], list]]:
    """学習用 (input_seq, target) ペアを sliding window から構築。

    各 entry t (t >= SEQ_LENGTH) に対し:
      - input = entries[t - SEQ_LENGTH : t] の embedding sequence
      - target = entries[t] の embedding
    SEQ_LENGTH 未満の prefix は skip。
    """
    from core.memory import non_empty_subjective_entries
    entries = non_empty_subjective_entries(state.get("subjective_entries", []))
    recent = entries[-TRAIN_WINDOW_RECENT:]
    embedded = [
        e.get("embedding") for e in recent
        if isinstance(e.get("embedding"), list) and len(e.get("embedding")) == _pj.EMBED_DIM
    ]
    pairs: List[Tuple[List[list], list]] = []
    for t in range(_pj.SEQ_LENGTH, len(embedded)):
        seq = embedded[t - _pj.SEQ_LENGTH:t]
        target = embedded[t]
        pairs.append((seq, target))
    return pairs


def maybe_train_step(state: dict) -> bool:
    """step_counter += 1、TRAIN_TRIGGER_EVERY 達したら train を 1 batch 実行。

    train 実行: True、skip (counter 未達 or pairs 不足 or torch 未 install): False。
    """
    if not _pj.is_torch_available():
        return False
    counter = int(state.get(STATE_FIELD_STEP_COUNTER, 0)) + 1
    state[STATE_FIELD_STEP_COUNTER] = counter
    if counter < TRAIN_TRIGGER_EVERY:
        return False
    state[STATE_FIELD_STEP_COUNTER] = 0  # reset
    pairs = _collect_training_pairs(state)
    if len(pairs) < TRAIN_BATCH_SIZE:
        return False
    model = _get_model()
    optimizer = _get_optimizer()
    if model is None or optimizer is None:
        return False
    try:
        import torch
        import torch.nn.functional as F
        model.train()
        for _ in range(TRAIN_STEPS_PER_TRIGGER):
            batch = random.sample(pairs, k=min(TRAIN_BATCH_SIZE, len(pairs)))
            seqs = torch.tensor([p[0] for p in batch], dtype=torch.float32)
            targets = torch.tensor([p[1] for p in batch], dtype=torch.float32)
            targets = F.normalize(targets, p=2, dim=-1)  # bge-m3 既 normalize だが防御的
            pred = model(seqs)
            loss = _pj.hybrid_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return True
    except Exception as exc:
        print(f"  [jepa_runtime] maybe_train_step skip (error: {exc})")
        return False


# ============================================================
# 配線 observer (event_emitter subscribe 経由で呼ばれる)
# ============================================================

def jepa_observe_entry(state: dict, entry: dict) -> None:
    """event_emitter subscriber: subj entry 永続化後に
      1. 前回 predicted があれば actual との prediction_error 計算 + state history 蓄積
      2. 次 predict_next 計算 → state に保存 (次回比較用)
      3. step_counter += 1、trigger で background train

    Phase 0.4 check_raw_subjective_gap と同 pattern (post-write 観測 subscriber、
    entry mutate しない)。subj enrich 済 (Phase 1 _enrich_subjective_inline 経由
    で embedding が entry root に注入された) entry のみ対象。

    torch 未 install / subj 不在 / sequence 不足は全 graceful skip
    (Phase 0/1 影響ゼロ保証)。例外は print のみで握りつぶし、event_emitter の
    isolation 経路と整合 (他 subscriber を止めない)。
    """
    if not _pj.is_torch_available():
        return
    actual_embedding = entry.get("embedding")
    if not (isinstance(actual_embedding, list) and len(actual_embedding) == _pj.EMBED_DIM):
        return

    # 1. 前回 predicted があれば prediction_error 計算 + history 蓄積
    prev_predicted = state.get(STATE_FIELD_LAST_PREDICTED)
    if isinstance(prev_predicted, list):
        err = measure_prediction_error(prev_predicted, actual_embedding)
        if err is not None:
            _append_pred_error_history(state, err)

    # 2. 次 predict_next を計算 → state に保存 (次回比較用)
    next_pred = predict_next(state)
    if next_pred is not None:
        state[STATE_FIELD_LAST_PREDICTED] = next_pred

    # 3. step_counter += 1、trigger で background train
    maybe_train_step(state)
