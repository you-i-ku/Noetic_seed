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
import threading
from typing import Optional, List, Tuple

from core import predictor_jepa as _pj


# ============================================================
# V07.5 commit 1 (Codex AXIS 2 Medium): model eval/train mode lock
# ============================================================
# predict_next (eval) / predict_next_conditioned_batch (eval) / maybe_train_step
# (train) が同一 JEPAModel singleton の mode を切替えるため、reentrancy 競合
# を防ぐ module-level lock。Noetic は basically single-thread だが、subscriber
# 経路 (jepa_observe_entry) と Proposal 2 selection 経路の同時呼出は理論上
# 起こりうるため防御的に lock 配置 (PLAN §6-8 literal verify 対象)。
_model_mode_lock = threading.Lock()


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
    """JEPAModel singleton を lazy init で取得。torch 未 install なら None。

    V07.5 commit 1 Option B' (PLAN §4-2 #5 literal):
    `JEPAModel(seq_length=_pj.SEQ_LENGTH+1)` 呼出 override で `_PositionalEncoding`
    を `max_len=SEQ_LENGTH+1=9` で構築 → Proposal 2 (5 候補 1 batch forward) が
    T+1=9 sequence (history T=8 + π_token 1) を渡せるようにする。

    既存 `predict_next` (T=8) も `_PositionalEncoding.forward` の `pe[:, :x.size(1), :]`
    で literal 動作 (T=8 でも T=9 でも同 buffer で動く、後方互換完全維持)。

    constants `SEQ_LENGTH=8` は touch ゼロ (`stage13_phase2_status.md:107` 「触らない」literal 厳守)。
    `_get_model()` 内の `JEPAModel(...)` 呼出 override は `stage13_phase2_status.md:115`
    literal「`_pj.JEPAModel(embed_dim=512, ...)` で構築可能 = 触っていい」範囲継承。
    ゆう gut「memo に書いてたりしない?」起源 (2026-05-15 V07.5 PLAN session)。
    """
    global _model
    if not _pj.is_torch_available():
        return None
    if _model is None:
        _model = _pj.JEPAModel(seq_length=_pj.SEQ_LENGTH + 1)
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

    V07.5 commit 1 (Codex AXIS 2 Medium): _model_mode_lock で eval/train
    reentrancy guard、maybe_train_step (train mode) との競合防止。
    """
    model = _get_model()
    if model is None:
        return None
    seq = _build_input_sequence(state)
    if seq is None:
        return None
    try:
        import torch
        with _model_mode_lock:
            with torch.inference_mode():
                model.eval()
                pred = model(seq)  # (1, D)
        return pred.squeeze(0).tolist()
    except Exception as exc:
        print(f"  [jepa_runtime] predict_next skip (error: {exc})")
        return None


def _build_policy_embedding(tool: str, intent: str) -> Optional[list]:
    """π_token の 1024D embedding を bge-m3 "{tool}: {intent}" encode で生成。

    V07.5 commit 1 (A1 確定方式、PLAN §3-3 literal): semantic fusion で
    action variable (tool 離散 + intent 連続意味) を unified 意味空間に
    持ち上げる。bge-m3 既存基盤流用、追加学習ゼロ (「既存資産再配線が core」
    原則 literal 整合)。

    Codex AXIS 4 Medium (PLAN §6-9 verify): cache (embedding._embed_with_cache)
    経由で selection 毎 cycle 5 候補の同じ π_token 反復出現を cache hit で
    skip、5x latency 削減。

    Args:
        tool: tool name (e.g. "reflect", "memory_graph")
        intent: intent text (e.g. "自分は無名なのか考えたい")

    Returns:
        1024D vector (list[float]) on success、None on bge-m3 未起動 / 失敗
        (graceful skip、消費側で skip 判定)。
    """
    if not tool or not isinstance(tool, str):
        return None
    intent_str = intent if isinstance(intent, str) else ""
    text = f"{tool}: {intent_str}"
    from core.embedding import _embed_with_cache, is_vector_ready
    if not is_vector_ready():
        return None
    return _embed_with_cache(text)


def predict_next_conditioned_batch(
    state: dict, candidates: List[dict],
) -> Optional[List[Optional[list]]]:
    """policy-conditioned next prediction を 1 batch forward で N 候補同時計算。

    V07.5 commit 1 Proposal 2 = Batched Policy Tokens (PLAN §3-3 literal、Codex 推し):
    - history sequence (直近 T=SEQ_LENGTH=8 entry の embedding) を取得
    - 各 candidate の π_token (`_build_policy_embedding(tool, intent)`) を seq 末尾に append
    - (N, T+1, D) batch tensor で `JEPAModel.forward` (eval mode)
    - 結果は各 candidate の predicted next embedding list

    Option B' (PLAN §4-2 #5 literal):
    `_get_model()` で `JEPAModel(seq_length=SEQ_LENGTH+1)` 呼出 override 済 →
    `_PositionalEncoding(max_len=9)` で T+1=9 sequence literal 動作、constants
    `SEQ_LENGTH=8` touch ゼロ。

    Codex AXIS 2 Medium (PLAN §6-8 verify): `_model_mode_lock` で eval/train
    reentrancy guard、maybe_train_step (train mode) との競合防止。

    Args:
        state: state dict (history sequence の source)
        candidates: candidate list、各 dict に "tool" + "reason"/"intent" 想定

    Returns:
        各 candidate の 1024D vector list (要素 None possible: graceful skip)。
        全 skip / 例外 / torch 未 install 時は None。
    """
    if not _pj.is_torch_available():
        return None
    if not candidates:
        return []
    model = _get_model()
    if model is None:
        return None
    seq = _build_input_sequence(state)
    if seq is None:
        return None

    # 各 candidate の π_token text を集約 → 1 回 batch 呼出で 5x latency 削減
    # (Codex review 1 周目 P1 fix: _embed_batch_with_cache 経由が literal 必須)
    from core.embedding import _embed_batch_with_cache, is_vector_ready
    pi_tokens: List[Optional[list]] = [None] * len(candidates)
    texts_with_idx: List[Tuple[int, str]] = []
    for i, c in enumerate(candidates):
        if not isinstance(c, dict):
            continue
        tool_raw = c.get("tool", "")
        tool = tool_raw if isinstance(tool_raw, str) and tool_raw else ""
        intent_raw = c.get("intent") or c.get("reason", "")
        intent = intent_raw if isinstance(intent_raw, str) and intent_raw.strip() else ""
        # tool 必須 (intent 空でも tool で identify 可)、tool 欠落は graceful skip
        if tool:
            texts_with_idx.append((i, f"{tool}: {intent}"))

    if texts_with_idx and is_vector_ready():
        texts = [t for _, t in texts_with_idx]
        batch_results = _embed_batch_with_cache(texts)
        if batch_results is not None and len(batch_results) == len(texts_with_idx):
            for (idx, _), vec in zip(texts_with_idx, batch_results):
                pi_tokens[idx] = vec

    # 全 candidate の π_token が None なら graceful skip (bge-m3 未起動等)
    if all(t is None for t in pi_tokens):
        return None

    try:
        import torch
        N = len(candidates)
        T = seq.size(1)
        D = seq.size(2)
        # history を N 候補分 batch dimension に複製
        history_batch = seq.expand(N, T, D)  # (N, T, D)
        # 各 candidate の π_token tensor、None は zero vector で代用 (graceful)
        pi_tensors = []
        for t in pi_tokens:
            if t is None:
                pi_tensors.append(torch.zeros(D, dtype=torch.float32))
            else:
                pi_tensors.append(torch.tensor(t, dtype=torch.float32))
        pi_batch = torch.stack(pi_tensors, dim=0).unsqueeze(1)  # (N, 1, D)
        full_batch = torch.cat([history_batch, pi_batch], dim=1)  # (N, T+1, D)

        with _model_mode_lock:
            with torch.inference_mode():
                model.eval()
                pred = model(full_batch)  # (N, D)

        # 各 candidate に対応する predicted embedding を返却 (graceful skip 反映)
        results: List[Optional[list]] = []
        for i in range(N):
            if pi_tokens[i] is None:
                results.append(None)
            else:
                results.append(pred[i].tolist())
        return results
    except Exception as exc:
        print(f"  [jepa_runtime] predict_next_conditioned_batch skip (error: {exc})")
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

def _collect_training_pairs(
    state: dict,
) -> List[Tuple[List[list], Optional[list], list]]:
    """学習用 (input_seq, π_taken, target) triple を sliding window から構築。

    V07.5 commit 1 (PLAN §3-3 + §4-1 commit 1 literal): Proposal 2 training
    data 経路。各 entry t (t >= SEQ_LENGTH) に対し:
      - input_seq = entries[t - SEQ_LENGTH : t] の embedding sequence (既存 T=8)
      - π_taken = entries[t-1] の tool (raw 側) + intent (subj 側) を id join で
        復元し `_build_policy_embedding(tool, intent)` で 1024D 化 (新規)
      - target = entries[t] の embedding (既存)

    raw+subj id join (PLAN §6-7 literal): state["subjective_entries"] の entry id を
    key に state["raw_events"] から同 id の raw entry を lookup、raw.tool + subj.intent
    で π_token text を構築。raw 不在 / tool 欠落 / intent 欠落 case は graceful
    skip (pi_taken=None で triple に含む、消費側で skip 判定)。

    Phase 1 helper 流用契約 (`stage13_phase2_status.md:79` literal、Codex AXIS 4 Low):
    `non_empty_subjective_entries` 継続使用、blank subj 除外順序契約継承。

    SEQ_LENGTH 未満の prefix は skip。

    Returns:
        triple list `(seq, π_taken, target)`、π_taken は None possible (graceful skip)。
    """
    from core.memory import non_empty_subjective_entries
    entries = non_empty_subjective_entries(state.get("subjective_entries", []))
    recent = entries[-TRAIN_WINDOW_RECENT:]
    embedded = [
        e for e in recent
        if isinstance(e.get("embedding"), list) and len(e.get("embedding")) == _pj.EMBED_DIM
    ]
    # raw+subj id join: state["raw_events"] から同 id の raw を lookup する table
    raw_events = state.get("raw_events", [])
    raw_by_id: dict = {}
    for r in raw_events:
        if isinstance(r, dict) and r.get("id"):
            raw_by_id[r["id"]] = r

    triples: List[Tuple[List[list], Optional[list], list]] = []
    for t in range(_pj.SEQ_LENGTH, len(embedded)):
        seq = [e["embedding"] for e in embedded[t - _pj.SEQ_LENGTH:t]]
        # π_taken = entries[t-1] の tool (raw) + intent (subj)、id join 経路
        prev_subj = embedded[t - 1]
        prev_id = prev_subj.get("id")
        prev_raw = raw_by_id.get(prev_id) if prev_id else None
        # Codex review 1 周目 P1 fix: tool AND intent 両方 non-empty 必須
        # (片方欠落で空文字 embedding 生成は誤った training data になる、PLAN §6-7 graceful skip literal)
        tool = ""
        if isinstance(prev_raw, dict):
            tool_raw = prev_raw.get("tool")
            if isinstance(tool_raw, str) and tool_raw.strip():
                tool = tool_raw
        intent = ""
        intent_raw = prev_subj.get("intent")
        if isinstance(intent_raw, str) and intent_raw.strip():
            intent = intent_raw
        if tool and intent:
            pi_taken = _build_policy_embedding(tool, intent)
        else:
            pi_taken = None  # graceful skip: raw 欠落 or tool 欠落 or intent 欠落
        target = embedded[t]["embedding"]
        triples.append((seq, pi_taken, target))
    return triples


def maybe_train_step(state: dict) -> bool:
    """step_counter += 1、TRAIN_TRIGGER_EVERY 達したら train を 1 batch 実行。

    train 実行: True、skip (counter 未達 or pairs 不足 or torch 未 install): False。

    V07.5 commit 1: `_collect_training_pairs` が triple `(seq, π_taken, target)` を
    返却するように拡張済 (Proposal 2 training data 経路)、本関数では当面 (seq, target)
    pair に抽出して既存 T=8 forward 経路を維持。π_taken を使った T+1=9 policy-conditioned
    training は将来 commit (commit 3 controller 配線後に train ロジック側拡張)。

    Codex AXIS 2 Medium: `_model_mode_lock` で eval/train reentrancy guard、
    predict_next + predict_next_conditioned_batch (eval) との競合防止。
    """
    if not _pj.is_torch_available():
        return False
    counter = int(state.get(STATE_FIELD_STEP_COUNTER, 0)) + 1
    state[STATE_FIELD_STEP_COUNTER] = counter
    if counter < TRAIN_TRIGGER_EVERY:
        return False
    state[STATE_FIELD_STEP_COUNTER] = 0  # reset
    triples = _collect_training_pairs(state)
    # V07.5 commit 1: triple → pair adapter (π_taken は当面未使用、future commit で活用)
    pairs: List[Tuple[List[list], list]] = [(t[0], t[2]) for t in triples]
    if len(pairs) < TRAIN_BATCH_SIZE:
        return False
    model = _get_model()
    optimizer = _get_optimizer()
    if model is None or optimizer is None:
        return False
    try:
        import torch
        import torch.nn.functional as F
        with _model_mode_lock:
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
