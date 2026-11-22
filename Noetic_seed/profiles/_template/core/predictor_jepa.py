"""段階13 Phase 2: Layer C JEPA 流次状態予測 model。

PLAN §2-4 + §22 literal: subj entry の bge-m3 1024D embedding 空間で、次 entry
がどこに出るかを予測。誤差 = 予測 latent と実 latent の cosine 距離 (空間的
構造的誤差)。既存 core/predictor.py (LLM① scalar 経路) とは責務分離 ──
pragmatic value (LLM 自己評価) vs epistemic value (data-driven 統計予測) で
Active Inference 二輪を構造分離 (判断 4 案 Y、完全独立 module)。

出口は両者とも prediction_error 共通 abstract に集約: 既存 scalar は
predictor.update_predictor_confidence の β+ 学習へ、JEPA cosine は Phase 3 で
graph 上の各 link の TE 測定経由で update_link_strength_used へ (judging 4 案 Y
の "同じ場所整合は出口側で達成" 構造)。

設計 (web 調査 V-JEPA 2 (2025/06, arxiv:2506.09985) / LeWM (2026/3, arxiv:2603.19312) 反映):
  - causal transformer (decoder-only)、最終 token から次 entry を predict
  - 4 layer / 8 head (head_dim=128) / hidden_dim=1024 (bge-m3 native) / mlp_ratio=2
  - seq_length=8 (直近 8 entry の embedding sequence)
  - hybrid loss: 0.8·(1-cos(pred, actual)) + 0.2·MSE(pred, actual)
  - 出力 L2 normalize 強制 (bge-m3 の L2 normalized 空間と一致)
  - ~33M param、CPU 1 step 0.5-2 秒目安
  - bge-m3 が pre-trained 固定 = 射影層 train する gradient signal なし、
    LeWM 流の低次元射影は Noetic では不要 → 1024D ネイティブで使う
  - encoder 固定 = trivial collapse 構造的になし → SIGReg 不要 (LeWM 比簡素化)

graceful skip: torch optional import、未 install 時は is_torch_available()=False
で caller が全機能 skip (Phase 0/1 への影響ゼロ保証)。
"""
from __future__ import annotations

import math
from typing import Optional

# 段階12 venv per-profile + requirements.txt --extra-index-url で CPU 版 torch を
# 確実に install する経路。万一 install 失敗してもこの module の他の経路は
# graceful skip で動く (Phase 2 commit 1 範囲では import 失敗 = Layer C 全機能無効、
# Phase 0/1 は無影響継続)。
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


# ============================================================
# モデル定数 (web 調査 V-JEPA 2 / LeWM 2026 update + Noetic CPU 制約 反映)
# ============================================================

EMBED_DIM = 1024            # bge-m3 native dim (entity_resolver と同 source、Phase 1 流用)
NUM_LAYERS = 4              # LeWM (6) より浅い、CPU + seq=8 短に lean
NUM_HEADS = 8               # head_dim = EMBED_DIM // NUM_HEADS = 128
MLP_RATIO = 2               # FFN inner = EMBED_DIM * 2、通常 4 を半減で param 削減
SEQ_LENGTH = 8              # 直近 N entry を入力に取る history window
LOSS_COSINE_WEIGHT = 0.8    # hybrid loss における cosine の重み
LOSS_MSE_WEIGHT = 0.2       # hybrid loss における MSE の重み


def is_torch_available() -> bool:
    """torch optional import が成功してるか。caller の graceful skip 判定用。"""
    return _TORCH_AVAILABLE


if _TORCH_AVAILABLE:
    class _PositionalEncoding(nn.Module):
        """sinusoidal positional encoding。RoPE は seq=8 で overkill、parameter free。"""

        def __init__(self, dim: int = EMBED_DIM, max_len: int = SEQ_LENGTH):
            super().__init__()
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

        def forward(self, x):  # x: (B, T, D)
            return x + self.pe[:, : x.size(1), :]


    class JEPAModel(nn.Module):
        """causal transformer による次 entry embedding 予測 model。

        入力: (B, seq_length, EMBED_DIM) bge-m3 embedding sequence
        出力: (B, EMBED_DIM) 次 entry の予測 embedding (L2 normalized)

        causal mask で future token を見ない、最終 token を decoder-only style で
        次予測に使う。出力は L2 normalize 強制で bge-m3 と同空間に揃える。
        """

        def __init__(
            self,
            embed_dim: int = EMBED_DIM,
            num_layers: int = NUM_LAYERS,
            num_heads: int = NUM_HEADS,
            mlp_ratio: int = MLP_RATIO,
            seq_length: int = SEQ_LENGTH,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.seq_length = seq_length
            self.pos_enc = _PositionalEncoding(embed_dim, max_len=seq_length)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, seq):
            """seq: (B, T, D) → (B, D) L2 normalized 予測 embedding。"""
            T = seq.size(1)
            x = self.pos_enc(seq)
            # causal mask (T, T): True = mask out (future tokens を見ない)
            mask = torch.triu(torch.ones(T, T, device=seq.device), diagonal=1).bool()
            out = self.transformer(x, mask=mask)
            last = self.norm(out[:, -1, :])  # 最終 token を次予測 head に
            return F.normalize(last, p=2, dim=-1)


    def hybrid_loss(predicted, actual):
        """hybrid loss: LOSS_COSINE_WEIGHT·(1-cos) + LOSS_MSE_WEIGHT·MSE。

        bge-m3 は L2 normalized なので、数学的には ||a-b||² = 2(1-cos) で MSE と
        cosine は等価だが、勾配の局所性が cosine の方が高次元で安定。MSE 補助で
        magnitude 学習も保持 (web 調査 §4 reference、sentence-transformers
        CosineSimilarityLoss と同設計、加えて encoder 固定で SIGReg 不要)。

        predicted, actual: (B, D) tensors、両方 L2 normalized 前提。
        return: scalar tensor。
        """
        cos_sim = (predicted * actual).sum(dim=-1)  # (B,)
        cos_loss = (1.0 - cos_sim).mean()
        mse_loss = F.mse_loss(predicted, actual)
        return LOSS_COSINE_WEIGHT * cos_loss + LOSS_MSE_WEIGHT * mse_loss


    def cosine_distance(predicted, actual) -> float:
        """1 - cosine similarity を 0.0-1.0 に clamp。prediction_error 計算用。

        predicted, actual: (D,) tensors、両方 L2 normalized 前提。
        return: float (0.0 = 完全一致、1.0 = 直交以上、Phase 3 入力素材)。
        """
        cos_sim = float((predicted * actual).sum().item())
        return max(0.0, min(1.0, 1.0 - cos_sim))
