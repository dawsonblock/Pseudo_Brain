# digital_block/emotion/emotion_state_engine.py

from typing import Dict
import torch
import torch.nn as nn

from digital_block.block_style.block_style_mapper import AFFECT_KEYS, BLOCK_KEYS


class EmotionStateEngine(nn.Module):
    """
    Recurrent emotional state engine with inertia.

    Computes E_t from:
      - previous emotion E_{t-1}
      - fused affect
      - Block-style sliders
      - traits
    """

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.dim = dim

        in_dim = dim + len(AFFECT_KEYS) + len(BLOCK_KEYS) + 5  # traits=5

        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, dim),
            nn.Tanh(),
        )

        # Learnable decay, but clamped for stability
        self.decay = nn.Parameter(torch.tensor(0.92))

    def forward(
        self,
        E_prev: torch.Tensor,
        fused_affect: Dict[str, float],
        block_labels: Dict[str, float],
        traits: torch.Tensor,
    ) -> torch.Tensor:
        if E_prev.ndim != 1:
            raise ValueError("EmotionStateEngine expects E_prev as 1D [dim] tensor")

        aff_vec = torch.tensor(
            [fused_affect.get(k, 0.0) for k in AFFECT_KEYS], dtype=torch.float32
        )
        blk_vec = torch.tensor(
            [block_labels.get(k, 0.5) for k in BLOCK_KEYS], dtype=torch.float32
        )

        x = torch.cat([E_prev, aff_vec, blk_vec, traits], dim=0)
        delta = self.update_mlp(x)

        decay = torch.clamp(self.decay, 0.7, 0.99)
        E_new = decay * E_prev + (1.0 - decay) * delta
        return E_new.clamp(-1.0, 1.0)
