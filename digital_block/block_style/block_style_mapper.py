# digital_block/block_style/block_style_mapper.py

from typing import Dict
import torch
import torch.nn as nn

AFFECT_KEYS = [
    "valence",
    "arousal",
    "anxiety",
    "anger",
    "sadness",
    "curiosity",
    "confidence",
    "tension",
]

BLOCK_KEYS = ["clarity", "calm", "caution", "curiosity", "positivity"]


class BlockStyleMapper(nn.Module):
    """
    Generic affect → Block-style sliders in [0,1]:
      clarity, calm, caution, curiosity, positivity
    """

    def __init__(self) -> None:
        super().__init__()
        in_dim = len(AFFECT_KEYS)
        hidden = 16
        out_dim = len(BLOCK_KEYS)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def forward_from_dict(self, affect: Dict[str, float]) -> Dict[str, float]:
        x = torch.tensor([[affect.get(k, 0.0) for k in AFFECT_KEYS]], dtype=torch.float32)
        with torch.no_grad():
            out = self.mlp(x)[0].cpu().numpy().tolist()
        return {name: float(v) for name, v in zip(BLOCK_KEYS, out)}
