# -*- coding: utf-8 -*-
"""
Feeling Layer

Maintains an 8D emotion distribution over tones, smoothly updated via EMA.
"""
from dataclasses import dataclass
from typing import Dict, Any
import torch


@dataclass
class FeelingLayer:
    """
    Maintains 8D internal feeling vector F (softmax-normalized).

    Index mapping:
      0: calm
      1: focused
      2: excited
      3: warm
      4: urgent
      5: uncertain
      6: stressed
      7: intense
    """
    alpha: float = 0.3
    device: str = "cpu"

    def __post_init__(self):
        self.device = torch.device(self.device)
        self.F = torch.ones(1, 8, device=self.device) / 8.0

    def update(self, tone_idx: int) -> torch.Tensor:
        """
        Update feelings with EMA towards new tone label.
        """
        one_hot = torch.zeros_like(self.F)
        one_hot[0, tone_idx] = 1.0
        self.F = (1.0 - self.alpha) * self.F + self.alpha * one_hot
        self.F = torch.softmax(self.F, dim=-1)
        return self.F.clone()

    def get_dominant_tone(self) -> int:
        return int(torch.argmax(self.F[0]).item())

    def to_tensor(self) -> torch.Tensor:
        return self.F.clone()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self.F[0].tolist(),
            "dominant_idx": self.get_dominant_tone(),
        }
