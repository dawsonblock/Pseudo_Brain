# digital_block/traits/trait_updater.py

from typing import List, Dict, Any
import torch

from .trait_vector import TraitVector
from digital_block.conversation_event import ConversationEvent


class TraitUpdater:
    """
    Slow personality updater. Uses crude gradient-like rules for now.

    You can swap this with a more principled Bayesian/optimizer later.
    """

    def __init__(self, lr: float = 1e-3) -> None:
        self.lr = lr

    def update(
        self,
        traits: TraitVector,
        events: List[ConversationEvent],
        outcome_score: float,
        target_style_adjustments: Dict[str, float],
    ) -> TraitVector:
        t = traits.to_tensor()
        grad = torch.zeros_like(t)

        # Outcome-based shaping (example rules)
        if outcome_score < 0:
            grad[0] -= 0.1  # risk_tolerance --
            grad[4] += 0.1  # caution_bias ++
        elif outcome_score > 0:
            grad[0] += 0.05  # risk_tolerance ++
            grad[3] += 0.05  # curiosity_bias ++

        # Apply explicit manual adjustments, if any
        idx = {
            "risk_tolerance": 0,
            "baseline_calm": 1,
            "baseline_positivity": 2,
            "curiosity_bias": 3,
            "caution_bias": 4,
        }
        for name, delta in target_style_adjustments.items():
            if name in idx:
                grad[idx[name]] += delta

        t_new = t + self.lr * grad
        t_new = torch.clamp(t_new, 0.0, 1.0)

        return TraitVector(
            risk_tolerance=float(t_new[0]),
            baseline_calm=float(t_new[1]),
            baseline_positivity=float(t_new[2]),
            curiosity_bias=float(t_new[3]),
            caution_bias=float(t_new[4]),
        )
