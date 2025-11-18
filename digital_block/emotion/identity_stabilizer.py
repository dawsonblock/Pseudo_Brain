# digital_block/emotion/identity_stabilizer.py

from typing import Optional
import torch
import torch.nn as nn


class IdentityStabilizer(nn.Module):
    """
    Keeps emotional state aligned with:
      - trait-defined target in emotion space
      - optional global baseline (E_BASELINE)

    Mechanism:
      E_out = (1 - alpha) * E + alpha * target
      target = 0.5 * trait_to_emotion(traits) + 0.5 * baseline   (if baseline given)
    """

    def __init__(self, trait_dim: int = 5, emotion_dim: int = 16) -> None:
        super().__init__()
        self.trait_to_emotion = nn.Sequential(
            nn.Linear(trait_dim, 32),
            nn.ReLU(),
            nn.Linear(32, emotion_dim),
            nn.Tanh(),
        )
        # How strongly we pull emotion toward the trait/baseline anchor
        self.alpha = nn.Parameter(torch.tensor(0.05))

    def forward(
        self,
        E: torch.Tensor,
        traits: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if E.ndim == 1:
            E_in = E.unsqueeze(0)
        else:
            E_in = E

        traits = traits.to(E_in.device)
        traits_expanded = traits.unsqueeze(0)  # [1, trait_dim]

        target_from_traits = self.trait_to_emotion(traits_expanded)  # [1, emotion_dim]

        if baseline is not None:
            baseline = baseline.to(E_in.device)
            if baseline.ndim == 1:
                baseline = baseline.unsqueeze(0)
            target = 0.5 * target_from_traits + 0.5 * baseline
        else:
            target = target_from_traits

        alpha = torch.clamp(self.alpha, 0.0, 0.3)  # safety
        E_out = (1.0 - alpha) * E_in + alpha * target
        return E_out.squeeze(0).clamp(-1.0, 1.0)

    def stabilize(
        self,
        E: torch.Tensor,
        traits: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(E, traits, baseline=baseline)
