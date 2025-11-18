# digital_block/emotion/contagion.py

from typing import Optional
import torch


def contagion_update(
    E_prev: torch.Tensor,
    E_candidate: torch.Tensor,
    E_human: Optional[torch.Tensor],
    traits: torch.Tensor,
    kappa_base: float = 0.3,
) -> torch.Tensor:
    """
    Emotional contagion from human → AI, filtered by traits.

    If E_human is None, this reduces to just returning E_candidate.
    """

    if E_human is None:
        return E_candidate

    risk_tolerance, baseline_calm, baseline_positivity, curiosity_bias, caution_bias = traits.tolist()

    kappa = kappa_base * (0.5 + 0.5 * curiosity_bias) * (1.0 - 0.5 * baseline_calm)
    kappa = max(0.0, min(0.8, kappa))

    E_trans = E_human.clone()
    if E_trans.numel() > 1:
        # Example: dampen "arousal" index 1 by calm
        E_trans[1] = E_trans[1] * (1.0 - baseline_calm)

    E_mixed = (1.0 - kappa) * E_candidate + kappa * E_trans
    return E_mixed.clamp(-1.0, 1.0)
