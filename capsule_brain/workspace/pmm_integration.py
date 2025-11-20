# -*- coding: utf-8 -*-
"""
PMM Integration Hook

Connects FRNN workspace to PMM for memory retrieval.
"""
import torch
from typing import Callable

from capsule_brain.config import CapsuleBrainConfig


def build_pmm_retrieval_fn(
    pmm,
    cfg: CapsuleBrainConfig
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build retrieval function that queries PMM from FRNN input x_t.

    x_t shape: (B, latent_dim + feelings_dim).
    We use the first latent_dim as query.

    PMM must expose: retrieve(query: torch.Tensor) -> torch.Tensor
    """
    latent_dim = cfg.latent_dim

    def retrieval_hook(x_t: torch.Tensor) -> torch.Tensor:
        query = x_t[:, :latent_dim]
        reconstruction, components = pmm.retrieve(query)
        return reconstruction

    return retrieval_hook
