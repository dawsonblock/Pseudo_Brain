# -*- coding: utf-8 -*-
"""
Global Configuration for Capsule Brain
"""
from dataclasses import dataclass


@dataclass
class CapsuleBrainConfig:
    device: str = "cuda"
    latent_dim: int = 256
    feelings_dim: int = 8

    # PMM
    max_modes: int = 128
    init_modes: int = 32

    # ToneNet
    harmonics: int = 16
    sample_rate: int = 48000

    # FRNN Workspace
    num_states: int = 64
    memory_dim: int = 256
    hidden_dim: int = 256
    bank_size: int = 32
    retrieval_dim: int = 256

    # Feelings
    feeling_alpha: float = 0.3


DEFAULT_CONFIG = CapsuleBrainConfig()
