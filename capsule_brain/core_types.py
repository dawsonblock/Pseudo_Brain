# -*- coding: utf-8 -*-
"""
Core Data Types for Capsule Brain

Defines SpikePacket and WorkspaceState - the fundamental data structures
that flow through the entire system.
"""
from dataclasses import dataclass
from typing import Dict, Any
import torch


@dataclass
class SpikePacket:
    """
    Canonical event container for Capsule Brain.

    All computation flows through SpikePackets.
    """
    content: torch.Tensor        # (B, latent_dim)
    routing_key: str             # e.g. "audio_in", "text_in", "internal_plan"
    priority: float              # scalar priority
    modality: str                # "audio" | "text" | "internal" | "emotion"
    metadata: Dict[str, Any]     # {tone, glyph_idx, f0, timestamp, tags, ...}


@dataclass
class WorkspaceState:
    """
    Snapshot of the global workspace at a single time step.
    """
    broadcast: torch.Tensor      # (B, latent_dim) – broadcast to capsules
    mode_probs: torch.Tensor     # (B, K) – FRNN m_t (discrete mode dist)
    current_memory: torch.Tensor # (B, mem_dim) – FRNN internal memory
    feelings: torch.Tensor       # (B, 8) – feeling vector
    last_spike: 'SpikePacket'    # last processed spike


def create_dummy_spike(
    batch_size: int = 1,
    latent_dim: int = 256,
    device: str = "cpu"
) -> SpikePacket:
    """Create a dummy SpikePacket for testing."""
    return SpikePacket(
        content=torch.randn(batch_size, latent_dim, device=device),
        routing_key="test",
        priority=0.5,
        modality="internal",
        metadata={"tone": 0, "timestamp": 0.0, "tags": ["test"]}
    )
