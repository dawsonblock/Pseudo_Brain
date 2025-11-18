# -*- coding: utf-8 -*-
"""
FRNN Workspace Controller

Wraps FRNNCore_v3 to integrate with Capsule Brain components:
- Combines spike content + feelings as input
- Integrates PMM via retrieval hook
- Returns WorkspaceState
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional, Callable
import torch
import torch.nn as nn

from core_types import SpikePacket, WorkspaceState
from workspace.frnn_core_v3 import FRNNCore_v3, FRNNConfig_v3, FRNNState
from config import CapsuleBrainConfig


class FRNNWorkspaceController(nn.Module):
    """
    Global Workspace implemented as FRNN over SpikePackets.

    Integrates:
    - FRNNCore_v3 as recurrent engine
    - PMM via retrieval_hook
    - Feelings as part of input
    """

    def __init__(self, cfg: CapsuleBrainConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Build FRNN config
        frnn_cfg = FRNNConfig_v3(
            input_dim=cfg.frnn_input_dim,  # latent_dim + feelings_dim
            output_dim=cfg.latent_dim,  # Broadcast in latent space
            num_states=cfg.num_states,
            memory_dim=cfg.memory_dim,
            hidden_dim=cfg.hidden_dim,
            gumbel_temp=cfg.gumbel_temp,
            gumbel_hard=cfg.gumbel_hard,
            stickiness=cfg.stickiness,
            selective_write=cfg.selective_write,
            mlp_dropout=0.1,
            attention_bank_in_readout=(cfg.bank_size > 0),
            bank_size=cfg.bank_size,
            ema_decay=cfg.ema_decay,
            retrieval_dim=cfg.retrieval_dim
        )

        # Create FRNN core
        self.frnn = FRNNCore_v3(frnn_cfg).to(self.device)

        # FRNN state
        self._state: Optional[FRNNState] = None

        # PMM retrieval hook
        self._pmm_retrieval_fn: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None

        # Step counter
        self.step_count = 0

    def attach_pmm_retrieval(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """
        Attach PMM retrieval function.

        Args:
            fn: Maps x_t (B, input_dim) -> (B, retrieval_dim)
        """
        self._pmm_retrieval_fn = fn

    def reset(self, batch_size: int = 1) -> None:
        """Reset FRNN state for new stream."""
        self._state = self.frnn.reset_state(batch_size, self.device)
        self.step_count = 0

    @torch.no_grad()
    def step(
        self,
        spike: SpikePacket,
        feelings: torch.Tensor
    ) -> WorkspaceState:
        """
        Process one SpikePacket + feelings, return WorkspaceState.

        Args:
            spike: Input spike with content (B, latent_dim)
            feelings: Current feeling vector (B, 8)

        Returns:
            WorkspaceState with broadcast, mode_probs, etc.
        """
        # Ensure device consistency
        x = spike.content.to(self.device)
        F = feelings.to(self.device)

        B, D = x.shape
        assert D == self.cfg.latent_dim, \
            f"Expected latent_dim={self.cfg.latent_dim}, got {D}"
        assert F.shape == (B, self.cfg.feelings_dim), \
            f"Feelings must be (B, {self.cfg.feelings_dim})"

        # Build FRNN input: concat(content, feelings)
        x_t = torch.cat([x, F], dim=1)  # (B, latent_dim + 8)

        # Build retrieval hook if PMM attached
        retrieval_hook = None
        if self.cfg.retrieval_dim > 0 and self._pmm_retrieval_fn is not None:
            def _hook(x_local: torch.Tensor) -> torch.Tensor:
                return self._pmm_retrieval_fn(x_local)
            retrieval_hook = _hook

        # Run FRNN step
        y_t, self._state = self.frnn.step(
            x_t,
            self._state,
            retrieval_hook=retrieval_hook
        )

        # Extract probes
        probes = self.frnn.get_probes()
        m_t = probes.get("m_t", None)  # (B, K)
        current_memory = probes.get("current_memory", None)  # (B, mem_dim)

        # Fallbacks if probes not populated
        if m_t is None:
            K = self.cfg.num_states
            m_t = torch.full((B, K), 1.0 / K, device=self.device)
        if current_memory is None:
            current_memory = torch.zeros(
                B, self.cfg.memory_dim, device=self.device
            )

        # Construct WorkspaceState
        ws_state = WorkspaceState(
            broadcast=y_t,  # (B, latent_dim)
            mode_probs=m_t,  # (B, K)
            current_memory=current_memory,  # (B, mem_dim)
            feelings=F,  # (B, 8)
            last_spike=spike,
            step_count=self.step_count
        )

        self.step_count += 1

        return ws_state

    def get_state_summary(self) -> dict:
        """Return summary of current FRNN state."""
        if self._state is None:
            return {"initialized": False}

        m_t = self._state.m_t
        B, K = m_t.shape

        # Find dominant mode
        dominant_mode = torch.argmax(m_t[0]).item()
        dominant_prob = m_t[0, dominant_mode].item()

        # Entropy of mode distribution
        entropy = -(m_t * torch.log(m_t + 1e-8)).sum(dim=1).mean().item()

        return {
            "initialized": True,
            "step_count": self.step_count,
            "num_modes": K,
            "dominant_mode": dominant_mode,
            "dominant_prob": dominant_prob,
            "mode_entropy": entropy
        }
