# -*- coding: utf-8 -*-
"""
FRNN Workspace Controller

Global Workspace implemented as an FRNN over latent + feelings.
"""
from typing import Optional, Callable, Dict, Any
import torch
import torch.nn as nn

from capsule_brain.config import CapsuleBrainConfig
from capsule_brain.core_types import SpikePacket, WorkspaceState

from capsule_brain.workspace.frnn_core_v3 import FRNNCore_v3, FRNNConfig_v3


class FRNNWorkspaceController(nn.Module):
    """
    Global Workspace implemented as an FRNN over latent + feelings.

    Input per step: SpikePacket.content (B, latent_dim) and feelings (B, 8).
    Output: WorkspaceState (broadcast + probes).
    """

    def __init__(self, cfg: CapsuleBrainConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        frnn_input_dim = cfg.latent_dim + cfg.feelings_dim

        frnn_cfg = FRNNConfig_v3(
            input_dim=frnn_input_dim,
            output_dim=cfg.latent_dim,
            num_states=cfg.num_states,
            memory_dim=cfg.memory_dim,
            hidden_dim=cfg.hidden_dim,
            gumbel_temp=1.0,
            gumbel_hard=True,
            stickiness=0.1,
            selective_write=True,
            mlp_dropout=0.1,
            attention_bank_in_readout=(cfg.bank_size > 0),
            bank_size=cfg.bank_size,
            ema_decay=0.99,
            retrieval_dim=cfg.retrieval_dim,
        )

        self.frnn = FRNNCore_v3(frnn_cfg).to(self.device)
        self._state = None
        self._pmm_retrieval_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def attach_pmm_retrieval(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Attach a retrieval function: x_t -> PMM vector.
        """
        self._pmm_retrieval_fn = fn

    def reset(self, batch_size: int = 1) -> None:
        """
        Reset FRNN internal state for a new stream.
        """
        self._state = self.frnn.reset_state(batch_size, self.device)

    @torch.no_grad()
    def step(self, spike: SpikePacket, feelings: torch.Tensor) -> WorkspaceState:
        """
        Process one SpikePacket + current feelings, return WorkspaceState.
        """
        x = spike.content.to(self.device)
        F = feelings.to(self.device)

        B, D = x.shape
        assert D == self.cfg.latent_dim
        assert F.shape == (B, self.cfg.feelings_dim)

        x_t = torch.cat([x, F], dim=1)

        retrieval_hook = None
        if self._pmm_retrieval_fn is not None and self.cfg.retrieval_dim > 0:
            def _hook(x_local: torch.Tensor) -> torch.Tensor:
                return self._pmm_retrieval_fn(x_local)
            retrieval_hook = _hook

        y_t, self._state = self.frnn.step(x_t, self._state, retrieval_hook=retrieval_hook)

        probes: Dict[str, torch.Tensor] = self.frnn.get_probes()
        m_t = probes.get("m_t")
        current_memory = probes.get("current_memory")

        if m_t is None:
            K = self.cfg.num_states
            m_t = torch.full((B, K), 1.0 / K, device=self.device)
        if current_memory is None:
            current_memory = torch.zeros(B, self.cfg.memory_dim, device=self.device)

        return WorkspaceState(
            broadcast=y_t,
            mode_probs=m_t,
            current_memory=current_memory,
            feelings=F,
            last_spike=spike,
        )

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Lightweight summary for monitoring / demo.
        """
        if self._state is None:
            return {"initialized": False}

        probes: Dict[str, torch.Tensor] = self.frnn.get_probes()
        m_t = probes.get("m_t")
        if m_t is None:
            return {"initialized": True, "mode_entropy": None}

        probs = m_t[0]
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        return {
            "initialized": True,
            "mode_entropy": entropy,
        }
