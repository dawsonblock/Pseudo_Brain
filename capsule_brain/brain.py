# -*- coding: utf-8 -*-
"""
Capsule Brain Main Orchestrator

Integrates PMM + ToneNet + Feelings + FRNN Workspace + Capsules.
"""
from typing import List, Dict, Any
import torch

from capsule_brain.config import CapsuleBrainConfig
from capsule_brain.core_types import SpikePacket, WorkspaceState
from capsule_brain.feelings import FeelingLayer
from capsule_brain.workspace.frnn_workspace import FRNNWorkspaceController
from capsule_brain.workspace.pmm_integration import build_pmm_retrieval_fn
from capsule_brain.capsules import BaseCapsule, SelfModelCapsule, SafetyCapsule

from ppm_new import StaticPseudoModeMemory
from tonenet import ToneNetRouter


class CapsuleBrain:
    """
    Main Capsule Brain orchestrator.

    Integrates:
      - PMM (StaticPseudoModeMemory)
      - ToneNet (audio front-end)
      - FeelingLayer (8D emotions)
      - FRNNWorkspaceController (global workspace)
      - Capsules (SelfModel + Safety + others later)
    """

    def __init__(self, cfg: CapsuleBrainConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # PMM
        self.pmm = StaticPseudoModeMemory(
            latent_dim=cfg.latent_dim,
            max_modes=cfg.max_modes,
            init_modes=cfg.init_modes,
            device=cfg.device,
        )

        # ToneNet
        self.tonenet = ToneNetRouter(
            self.pmm,
            harmonics=cfg.harmonics,
            sample_rate=cfg.sample_rate,
            device=cfg.device,
        )

        # Feelings
        self.feelings = FeelingLayer(alpha=cfg.feeling_alpha, device=cfg.device)

        # Workspace
        self.workspace = FRNNWorkspaceController(cfg)
        retrieval_fn = build_pmm_retrieval_fn(self.pmm, cfg)
        self.workspace.attach_pmm_retrieval(retrieval_fn)
        self.workspace.reset(batch_size=1)

        # Capsules
        self.capsules: List[BaseCapsule] = [
            SelfModelCapsule(cfg),
            SafetyCapsule(cfg),
        ]

        self.step_count = 0

    def register_capsule(self, capsule: BaseCapsule) -> None:
        self.capsules.append(capsule)

    def step(self, audio: torch.Tensor, timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Execute one full brain step given audio input (B, T).
        """
        # 1. Audio → Spike
        spike, tone_idx = self.tonenet.audio_to_spike(audio)
        spike.metadata["timestamp"] = float(timestamp)

        # 2. Update feelings
        F = self.feelings.update(tone_idx)

        # 3. Store in PMM
        pmm_result = self.pmm.store(spike)

        # 4. Workspace step
        ws_state: WorkspaceState = self.workspace.step(spike, F)

        # 5. Capsules
        capsule_outputs: List[SpikePacket] = []
        for capsule in self.capsules:
            if capsule.should_activate(ws_state):
                out = capsule.process(ws_state)
                capsule_outputs.extend(out)

        self.step_count += 1

        return {
            "step": self.step_count,
            "workspace_state": ws_state,
            "feelings": F,
            "dominant_tone": self.feelings.get_dominant_tone(),
            "pmm_novelty": float(pmm_result.get("novelty", 0.0)),
            "pmm_active_modes": int(pmm_result.get("active_modes", 0)),
            "capsule_outputs": capsule_outputs,
            "timestamp": float(timestamp),
        }

    def get_summary(self) -> Dict[str, Any]:
        occ = self.pmm.occupancy[self.pmm.active_mask]
        total_mass = float(occ.sum().item()) if occ.numel() > 0 else 0.0

        return {
            "step_count": self.step_count,
            "pmm": {
                "active_modes": int(self.pmm.active_mask.sum().item()),
                "total_mass": total_mass,
            },
            "feelings": self.feelings.to_dict(),
            "workspace": self.workspace.get_state_summary(),
            "capsules": [c.get_activation_stats() for c in self.capsules],
        }
