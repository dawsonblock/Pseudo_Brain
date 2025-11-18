# -*- coding: utf-8 -*-
"""
Safety Capsule

Monitors workspace and feelings for instability.
"""
from typing import List
import torch

from capsule_brain.config import CapsuleBrainConfig
from capsule_brain.core_types import WorkspaceState, SpikePacket
from capsule_brain.capsules.base import BaseCapsule


class SafetyCapsule(BaseCapsule):
    """
    Monitors workspace and feelings for instability.

    Heuristics:
      - Rapid mode switching (high diversity over recent steps).
      - Feeling stuck in a single index for many steps.
    """

    def __init__(self, cfg: CapsuleBrainConfig):
        super().__init__(cfg, name="safety")
        self.mode_history = []
        self.feeling_history = []
        self.max_hist = 20

    def process(self, ws_state: WorkspaceState) -> List[SpikePacket]:
        self.activation_count += 1
        warnings = []

        dom_mode = int(torch.argmax(ws_state.mode_probs[0]).item())
        dom_feel = int(torch.argmax(ws_state.feelings[0]).item())

        self.mode_history.append(dom_mode)
        self.feeling_history.append(dom_feel)
        if len(self.mode_history) > self.max_hist:
            self.mode_history.pop(0)
        if len(self.feeling_history) > self.max_hist:
            self.feeling_history.pop(0)

        # Rapid mode switching – last 5 all different
        if len(self.mode_history) >= 5:
            if len(set(self.mode_history[-5:])) == 5:
                msg = "Rapid mode switching detected (5 distinct modes in last 5 steps)."
                warnings.append(msg)

        # Feeling stuck – last 10 identical
        if len(self.feeling_history) >= 10:
            if len(set(self.feeling_history[-10:])) == 1:
                warnings.append("Feeling stuck in a single state for last 10 steps.")

        if not warnings:
            return []

        spike = SpikePacket(
            content=ws_state.broadcast,
            routing_key="safety_warning",
            priority=0.9,
            modality="internal",
            metadata={
                "type": "safety_warning",
                "warnings": warnings,
                "mode_history": self.mode_history[-5:],
                "feeling_history": self.feeling_history[-5:],
            },
        )
        return [spike]
