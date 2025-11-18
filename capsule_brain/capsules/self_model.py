# -*- coding: utf-8 -*-
"""
Self-Model Capsule

Introspective capsule that generates internal explanations.
"""
from typing import List
import torch

from capsule_brain.config import CapsuleBrainConfig
from capsule_brain.core_types import WorkspaceState, SpikePacket
from capsule_brain.capsules.base import BaseCapsule


class SelfModelCapsule(BaseCapsule):
    """
    Introspective capsule.

    Generates internal explanations of the current workspace and feeling state.
    """

    def __init__(self, cfg: CapsuleBrainConfig):
        super().__init__(cfg, name="self_model")

    def process(self, ws_state: WorkspaceState) -> List[SpikePacket]:
        self.activation_count += 1

        dominant_mode = int(torch.argmax(ws_state.mode_probs[0]).item())
        mode_prob = float(ws_state.mode_probs[0, dominant_mode].item())

        dominant_feeling = int(torch.argmax(ws_state.feelings[0]).item())
        feelings_vector = ws_state.feelings[0].tolist()

        explanation = {
            "type": "self_model_explanation",
            "dominant_mode": dominant_mode,
            "mode_confidence": mode_prob,
            "dominant_feeling": dominant_feeling,
            "feelings_distribution": feelings_vector,
            "message": (
                f"Currently broadcasting from mode {dominant_mode} with "
                f"{mode_prob:.2f} confidence and feeling index {dominant_feeling}."
            ),
        }

        spike = SpikePacket(
            content=ws_state.broadcast,
            routing_key="self_model_output",
            priority=0.6,
            modality="internal",
            metadata=explanation,
        )
        return [spike]
