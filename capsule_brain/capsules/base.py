# -*- coding: utf-8 -*-
"""
Base Capsule Class
"""
from abc import ABC, abstractmethod
from typing import List
import torch

from capsule_brain.config import CapsuleBrainConfig
from capsule_brain.core_types import WorkspaceState, SpikePacket


class BaseCapsule(ABC):
    """
    Base class for all Capsules in Capsule Brain.

    Each capsule:
      - Receives WorkspaceState each step.
      - May maintain internal state.
      - May emit zero or more SpikePackets.
    """

    def __init__(self, cfg: CapsuleBrainConfig, name: str):
        self.cfg = cfg
        self.name = name
        self.device = torch.device(cfg.device)
        self.activation_count = 0

    def should_activate(self, workspace: WorkspaceState) -> bool:
        """
        Optional gating; default always true.
        """
        return True

    @abstractmethod
    def process(self, workspace: WorkspaceState) -> List[SpikePacket]:
        """
        Implement per-capsule logic here.
        """
        raise NotImplementedError

    def get_activation_stats(self):
        return {"name": self.name, "activations": self.activation_count}
