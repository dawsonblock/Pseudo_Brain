# -*- coding: utf-8 -*-
"""
ToneNet Router - Minimal Stub for Capsule Brain

Provides basic audio-to-latent conversion and crude tone classification.
Replace with full harmonic implementation later.
"""
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn

from capsule_brain.core_types import SpikePacket


class ToneNetRouter(nn.Module):
    """
    Minimal ToneNet stub.

    Responsibilities:
      - Convert raw audio (B, T) into latent content (B, latent_dim).
      - Assign a crude tone index (0–7) based on signal energy.
      - Package everything into a SpikePacket.

    This is a placeholder implementation to get Capsule Brain running.
    """

    def __init__(
        self,
        pmm,
        harmonics: int = 16,
        sample_rate: int = 48_000,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.harmonics = harmonics

        # Use PMM latent_dim so everything stays consistent.
        if not hasattr(pmm, "latent_dim"):
            raise ValueError("PMM object must expose .latent_dim")

        self.latent_dim = int(pmm.latent_dim)

        # Simple learnable projection: features → latent space
        feature_dim = 4  # [mean, std, energy, max_abs]
        self.proj = nn.Linear(feature_dim, self.latent_dim, bias=True)
        self.proj = self.proj.to(self.device)

    def _extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compress audio (B, T) into simple features (B, 4).

        Features:
          - mean
          - std
          - energy (mean of squared)
          - max abs value
        """
        # Ensure 2D
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)

        mean = audio.mean(dim=1, keepdim=True)
        std = audio.std(dim=1, keepdim=True)
        energy = (audio ** 2).mean(dim=1, keepdim=True)
        max_abs = audio.abs().max(dim=1, keepdim=True).values

        feats = torch.cat([mean, std, energy, max_abs], dim=1)
        return feats

    def _classify_tone(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Crude tone classifier: maps average energy to 8 bins.

        Returns:
          tone_idx: (B,) int64 in [0..7]
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        energy = (audio ** 2).mean(dim=1)
        # Normalize energy roughly to [0, 1] then scale to 0..7
        norm = torch.clamp(energy / (energy.mean() + 1e-8), 0.0, 2.0) / 2.0
        idx = (norm * 8.0).long()
        idx = torch.clamp(idx, 0, 7)
        return idx

    def audio_to_spike(
        self,
        audio: torch.Tensor
    ) -> Tuple[SpikePacket, int]:
        """
        Main entry: audio → (SpikePacket, tone_idx)

        Args:
          audio: (B, T) waveform (float32/float64)

        Returns:
          spike: SpikePacket with content in latent_dim
          tone_idx: int in [0..7] for the first element in batch
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        # 1) Extract features and project to latent
        feats = self._extract_features(audio)
        latent = self.proj(feats)

        # 2) Tone classification
        tone_vec = self._classify_tone(audio)
        tone_idx = int(tone_vec[0].item())

        # 3) Dummy glyph + f0 (placeholders)
        glyph_idx = 0
        f0 = 0.0

        # 4) Build metadata
        metadata: Dict[str, Any] = {
            "tone": tone_idx,
            "glyph_idx": glyph_idx,
            "f0": f0,
            "tags": ["stub_tonenet"],
        }

        spike = SpikePacket(
            content=latent,
            routing_key="audio_in",
            priority=1.0,
            modality="audio",
            metadata=metadata,
        )

        return spike, tone_idx

    def spike_to_audio(
        self,
        spike: SpikePacket,
        length: int = None
    ) -> torch.Tensor:
        """
        Stub reconstruction: returns silence.

        Args:
          spike: SpikePacket with latent content
          length: desired length T; default = 1 second at sample_rate

        Returns:
          audio: (B, T) waveform of zeros (for now).
        """
        if length is None:
            length = self.sample_rate

        B = spike.content.shape[0]
        return torch.zeros(B, length, device=self.device)
