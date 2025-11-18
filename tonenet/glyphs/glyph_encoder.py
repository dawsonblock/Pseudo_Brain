# -*- coding: utf-8 -*-
"""
ToneNet Glyph Encoder
Converts harmonic math (f0, H_k, φ_k) → discrete glyph indices
"""
import torch
import torch.nn as nn


class MathToSymbolEncoder(nn.Module):
    """
    Encodes harmonic math (f0, H_k, phi_k) into a discrete glyph.
    Output: logits over 128-glyph space + predicted glyph index.
    """

    def __init__(self, harmonics=16, glyph_count=128, device='cpu'):
        super().__init__()
        self.harmonics = harmonics
        self.glyph_count = glyph_count
        input_dim = 1 + harmonics * 2  # f0 + H + phi

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, glyph_count)
        ).to(device)
        
        self.device = device

    def forward(self, f0, H, phi):
        """
        Args:
            f0:  (B) - fundamental frequency
            H:   (B,K) - harmonic amplitudes
            phi: (B,K) - harmonic phases
        
        Returns:
            glyph_idx: (B) - predicted glyph indices
            logits: (B, glyph_count) - logits over glyph space
        """
        # Ensure inputs are on correct device
        f0 = f0.to(self.device)
        H = H.to(self.device)
        phi = phi.to(self.device)
        
        # Concatenate features
        x = torch.cat([f0.unsqueeze(1), H, phi], dim=1)
        
        # Forward pass
        logits = self.net(x)
        glyph_idx = torch.argmax(logits, dim=1)
        
        return glyph_idx, logits
    
    def encode_with_temperature(self, f0, H, phi, temperature=1.0):
        """Sample from glyph distribution with temperature"""
        _, logits = self.forward(f0, H, phi)
        probs = torch.softmax(logits / temperature, dim=-1)
        glyph_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
        return glyph_idx, probs
