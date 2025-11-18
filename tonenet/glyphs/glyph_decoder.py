# -*- coding: utf-8 -*-
"""
ToneNet Glyph Decoder
Converts discrete glyph indices → harmonic math (f0, H_k, φ_k)
"""
import torch
import torch.nn as nn


class SymbolToMathDecoder(nn.Module):
    """
    Converts glyph indices → harmonic math representation (f0, H_k, phi_k).
    """

    def __init__(
        self,
        glyph_count=128,
        embed_dim=256,
        harmonics=16,
        device='cpu'
    ):
        super().__init__()
        self.harmonics = harmonics
        self.glyph_count = glyph_count
        self.device = device

        # Glyph embedding layer
        self.embedding = nn.Embedding(glyph_count, embed_dim).to(device)

        output_dim = 1 + harmonics * 2  # f0, H_k, phi_k

        # Decoder network
        self.out = nn.Sequential(
            nn.Linear(embed_dim, 384),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        ).to(device)

    def forward(self, glyph_idx):
        """
        Args:
            glyph_idx: (B) - glyph indices
        
        Returns:
            f0:  (B) - fundamental frequency [0, 2000] Hz
            H:   (B, K) - harmonic amplitudes (non-negative)
            phi: (B, K) - harmonic phases [0, 2π]
        """
        # Ensure input is on correct device
        glyph_idx = glyph_idx.to(self.device)
        
        # Embed glyph
        e = self.embedding(glyph_idx)
        
        # Decode to harmonic parameters
        x = self.out(e)

        # f0 in 0–2000 Hz
        f0 = torch.sigmoid(x[:, 0]) * 2000.0

        # Harmonic amplitudes (non-negative)
        H = torch.relu(x[:, 1:1+self.harmonics])
        
        # Normalize amplitudes
        H = H / (H.sum(dim=1, keepdim=True) + 1e-8)

        # Harmonic phases [0, 2π]
        phi = torch.sigmoid(x[:, 1+self.harmonics:]) * 2 * 3.14159265359

        return f0, H, phi
    
    def decode_batch(self, glyph_sequence):
        """
        Decode sequence of glyphs
        
        Args:
            glyph_sequence: (B, T) - sequence of glyph indices
        
        Returns:
            f0_seq:  (B, T) - f0 trajectory
            H_seq:   (B, T, K) - harmonic amplitude trajectory
            phi_seq: (B, T, K) - harmonic phase trajectory
        """
        B, T = glyph_sequence.shape
        
        # Flatten for batch processing
        glyphs_flat = glyph_sequence.reshape(-1)
        
        # Decode all at once
        f0_flat, H_flat, phi_flat = self.forward(glyphs_flat)
        
        # Reshape back to sequence
        f0_seq = f0_flat.reshape(B, T)
        H_seq = H_flat.reshape(B, T, self.harmonics)
        phi_seq = phi_flat.reshape(B, T, self.harmonics)
        
        return f0_seq, H_seq, phi_seq
