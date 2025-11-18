# -*- coding: utf-8 -*-
"""
GPU-Accelerated Harmonic Synthesis
Fallback to CPU if CUDA not available
"""
import torch
import torch.nn as nn
import numpy as np


class GPUHarmonicSynth:
    """
    GPU-accelerated harmonic additive synthesis with envelope shaping.
    Falls back to CPU if CUDA unavailable.
    """
    
    def __init__(self, sample_rate=48000, harmonics=16, device='cpu'):
        self.sr = sample_rate
        self.harmonics = harmonics
        self.device = device
        
        # Try to use CUDA if available
        if torch.cuda.is_available() and device == 'cuda':
            self.use_cuda = True
            try:
                # Try to load CUDA kernels (if compiled)
                from torch.utils.cpp_extension import load
                self.cuda_additive = load(
                    name="harmonic_additive",
                    sources=["tonenet/cuda_kernels/harmonic_additive.cu"],
                    verbose=False
                )
                self.cuda_available = True
            except Exception:
                self.cuda_available = False
        else:
            self.use_cuda = False
            self.cuda_available = False
    
    def synthesize(self, f0, H, phi, duration=1.0):
        """
        GPU-accelerated harmonic synthesis with envelope
        
        Args:
            f0:  (B) - fundamental frequency [Hz]
            H:   (B, K) - harmonic amplitudes
            phi: (B, K) - harmonic phases [radians]
            duration: float - duration in seconds
        
        Returns:
            waveform: (B, T) - synthesized audio
        """
        B = f0.shape[0]
        T = int(self.sr * duration)
        
        # Ensure tensors on correct device
        f0 = f0.to(self.device)
        H = H.to(self.device)
        phi = phi.to(self.device)
        
        if self.cuda_available and self.use_cuda:
            # Use CUDA kernel
            waveform = self._synthesize_cuda(f0, H, phi, T)
        else:
            # Use PyTorch (CPU or GPU)
            waveform = self._synthesize_pytorch(f0, H, phi, T)
        
        # Apply envelope
        waveform = self._apply_envelope(waveform, attack=0.02, decay=0.1)
        
        return waveform
    
    def _synthesize_cuda(self, f0, H, phi, T):
        """CUDA kernel synthesis (if available)"""
        B = f0.shape[0]
        waveform = torch.zeros((B, T), device=self.device)
        
        # Launch CUDA kernel
        self.cuda_additive.forward(
            f0.contiguous(),
            H.contiguous(),
            phi.contiguous(),
            waveform,
            B, self.harmonics, T
        )
        
        return waveform
    
    def _synthesize_pytorch(self, f0, H, phi, T):
        """
        PyTorch-based additive synthesis (CPU/GPU compatible)
        Vectorized for efficiency
        """
        B = f0.shape[0]
        K = self.harmonics
        
        # Time vector: (T,)
        t = torch.arange(T, device=self.device, dtype=torch.float32)
        
        # Harmonic numbers: (K,)
        k = torch.arange(1, K + 1, device=self.device, dtype=torch.float32)
        
        # Reshape for broadcasting: f0 (B,1,1), k (1,K,1), t (1,1,T)
        f0 = f0.view(B, 1, 1)
        k = k.view(1, K, 1)
        t = t.view(1, 1, T)
        H = H.view(B, K, 1)
        phi = phi.view(B, K, 1)
        
        # Compute ω = 2π * f_k / sr for each harmonic
        omega = 2 * 3.14159265359 * f0 * k / self.sr
        
        # Additive synthesis: sum_k H_k * cos(ω_k * t + φ_k)
        # Broadcasting: (B, K, T)
        harmonics = H * torch.cos(omega * t + phi)
        
        # Sum over harmonics: (B, T)
        waveform = harmonics.sum(dim=1)
        
        return waveform
    
    def _apply_envelope(self, waveform, attack=0.02, decay=0.1):
        """
        Apply ADSR-like envelope
        
        Args:
            waveform: (B, T)
            attack: attack time in seconds
            decay: decay time constant in seconds
        """
        B, T = waveform.shape
        
        # Time vector
        t = torch.arange(T, device=self.device, dtype=torch.float32)
        
        # Attack samples
        attack_samples = int(attack * self.sr)
        decay_samples = int(decay * self.sr)
        
        # Create envelope
        envelope = torch.ones(T, device=self.device)
        
        # Attack phase (linear ramp)
        if attack_samples > 0:
            envelope[:attack_samples] = t[:attack_samples] / attack_samples
        
        # Decay phase (exponential decay)
        if attack_samples < T:
            decay_t = t[attack_samples:] - attack_samples
            envelope[attack_samples:] = torch.exp(-decay_t / decay_samples)
        
        # Apply envelope (broadcast over batch)
        waveform = waveform * envelope.unsqueeze(0)
        
        return waveform
    
    def synthesize_sequence(self, f0_seq, H_seq, phi_seq):
        """
        Synthesize from parameter sequences with smooth transitions
        
        Args:
            f0_seq:  (B, T_seq) - f0 trajectory
            H_seq:   (B, T_seq, K) - harmonic amplitude trajectory
            phi_seq: (B, T_seq, K) - harmonic phase trajectory
        
        Returns:
            waveform: (B, T_audio) - synthesized audio
        """
        B, T_seq = f0_seq.shape
        frame_duration = 0.02  # 20ms frames
        
        # Synthesize each frame
        frames = []
        for i in range(T_seq):
            frame = self.synthesize(
                f0_seq[:, i],
                H_seq[:, i, :],
                phi_seq[:, i, :],
                duration=frame_duration
            )
            frames.append(frame)
        
        # Concatenate with overlap-add
        waveform = torch.cat(frames, dim=1)
        
        return waveform
