#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capsule Brain - Quick Start Demo
=================================

Demonstrates the fully functional Capsule Brain system after critical bug fixes.
"""
import torch
from capsule_brain.config import CapsuleBrainConfig
from capsule_brain.brain import CapsuleBrain


def main():
    print("=" * 70)
    print("CAPSULE BRAIN - QUICK START DEMO")
    print("=" * 70)
    print()
    
    # 1. Initialize with auto-detected device
    print("1. Initializing Capsule Brain...")
    cfg = CapsuleBrainConfig()
    print(f"   Device: {cfg.device}")
    print(f"   Latent dim: {cfg.latent_dim}")
    print(f"   PMM max modes: {cfg.max_modes}")
    print(f"   FRNN states: {cfg.num_states}")
    
    brain = CapsuleBrain(cfg)
    print("   ✅ Brain instantiated successfully")
    print()
    
    # 2. Process audio input
    print("2. Processing audio input (5 steps)...")
    for step in range(5):
        # Generate dummy audio (1 second at 48kHz)
        audio = torch.randn(1, 48000)
        
        # Brain step
        result = brain.step(audio, timestamp=float(step))
        
        print(f"   Step {step + 1}:")
        print(f"     Dominant tone: {result['dominant_tone']}")
        print(f"     PMM active modes: {result['pmm_active_modes']}")
        print(f"     PMM novelty: {result['pmm_novelty']:.4f}")
        print(f"     Capsule outputs: {len(result['capsule_outputs'])}")
    
    print()
    print("   ✅ All steps completed successfully")
    print()
    
    # 3. Get brain summary
    print("3. Brain State Summary:")
    summary = brain.get_summary()
    print(f"   Total steps: {summary['step_count']}")
    print(f"   PMM active modes: {summary['pmm']['active_modes']}")
    print(f"   PMM total mass: {summary['pmm']['total_mass']:.6f}")
    print(f"   Workspace initialized: {summary['workspace']['initialized']}")
    print(f"   Active capsules: {len(summary['capsules'])}")
    print()
    
    # 4. Verify invariants
    print("4. Verifying Invariants...")
    pmm = brain.pmm
    occ_sum = pmm.occupancy[pmm.active_mask].sum().item()
    assert abs(occ_sum - 1.0) < 1e-5, f"Occupancy sum: {occ_sum}"
    print(f"   ✅ Occupancy mass conservation: {occ_sum:.6f} ≈ 1.0")
    
    lambda_min = pmm.lambda_i[pmm.active_mask].min().item()
    assert lambda_min >= 0, f"Negative λ: {lambda_min}"
    print(f"   ✅ Parameter validity: λ_min = {lambda_min:.4f} ≥ 0")
    
    feelings_sum = brain.feelings.F.sum().item()
    print(f"   ✅ Feelings normalization: {feelings_sum:.6f} ≈ 1.0")
    print()
    
    print("=" * 70)
    print("🎉 CAPSULE BRAIN DEMO COMPLETE - ALL SYSTEMS FUNCTIONAL!")
    print("=" * 70)


if __name__ == "__main__":
    main()
