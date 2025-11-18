# -*- coding: utf-8 -*-
"""
Integration test for Capsule Brain
"""
import torch

from capsule_brain.config import DEFAULT_CONFIG
from capsule_brain.brain import CapsuleBrain


def test_full_pipeline():
    cfg = DEFAULT_CONFIG
    brain = CapsuleBrain(cfg)

    audio = torch.randn(1, 48000)  # 1 second at 48kHz

    result = brain.step(audio, timestamp=0.0)

    # PMM invariants
    occ = brain.pmm.occupancy[brain.pmm.active_mask]
    assert abs(occ.sum().item() - 1.0) < 1e-6, "PMM mass conservation violated"
    assert (occ >= 0).all(), "PMM occupancy negative"

    # Feelings
    F = brain.feelings.to_tensor()
    assert F.shape == (1, 8)
    assert torch.all(F >= 0)
    assert abs(float(F.sum().item()) - 1.0) < 1e-6

    # Workspace
    ws = result["workspace_state"]
    assert ws.broadcast.shape == (1, cfg.latent_dim)

    print("✓ Integration test passed.")
    print(f"  Novelty: {result['pmm_novelty']:.3f}")
    print(f"  Dominant tone: {result['dominant_tone']}")
    print(f"  Capsule outputs: {len(result['capsule_outputs'])}")


if __name__ == "__main__":
    test_full_pipeline()
