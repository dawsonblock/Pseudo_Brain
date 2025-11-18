# -*- coding: utf-8 -*-
"""
Capsule Brain Demo
"""
import torch

from capsule_brain.config import DEFAULT_CONFIG
from capsule_brain.brain import CapsuleBrain


def run_demo(steps: int = 10):
    print("=" * 60)
    print("CAPSULE BRAIN DEMO")
    print("=" * 60)

    cfg = DEFAULT_CONFIG
    brain = CapsuleBrain(cfg)

    print("\nInitialized Capsule Brain:")
    summary = brain.get_summary()
    print(f"  Active modes: {summary['pmm']['active_modes']}")
    print(f"  PMM mass: {summary['pmm']['total_mass']:.6f}")

    for i in range(steps):
        audio = torch.randn(1, 48000) * (0.5 + 0.1 * i)
        result = brain.step(audio, timestamp=i * 0.1)

        print(f"\nStep {i + 1}:")
        print(f"  Dominant tone: {result['dominant_tone']}")
        print(f"  PMM novelty: {result['pmm_novelty']:.3f}")
        print(f"  Capsule outputs: {len(result['capsule_outputs'])}")

        for spike in result["capsule_outputs"]:
            meta = spike.metadata
            if meta.get("type") == "self_model_explanation":
                print(f"    Self-model: {meta['message']}")
            elif meta.get("type") == "safety_warning":
                for w in meta["warnings"]:
                    print(f"    SAFETY: {w}")

    print("\nFinal summary:")
    summary = brain.get_summary()
    print(f"  Steps: {summary['step_count']}")
    print(f"  Active PMM modes: {summary['pmm']['active_modes']}")
    print(f"  PMM mass: {summary['pmm']['total_mass']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
