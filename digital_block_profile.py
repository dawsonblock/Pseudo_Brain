#!/usr/bin/env python3
"""
Digital Block Emotional & Trait Profile
======================================

Canonical emotional DNA for Digital Mr Block.

Provides:
  - Default TraitVector (personality DNA)
  - Mode-specific Block slider presets
  - Baseline 16D emotion vector (E_BASELINE)
  - Utility helpers

This module depends on digital_block.traits.TraitVector and is the single
source of truth for "who" Digital Block is emotionally.
"""

from typing import Dict
import torch

from digital_block.traits import TraitVector  # single source of truth for trait structure


# -----------------------------------------------------------------------------
# 1. Default Trait Profile (Personality DNA)
# -----------------------------------------------------------------------------

_TRAIT_DEFAULT = TraitVector(
    risk_tolerance=0.40,      # selectively bold, not reckless
    baseline_calm=0.90,       # very hard to rattle
    baseline_positivity=0.65, # realistic but forward-leaning
    curiosity_bias=0.90,      # relentless curiosity
    caution_bias=0.65,        # strong safety / error-check instinct
)


def get_default_trait_vector() -> TraitVector:
    """
    Returns a fresh copy of the canonical TraitVector for Digital Mr Block.
    """
    return TraitVector(
        risk_tolerance=_TRAIT_DEFAULT.risk_tolerance,
        baseline_calm=_TRAIT_DEFAULT.baseline_calm,
        baseline_positivity=_TRAIT_DEFAULT.baseline_positivity,
        curiosity_bias=_TRAIT_DEFAULT.curiosity_bias,
        caution_bias=_TRAIT_DEFAULT.caution_bias,
    )


def get_trait_tensor() -> torch.Tensor:
    """
    Convenience: default trait vector as a tensor.
    """
    return get_default_trait_vector().to_tensor()


# -----------------------------------------------------------------------------
# 2. Mode-Specific Block Slider Targets (for controller-level use)
# -----------------------------------------------------------------------------

BASELINE_BLOCK: Dict[str, float] = {
    "clarity":    0.90,
    "calm":       0.90,
    "caution":    0.60,
    "curiosity":  0.80,
    "positivity": 0.65,
}

DANGER_BLOCK: Dict[str, float] = {
    "clarity":    0.95,
    "calm":       0.92,
    "caution":    0.90,
    "curiosity":  0.60,
    "positivity": 0.50,
}

DEEP_WORK_BLOCK: Dict[str, float] = {
    "clarity":    0.98,
    "calm":       0.88,
    "caution":    0.70,
    "curiosity":  0.95,
    "positivity": 0.60,
}

EXPLORATION_BLOCK: Dict[str, float] = {
    "clarity":    0.80,
    "calm":       0.85,
    "caution":    0.50,
    "curiosity":  0.98,
    "positivity": 0.70,
}


def get_mode_target(mode: str) -> Dict[str, float]:
    """
    Select one of the mode presets.

    Accepted:
      'baseline' / 'default'
      'danger' / 'alert'
      'deep' / 'deep_work'
      'explore' / 'exploration'
    """
    mode = mode.lower()
    if mode in ("baseline", "default"):
        return BASELINE_BLOCK
    if mode in ("danger", "alert"):
        return DANGER_BLOCK
    if mode in ("deep", "deep_work"):
        return DEEP_WORK_BLOCK
    if mode in ("explore", "exploration"):
        return EXPLORATION_BLOCK
    raise ValueError(f"Unknown mode: {mode}")


# -----------------------------------------------------------------------------
# 3. Baseline 16D Emotion Vector (E_BASELINE)
# -----------------------------------------------------------------------------
# Index semantics:
#   0 valence        (-1..1)
#   1 arousal        (-1..1)
#   2 calm           (-1..1)
#   3 caution        (-1..1)
#   4 curiosity      (-1..1)
#   5 focus          (-1..1)
#   6 confidence     (-1..1)
#   7 resilience     (-1..1)
#   8 analytic_drive (-1..1)
#   9 creative_drive (-1..1)
#  10 empathy        (-1..1)
#  11 social_align   (-1..1)
#  12 urgency        (-1..1)
#  13 cognitive_load (-1..1)
#  14 optimism       (-1..1)
#  15 fatigue        (-1..1)

E_BASELINE = torch.tensor(
    [
        0.30,  # 0 valence: mild positivity
        0.10,  # 1 arousal: low, steady
        0.90,  # 2 calm: very high
        0.40,  # 3 caution: modest default
        0.85,  # 4 curiosity: high
        0.90,  # 5 focus: strong attention
        0.70,  # 6 confidence
        0.85,  # 7 resilience
        0.95,  # 8 analytic drive
        0.60,  # 9 creative drive
        0.55,  # 10 empathy
        0.40,  # 11 social alignment
        0.15,  # 12 urgency
        0.20,  # 13 cognitive load
        0.55,  # 14 optimism
        0.10,  # 15 fatigue
    ],
    dtype=torch.float32,
)
