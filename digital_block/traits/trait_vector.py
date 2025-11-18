# digital_block/traits/trait_vector.py

from dataclasses import dataclass
import torch


@dataclass
class TraitVector:
    """
    TraitVector defines the personality axes.
    Actual target values live in digital_block_profile.get_default_trait_vector().
    """
    risk_tolerance: float
    baseline_calm: float
    baseline_positivity: float
    curiosity_bias: float
    caution_bias: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.risk_tolerance,
                self.baseline_calm,
                self.baseline_positivity,
                self.curiosity_bias,
                self.caution_bias,
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def default() -> "TraitVector":
        """
        Neutral placeholder. For the true Digital Block profile,
        use digital_block_profile.get_default_trait_vector().
        """
        return TraitVector(
            risk_tolerance=0.5,
            baseline_calm=0.5,
            baseline_positivity=0.5,
            curiosity_bias=0.5,
            caution_bias=0.5,
        )
