from __future__ import annotations

from abc import ABC, abstractmethod
from math import pi

import torch
import torch.nn as nn
from torch import Tensor


class BaseCutoff(ABC, nn.Module):
    """Layer to calculate radial cutoff function values."""

    def __init__(self, cutoff: float, **kwargs):
        """
        Args:
            cutoff (float): Cutoff radius.
        """
        super().__init__()
        self.cutoff = cutoff

    def extra_repr(self) -> str:
        return f"cutoff={self.cutoff}"

    @abstractmethod
    def cutoff_weight(self, r: Tensor) -> Tensor:
        return r

    def forward(self, r: Tensor) -> Tensor:
        return self.cutoff_weight(r)


class PolynomialCutoff(BaseCutoff):
    def __init__(self, cutoff: float):
        super().__init__(cutoff)

    def cutoff_weight(self, r: Tensor) -> Tensor:
        ratio = r / self.cutoff
        # Remove contributions beyond the cutoff radius
        return torch.where(
            r <= self.cutoff, 1 - 6 * ratio**5 + 15 * ratio**4 - 10 * ratio**3, torch.zeros_like(r)
        )


class CosineCutoff(BaseCutoff):
    def __init__(self, cutoff: float):
        super().__init__(cutoff)

    def cutoff_weight(self, r: Tensor) -> Tensor:
        cutoffs = 0.5 * (torch.cos(r * pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        return torch.where(r <= self.cutoff, cutoffs, torch.zeros_like(r))


class Envelope(torch.nn.Module):
    """Envelope function that ensures a smooth cutoff."""

    def __init__(self, cutoff: float, p: int = 5):
        """
        Args:
            cutoff (float): Cutoff radius.
            p (int, optional): Exponent of the envelope function. Defaults to `5`.
        """
        super().__init__()
        assert p > 0
        self.p = p
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def cutoff_weight(self, r_scaled: Tensor) -> Tensor:
        env_val = (
            1 + self.a * r_scaled**self.p + self.b * r_scaled ** (self.p + 1) + self.c * r_scaled ** (self.p + 2)
        )
        return torch.where(r_scaled <= 1, env_val, torch.zeros_like(r_scaled))
