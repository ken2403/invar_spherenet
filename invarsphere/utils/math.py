from __future__ import annotations

import torch
from torch import Tensor


def inner_product_normalized(x: Tensor, y: Tensor) -> Tensor:
    """Calculate the inner product between the given normalized vectors, giving
    a result between -1 and 1."""
    return torch.sum(x * y, dim=-1).clamp(min=-1, max=1)
