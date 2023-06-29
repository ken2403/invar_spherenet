from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


class SphericalHarmonics2dDirect(nn.Module):
    def __init__(self, max_l: int, normalization: Literal["integral", "component"] = "integral"):
        """
        Args:
            max_l (int): max degree of spherical harmonics (excluding l)
        """
        super().__init__()
        if max_l >= 11:
            raise NotImplementedError("spherical_harmonics maximum l implemented is 11, please use a lower value.")
        if normalization not in ["integral", "component"]:
            raise ValueError(f"normalization must be one of 'integral', 'component', got {normalization}")
        self.max_l = max_l
        self.normalization = normalization

    def extra_repr(self) -> str:
        return f"max_l={self.max_l}, normalization={self.normalization}"

    def forward(self, costheta: Tensor) -> Tensor:
        """Forward calculation of SphericalHarmonics2dDirect.

        Args:
            costheta (torch.Tensor): 2d angle with (*) shape.

        Returns:
            shb (torch.Tensor): 2d spherical harmonics basis with (*, max_l) shape.
        """  # noqa: E501
        # - PROFILER - with torch.autograd.profiler.record_function("spherical_harmonics_polynomial"):
        sh = _spherical_harmonics(self.max_l - 1, costheta)

        if self.normalization == "integral":
            sh.div_(math.sqrt(4 * math.pi))
        # elif self.normalization == "norm":
        #     sh.div_(
        #         torch.cat(
        #             [
        #                 math.sqrt(2 * lq + 1) * torch.ones(2 * lq + 1, dtype=sh.dtype, device=sh.device)
        #                 for lq in self.ls
        #             ]
        #         )
        #     )

        return sh


@torch.jit.script
def _spherical_harmonics(lmax: int, costheta: Tensor) -> Tensor:
    sh_0_0 = torch.ones_like(costheta)
    if lmax == 0:
        return torch.stack([sh_0_0], dim=-1)

    sh_1_0 = math.sqrt(3) * costheta
    if lmax == 1:
        return torch.stack([sh_0_0, sh_1_0], dim=-1)

    sh_2_0 = math.sqrt(5 / 4) * 3 * costheta.pow(2) - 1
    if lmax == 2:
        return torch.stack([sh_0_0, sh_1_0, sh_2_0], dim=-1)

    sh_3_0 = math.sqrt(7 / 4) * (5 * costheta.pow(3) - 3 * costheta)
    if lmax == 3:
        return torch.stack([sh_0_0, sh_1_0, sh_2_0, sh_3_0], dim=-1)

    sh_4_0 = (3 / 8) * (35 * costheta.pow(4) - 30 * costheta.pow(2) + 3)
    if lmax == 4:
        return torch.stack([sh_0_0, sh_1_0, sh_2_0, sh_3_0, sh_4_0], dim=-1)

    sh_5_0 = (1 / 8) * math.sqrt(11) * (63 * costheta.pow(5) - 70 * costheta.pow(3) + 15 * costheta)
    if lmax == 5:
        return torch.stack([sh_0_0, sh_1_0, sh_2_0, sh_3_0, sh_4_0, sh_5_0], dim=-1)

    sh_6_0 = (1 / 16) * math.sqrt(13) * (231 * costheta.pow(6) - 315 * costheta.pow(4) + 105 * costheta.pow(2) - 5)
    if lmax == 6:
        return torch.stack([sh_0_0, sh_1_0, sh_2_0, sh_3_0, sh_4_0, sh_5_0, sh_6_0], dim=-1)

    sh_7_0 = (
        (1 / 16)
        * math.sqrt(15)
        * (429 * costheta.pow(7) - 693 * costheta.pow(5) + 315 * costheta.pow(3) - 35 * costheta)
    )
    if lmax == 7:
        return torch.stack([sh_0_0, sh_1_0, sh_2_0, sh_3_0, sh_4_0, sh_5_0, sh_6_0, sh_7_0], dim=-1)

    sh_8_0 = (
        (1 / 128)
        * math.sqrt(17)
        * (6435 * costheta.pow(8) - 12012 * costheta.pow(6) + 6930 * costheta.pow(4) - 1260 * costheta.pow(2) + 35)
    )
    if lmax == 8:
        return torch.stack([sh_0_0, sh_1_0, sh_2_0, sh_3_0, sh_4_0, sh_5_0, sh_6_0, sh_7_0, sh_8_0], dim=-1)

    sh_9_0 = (
        (1 / 128)
        * math.sqrt(19)
        * (
            12155 * costheta.pow(9)
            - 25740 * costheta.pow(7)
            + 18018 * costheta.pow(5)
            - 4620 * costheta.pow(3)
            + 315 * costheta
        )
    )
    if lmax == 9:
        return torch.stack([sh_0_0, sh_1_0, sh_2_0, sh_3_0, sh_4_0, sh_5_0, sh_6_0, sh_7_0, sh_8_0, sh_9_0], dim=-1)

    sh_10_0 = (
        (1 / 256)
        * math.sqrt(21)
        * (
            46189 * costheta.pow(10)
            - 109395 * costheta.pow(8)
            + 90090 * costheta.pow(6)
            - 30030 * costheta.pow(4)
            + 3465 * costheta.pow(2)
            - 63
        )
    )
    return torch.stack(
        [sh_0_0, sh_1_0, sh_2_0, sh_3_0, sh_4_0, sh_5_0, sh_6_0, sh_7_0, sh_8_0, sh_9_0, sh_10_0], dim=-1
    )
