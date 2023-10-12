from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..utils.repeat_tensor import block_repeat
from .basis import SphericalBesselFunction


class SphericalHarmonicsWithBesselDirect(nn.Module):
    """Combine spherical Harmonics function and spherical Bessel fucntion."""

    def __init__(
        self,
        max_n: int,
        max_l: int,
        cutoff: float,
        cutoff_net: nn.Module | None,
        use_phi: bool,
        efficient: bool = False,
    ):
        """
        Args:
            max_n (int): max number of roots used in each l
            max_l (int): max degree of spherical harmonics (excluding l)
            cutoff (float): cutoff radius
            cutoff_net (torch.nn.Module | None): The torch.nn.Module of cutoff function
            use_phi (bool): whether to use the polar angle. If not, the function will compute `Y_l^0` only
            efficient (bool): whether to use the efficient bilinear.
        """
        super().__init__()
        self.max_n = max_n
        self.max_l = max_l
        self.cutoff = cutoff
        self.use_phi = use_phi
        self.efficient = efficient

        self.sbb = SphericalBesselFunction(max_n, max_l, cutoff, cutoff_net, smooth=False)
        if use_phi:
            self.shb: nn.Module = SphericalHarmonicsDirect(max_l, normalization="integral")
        else:
            self.shb = SphericalHarmonics2dDirect(max_l, normalization="integral")

    def combine_sbb_shb(self, sbb: Tensor, shb: Tensor):
        """Combine the spherical Bessel basis and the spherical Harmonics
        basis. For the spherical Bessel function, the column is ordered by.

            [n=[0, ..., max_n-1], n=[0, ..., max_n-1], ...], max_l blocks,

        For the spherical Harmonics function, the column is ordered by
            [m=[0], m=[-1, 0, 1], m=[-2, -1, 0, 1, 2], ...] max_l blocks, and each block has 2*l + 1
            if use_phi is False, then the columns become
            [m=[0], m=[0], ...] max_l columns

        Args:
            sbb (Tensor): spherical bessel basis results with (*, max_n*max_l) shape.
            shb (Tensor): spherical harmonics basis results with (*, max_l) shape if not use_phi, (*, max_l*max_l) shape if use_phi.

        Returns:
            combined_sbf_shf (Tensor): combination of spherical bessel and spherical harmonics.
        """  # noqa: E501
        if sbb.size(0) == 0:
            return sbb

        device = sbb.device
        if not self.use_phi:
            repeats_sbb = torch.tensor([1] * self.max_l * self.max_n, device=device)
            block_size = np.array([1] * self.max_l)
        else:
            # [1, 1, 1, ..., 1, 3, 3, 3, ..., 3, ... 5, ..., 2*max_l-1,...]
            repeats_sbb = torch.tensor(np.repeat(2 * np.arange(self.max_l) + 1, repeats=self.max_n), device=device)
            block_size = 2 * np.arange(self.max_l) + 1
        expanded_sbb = torch.repeat_interleave(sbb, repeats=repeats_sbb, dim=-1)
        expanded_shb = block_repeat(shb, block_size=block_size, repeats=np.array([self.max_n] * self.max_l), dim=1)
        shape = self.max_n * self.max_l
        if self.use_phi:
            shape *= self.max_l
        return (expanded_sbb * expanded_shb).view(-1, shape)

    def forward(
        self,
        r: Tensor,
        costheta: Tensor | None = None,
        vec: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            r (torch.Tensor): distance Tensor with (*) shape.
            costheta (torch.Tensor | None): the azimuthal angle cosine values with (*) shape.
            vec (torch.Tensor | None): the normalized neighbor vector with (*) shape.
        """
        sbb = self.sbb(r)
        if self.use_phi:
            assert vec is not None
            shb = self.shb(vec)
        else:
            assert costheta is not None
            shb = self.shb(costheta)

        if not self.efficient:
            combined_basis = self.combine_sbb_shb(sbb, shb)
            return combined_basis  # (*, max_n*max_l*max_l) or (*, max_n*max_l)

        # sbb: (1, *, max_n*max_l)
        # shb: (*, max_l) or (*, max_l*max_l)
        return sbb[None, :, :], shb


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
        """
        # - PROFILER - with torch.autograd.profiler.record_function("spherical_harmonics_2d"):
        sh = _spherical_harmonics_2d(self.max_l - 1, costheta)

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


class SphericalHarmonicsDirect(nn.Module):
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

    def forward(self, x: Tensor) -> Tensor:
        """Forward calculation of SphericalHarmonicsDirect.

        Args:
            x (torch.Tensor): normalized edge vector with (*, 3) shape.

        Returns:
            shb (torch.Tensor): spherical harmonics basis with (*, max_l*max_l) shape.
        """
        # - PROFILER - with torch.autograd.profiler.record_function("spherical_harmonics"):
        sh = _spherical_harmonics(self.max_l - 1, x[..., 0], x[..., 1], x[..., 2])

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
def _spherical_harmonics_2d(lmax: int, costheta: Tensor) -> Tensor:
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


@torch.jit.script
def _spherical_harmonics(lmax: int, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    sh_0_0 = torch.ones_like(x)
    if lmax == 0:
        return torch.stack(
            [
                sh_0_0,
            ],
            dim=-1,
        )

    sh_1_0 = math.sqrt(3) * x
    sh_1_1 = math.sqrt(3) * y
    sh_1_2 = math.sqrt(3) * z
    if lmax == 1:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], dim=-1)

    sh_2_0 = math.sqrt(15) * x * z
    sh_2_1 = math.sqrt(15) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = math.sqrt(5) * (y2 - (1 / 2) * x2z2)
    sh_2_3 = math.sqrt(15) * y * z
    sh_2_4 = (1 / 2) * math.sqrt(15) * (z.pow(2) - x.pow(2))

    if lmax == 2:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)

    sh_3_0 = (1 / 6) * math.sqrt(42) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = math.sqrt(7) * sh_2_0 * y
    sh_3_2 = (1 / 8) * math.sqrt(168) * (4.0 * y2 - x2z2) * x
    sh_3_3 = (1 / 2) * math.sqrt(7) * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = (1 / 8) * math.sqrt(168) * z * (4.0 * y2 - x2z2)
    sh_3_5 = math.sqrt(7) * sh_2_4 * y
    sh_3_6 = (1 / 6) * math.sqrt(42) * (sh_2_4 * z - sh_2_0 * x)

    if lmax == 3:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
            ],
            dim=-1,
        )

    sh_4_0 = (3 / 4) * math.sqrt(2) * (sh_3_0 * z + sh_3_6 * x)
    sh_4_1 = (3 / 4) * sh_3_0 * y + (3 / 8) * math.sqrt(6) * sh_3_1 * z + (3 / 8) * math.sqrt(6) * sh_3_5 * x
    sh_4_2 = (
        -3 / 56 * math.sqrt(14) * sh_3_0 * z
        + (3 / 14) * math.sqrt(21) * sh_3_1 * y
        + (3 / 56) * math.sqrt(210) * sh_3_2 * z
        + (3 / 56) * math.sqrt(210) * sh_3_4 * x
        + (3 / 56) * math.sqrt(14) * sh_3_6 * x
    )
    sh_4_3 = (
        -3 / 56 * math.sqrt(42) * sh_3_1 * z
        + (3 / 28) * math.sqrt(105) * sh_3_2 * y
        + (3 / 28) * math.sqrt(70) * sh_3_3 * x
        + (3 / 56) * math.sqrt(42) * sh_3_5 * x
    )
    sh_4_4 = (
        -3 / 28 * math.sqrt(42) * sh_3_2 * x + (3 / 7) * math.sqrt(7) * sh_3_3 * y - 3 / 28 * math.sqrt(42) * sh_3_4 * z
    )
    sh_4_5 = (
        -3 / 56 * math.sqrt(42) * sh_3_1 * x
        + (3 / 28) * math.sqrt(70) * sh_3_3 * z
        + (3 / 28) * math.sqrt(105) * sh_3_4 * y
        - 3 / 56 * math.sqrt(42) * sh_3_5 * z
    )
    sh_4_6 = (
        -3 / 56 * math.sqrt(14) * sh_3_0 * x
        - 3 / 56 * math.sqrt(210) * sh_3_2 * x
        + (3 / 56) * math.sqrt(210) * sh_3_4 * z
        + (3 / 14) * math.sqrt(21) * sh_3_5 * y
        - 3 / 56 * math.sqrt(14) * sh_3_6 * z
    )
    sh_4_7 = -3 / 8 * math.sqrt(6) * sh_3_1 * x + (3 / 8) * math.sqrt(6) * sh_3_5 * z + (3 / 4) * sh_3_6 * y
    sh_4_8 = (3 / 4) * math.sqrt(2) * (-sh_3_0 * x + sh_3_6 * z)
    if lmax == 4:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
                sh_4_0,
                sh_4_1,
                sh_4_2,
                sh_4_3,
                sh_4_4,
                sh_4_5,
                sh_4_6,
                sh_4_7,
                sh_4_8,
            ],
            dim=-1,
        )

    sh_5_0 = (1 / 10) * math.sqrt(110) * (sh_4_0 * z + sh_4_8 * x)
    sh_5_1 = (
        (1 / 5) * math.sqrt(11) * sh_4_0 * y
        + (1 / 5) * math.sqrt(22) * sh_4_1 * z
        + (1 / 5) * math.sqrt(22) * sh_4_7 * x
    )
    sh_5_2 = (
        -1 / 30 * math.sqrt(22) * sh_4_0 * z
        + (4 / 15) * math.sqrt(11) * sh_4_1 * y
        + (1 / 15) * math.sqrt(154) * sh_4_2 * z
        + (1 / 15) * math.sqrt(154) * sh_4_6 * x
        + (1 / 30) * math.sqrt(22) * sh_4_8 * x
    )
    sh_5_3 = (
        -1 / 30 * math.sqrt(66) * sh_4_1 * z
        + (1 / 15) * math.sqrt(231) * sh_4_2 * y
        + (1 / 30) * math.sqrt(462) * sh_4_3 * z
        + (1 / 30) * math.sqrt(462) * sh_4_5 * x
        + (1 / 30) * math.sqrt(66) * sh_4_7 * x
    )
    sh_5_4 = (
        -1 / 15 * math.sqrt(33) * sh_4_2 * z
        + (2 / 15) * math.sqrt(66) * sh_4_3 * y
        + (1 / 15) * math.sqrt(165) * sh_4_4 * x
        + (1 / 15) * math.sqrt(33) * sh_4_6 * x
    )
    sh_5_5 = (
        -1 / 15 * math.sqrt(110) * sh_4_3 * x
        + (1 / 3) * math.sqrt(11) * sh_4_4 * y
        - 1 / 15 * math.sqrt(110) * sh_4_5 * z
    )
    sh_5_6 = (
        -1 / 15 * math.sqrt(33) * sh_4_2 * x
        + (1 / 15) * math.sqrt(165) * sh_4_4 * z
        + (2 / 15) * math.sqrt(66) * sh_4_5 * y
        - 1 / 15 * math.sqrt(33) * sh_4_6 * z
    )
    sh_5_7 = (
        -1 / 30 * math.sqrt(66) * sh_4_1 * x
        - 1 / 30 * math.sqrt(462) * sh_4_3 * x
        + (1 / 30) * math.sqrt(462) * sh_4_5 * z
        + (1 / 15) * math.sqrt(231) * sh_4_6 * y
        - 1 / 30 * math.sqrt(66) * sh_4_7 * z
    )
    sh_5_8 = (
        -1 / 30 * math.sqrt(22) * sh_4_0 * x
        - 1 / 15 * math.sqrt(154) * sh_4_2 * x
        + (1 / 15) * math.sqrt(154) * sh_4_6 * z
        + (4 / 15) * math.sqrt(11) * sh_4_7 * y
        - 1 / 30 * math.sqrt(22) * sh_4_8 * z
    )
    sh_5_9 = (
        -1 / 5 * math.sqrt(22) * sh_4_1 * x
        + (1 / 5) * math.sqrt(22) * sh_4_7 * z
        + (1 / 5) * math.sqrt(11) * sh_4_8 * y
    )
    sh_5_10 = (1 / 10) * math.sqrt(110) * (-sh_4_0 * x + sh_4_8 * z)
    if lmax == 5:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
                sh_4_0,
                sh_4_1,
                sh_4_2,
                sh_4_3,
                sh_4_4,
                sh_4_5,
                sh_4_6,
                sh_4_7,
                sh_4_8,
                sh_5_0,
                sh_5_1,
                sh_5_2,
                sh_5_3,
                sh_5_4,
                sh_5_5,
                sh_5_6,
                sh_5_7,
                sh_5_8,
                sh_5_9,
                sh_5_10,
            ],
            dim=-1,
        )

    sh_6_0 = (1 / 6) * math.sqrt(39) * (sh_5_0 * z + sh_5_10 * x)
    sh_6_1 = (
        (1 / 6) * math.sqrt(13) * sh_5_0 * y
        + (1 / 12) * math.sqrt(130) * sh_5_1 * z
        + (1 / 12) * math.sqrt(130) * sh_5_9 * x
    )
    sh_6_2 = (
        -1 / 132 * math.sqrt(286) * sh_5_0 * z
        + (1 / 33) * math.sqrt(715) * sh_5_1 * y
        + (1 / 132) * math.sqrt(286) * sh_5_10 * x
        + (1 / 44) * math.sqrt(1430) * sh_5_2 * z
        + (1 / 44) * math.sqrt(1430) * sh_5_8 * x
    )
    sh_6_3 = (
        -1 / 132 * math.sqrt(858) * sh_5_1 * z
        + (1 / 22) * math.sqrt(429) * sh_5_2 * y
        + (1 / 22) * math.sqrt(286) * sh_5_3 * z
        + (1 / 22) * math.sqrt(286) * sh_5_7 * x
        + (1 / 132) * math.sqrt(858) * sh_5_9 * x
    )
    sh_6_4 = (
        -1 / 66 * math.sqrt(429) * sh_5_2 * z
        + (2 / 33) * math.sqrt(286) * sh_5_3 * y
        + (1 / 66) * math.sqrt(2002) * sh_5_4 * z
        + (1 / 66) * math.sqrt(2002) * sh_5_6 * x
        + (1 / 66) * math.sqrt(429) * sh_5_8 * x
    )
    sh_6_5 = (
        -1 / 66 * math.sqrt(715) * sh_5_3 * z
        + (1 / 66) * math.sqrt(5005) * sh_5_4 * y
        + (1 / 66) * math.sqrt(3003) * sh_5_5 * x
        + (1 / 66) * math.sqrt(715) * sh_5_7 * x
    )
    sh_6_6 = (
        -1 / 66 * math.sqrt(2145) * sh_5_4 * x
        + (1 / 11) * math.sqrt(143) * sh_5_5 * y
        - 1 / 66 * math.sqrt(2145) * sh_5_6 * z
    )
    sh_6_7 = (
        -1 / 66 * math.sqrt(715) * sh_5_3 * x
        + (1 / 66) * math.sqrt(3003) * sh_5_5 * z
        + (1 / 66) * math.sqrt(5005) * sh_5_6 * y
        - 1 / 66 * math.sqrt(715) * sh_5_7 * z
    )
    sh_6_8 = (
        -1 / 66 * math.sqrt(429) * sh_5_2 * x
        - 1 / 66 * math.sqrt(2002) * sh_5_4 * x
        + (1 / 66) * math.sqrt(2002) * sh_5_6 * z
        + (2 / 33) * math.sqrt(286) * sh_5_7 * y
        - 1 / 66 * math.sqrt(429) * sh_5_8 * z
    )
    sh_6_9 = (
        -1 / 132 * math.sqrt(858) * sh_5_1 * x
        - 1 / 22 * math.sqrt(286) * sh_5_3 * x
        + (1 / 22) * math.sqrt(286) * sh_5_7 * z
        + (1 / 22) * math.sqrt(429) * sh_5_8 * y
        - 1 / 132 * math.sqrt(858) * sh_5_9 * z
    )
    sh_6_10 = (
        -1 / 132 * math.sqrt(286) * sh_5_0 * x
        - 1 / 132 * math.sqrt(286) * sh_5_10 * z
        - 1 / 44 * math.sqrt(1430) * sh_5_2 * x
        + (1 / 44) * math.sqrt(1430) * sh_5_8 * z
        + (1 / 33) * math.sqrt(715) * sh_5_9 * y
    )
    sh_6_11 = (
        -1 / 12 * math.sqrt(130) * sh_5_1 * x
        + (1 / 6) * math.sqrt(13) * sh_5_10 * y
        + (1 / 12) * math.sqrt(130) * sh_5_9 * z
    )
    sh_6_12 = (1 / 6) * math.sqrt(39) * (-sh_5_0 * x + sh_5_10 * z)
    if lmax == 6:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
                sh_4_0,
                sh_4_1,
                sh_4_2,
                sh_4_3,
                sh_4_4,
                sh_4_5,
                sh_4_6,
                sh_4_7,
                sh_4_8,
                sh_5_0,
                sh_5_1,
                sh_5_2,
                sh_5_3,
                sh_5_4,
                sh_5_5,
                sh_5_6,
                sh_5_7,
                sh_5_8,
                sh_5_9,
                sh_5_10,
                sh_6_0,
                sh_6_1,
                sh_6_2,
                sh_6_3,
                sh_6_4,
                sh_6_5,
                sh_6_6,
                sh_6_7,
                sh_6_8,
                sh_6_9,
                sh_6_10,
                sh_6_11,
                sh_6_12,
            ],
            dim=-1,
        )

    sh_7_0 = (1 / 14) * math.sqrt(210) * (sh_6_0 * z + sh_6_12 * x)
    sh_7_1 = (
        (1 / 7) * math.sqrt(15) * sh_6_0 * y
        + (3 / 7) * math.sqrt(5) * sh_6_1 * z
        + (3 / 7) * math.sqrt(5) * sh_6_11 * x
    )
    sh_7_2 = (
        -1 / 182 * math.sqrt(390) * sh_6_0 * z
        + (6 / 91) * math.sqrt(130) * sh_6_1 * y
        + (3 / 91) * math.sqrt(715) * sh_6_10 * x
        + (1 / 182) * math.sqrt(390) * sh_6_12 * x
        + (3 / 91) * math.sqrt(715) * sh_6_2 * z
    )
    sh_7_3 = (
        -3 / 182 * math.sqrt(130) * sh_6_1 * z
        + (3 / 182) * math.sqrt(130) * sh_6_11 * x
        + (3 / 91) * math.sqrt(715) * sh_6_2 * y
        + (5 / 182) * math.sqrt(858) * sh_6_3 * z
        + (5 / 182) * math.sqrt(858) * sh_6_9 * x
    )
    sh_7_4 = (
        (3 / 91) * math.sqrt(65) * sh_6_10 * x
        - 3 / 91 * math.sqrt(65) * sh_6_2 * z
        + (10 / 91) * math.sqrt(78) * sh_6_3 * y
        + (15 / 182) * math.sqrt(78) * sh_6_4 * z
        + (15 / 182) * math.sqrt(78) * sh_6_8 * x
    )
    sh_7_5 = (
        -5 / 91 * math.sqrt(39) * sh_6_3 * z
        + (15 / 91) * math.sqrt(39) * sh_6_4 * y
        + (3 / 91) * math.sqrt(390) * sh_6_5 * z
        + (3 / 91) * math.sqrt(390) * sh_6_7 * x
        + (5 / 91) * math.sqrt(39) * sh_6_9 * x
    )
    sh_7_6 = (
        -15 / 182 * math.sqrt(26) * sh_6_4 * z
        + (12 / 91) * math.sqrt(65) * sh_6_5 * y
        + (2 / 91) * math.sqrt(1365) * sh_6_6 * x
        + (15 / 182) * math.sqrt(26) * sh_6_8 * x
    )
    sh_7_7 = (
        -3 / 91 * math.sqrt(455) * sh_6_5 * x
        + (1 / 13) * math.sqrt(195) * sh_6_6 * y
        - 3 / 91 * math.sqrt(455) * sh_6_7 * z
    )
    sh_7_8 = (
        -15 / 182 * math.sqrt(26) * sh_6_4 * x
        + (2 / 91) * math.sqrt(1365) * sh_6_6 * z
        + (12 / 91) * math.sqrt(65) * sh_6_7 * y
        - 15 / 182 * math.sqrt(26) * sh_6_8 * z
    )
    sh_7_9 = (
        -5 / 91 * math.sqrt(39) * sh_6_3 * x
        - 3 / 91 * math.sqrt(390) * sh_6_5 * x
        + (3 / 91) * math.sqrt(390) * sh_6_7 * z
        + (15 / 91) * math.sqrt(39) * sh_6_8 * y
        - 5 / 91 * math.sqrt(39) * sh_6_9 * z
    )
    sh_7_10 = (
        -3 / 91 * math.sqrt(65) * sh_6_10 * z
        - 3 / 91 * math.sqrt(65) * sh_6_2 * x
        - 15 / 182 * math.sqrt(78) * sh_6_4 * x
        + (15 / 182) * math.sqrt(78) * sh_6_8 * z
        + (10 / 91) * math.sqrt(78) * sh_6_9 * y
    )
    sh_7_11 = (
        -3 / 182 * math.sqrt(130) * sh_6_1 * x
        + (3 / 91) * math.sqrt(715) * sh_6_10 * y
        - 3 / 182 * math.sqrt(130) * sh_6_11 * z
        - 5 / 182 * math.sqrt(858) * sh_6_3 * x
        + (5 / 182) * math.sqrt(858) * sh_6_9 * z
    )
    sh_7_12 = (
        -1 / 182 * math.sqrt(390) * sh_6_0 * x
        + (3 / 91) * math.sqrt(715) * sh_6_10 * z
        + (6 / 91) * math.sqrt(130) * sh_6_11 * y
        - 1 / 182 * math.sqrt(390) * sh_6_12 * z
        - 3 / 91 * math.sqrt(715) * sh_6_2 * x
    )
    sh_7_13 = (
        -3 / 7 * math.sqrt(5) * sh_6_1 * x
        + (3 / 7) * math.sqrt(5) * sh_6_11 * z
        + (1 / 7) * math.sqrt(15) * sh_6_12 * y
    )
    sh_7_14 = (1 / 14) * math.sqrt(210) * (-sh_6_0 * x + sh_6_12 * z)
    if lmax == 7:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
                sh_4_0,
                sh_4_1,
                sh_4_2,
                sh_4_3,
                sh_4_4,
                sh_4_5,
                sh_4_6,
                sh_4_7,
                sh_4_8,
                sh_5_0,
                sh_5_1,
                sh_5_2,
                sh_5_3,
                sh_5_4,
                sh_5_5,
                sh_5_6,
                sh_5_7,
                sh_5_8,
                sh_5_9,
                sh_5_10,
                sh_6_0,
                sh_6_1,
                sh_6_2,
                sh_6_3,
                sh_6_4,
                sh_6_5,
                sh_6_6,
                sh_6_7,
                sh_6_8,
                sh_6_9,
                sh_6_10,
                sh_6_11,
                sh_6_12,
                sh_7_0,
                sh_7_1,
                sh_7_2,
                sh_7_3,
                sh_7_4,
                sh_7_5,
                sh_7_6,
                sh_7_7,
                sh_7_8,
                sh_7_9,
                sh_7_10,
                sh_7_11,
                sh_7_12,
                sh_7_13,
                sh_7_14,
            ],
            dim=-1,
        )

    sh_8_0 = (1 / 4) * math.sqrt(17) * (sh_7_0 * z + sh_7_14 * x)
    sh_8_1 = (
        (1 / 8) * math.sqrt(17) * sh_7_0 * y
        + (1 / 16) * math.sqrt(238) * sh_7_1 * z
        + (1 / 16) * math.sqrt(238) * sh_7_13 * x
    )
    sh_8_2 = (
        -1 / 240 * math.sqrt(510) * sh_7_0 * z
        + (1 / 60) * math.sqrt(1785) * sh_7_1 * y
        + (1 / 240) * math.sqrt(46410) * sh_7_12 * x
        + (1 / 240) * math.sqrt(510) * sh_7_14 * x
        + (1 / 240) * math.sqrt(46410) * sh_7_2 * z
    )
    sh_8_3 = (
        (1 / 80)
        * math.sqrt(2)
        * (
            -math.sqrt(85) * sh_7_1 * z
            + math.sqrt(2210) * sh_7_11 * x
            + math.sqrt(85) * sh_7_13 * x
            + math.sqrt(2210) * sh_7_2 * y
            + math.sqrt(2210) * sh_7_3 * z
        )
    )
    sh_8_4 = (
        (1 / 40) * math.sqrt(935) * sh_7_10 * x
        + (1 / 40) * math.sqrt(85) * sh_7_12 * x
        - 1 / 40 * math.sqrt(85) * sh_7_2 * z
        + (1 / 10) * math.sqrt(85) * sh_7_3 * y
        + (1 / 40) * math.sqrt(935) * sh_7_4 * z
    )
    sh_8_5 = (
        (1 / 48)
        * math.sqrt(2)
        * (
            math.sqrt(102) * sh_7_11 * x
            - math.sqrt(102) * sh_7_3 * z
            + math.sqrt(1122) * sh_7_4 * y
            + math.sqrt(561) * sh_7_5 * z
            + math.sqrt(561) * sh_7_9 * x
        )
    )
    sh_8_6 = (
        (1 / 16) * math.sqrt(34) * sh_7_10 * x
        - 1 / 16 * math.sqrt(34) * sh_7_4 * z
        + (1 / 4) * math.sqrt(17) * sh_7_5 * y
        + (1 / 16) * math.sqrt(102) * sh_7_6 * z
        + (1 / 16) * math.sqrt(102) * sh_7_8 * x
    )
    sh_8_7 = (
        -1 / 80 * math.sqrt(1190) * sh_7_5 * z
        + (1 / 40) * math.sqrt(1785) * sh_7_6 * y
        + (1 / 20) * math.sqrt(255) * sh_7_7 * x
        + (1 / 80) * math.sqrt(1190) * sh_7_9 * x
    )
    sh_8_8 = (
        -1 / 60 * math.sqrt(1785) * sh_7_6 * x
        + (1 / 15) * math.sqrt(255) * sh_7_7 * y
        - 1 / 60 * math.sqrt(1785) * sh_7_8 * z
    )
    sh_8_9 = (
        -1 / 80 * math.sqrt(1190) * sh_7_5 * x
        + (1 / 20) * math.sqrt(255) * sh_7_7 * z
        + (1 / 40) * math.sqrt(1785) * sh_7_8 * y
        - 1 / 80 * math.sqrt(1190) * sh_7_9 * z
    )
    sh_8_10 = (
        -1 / 16 * math.sqrt(34) * sh_7_10 * z
        - 1 / 16 * math.sqrt(34) * sh_7_4 * x
        - 1 / 16 * math.sqrt(102) * sh_7_6 * x
        + (1 / 16) * math.sqrt(102) * sh_7_8 * z
        + (1 / 4) * math.sqrt(17) * sh_7_9 * y
    )
    sh_8_11 = (
        (1 / 48)
        * math.sqrt(2)
        * (
            math.sqrt(1122) * sh_7_10 * y
            - math.sqrt(102) * sh_7_11 * z
            - math.sqrt(102) * sh_7_3 * x
            - math.sqrt(561) * sh_7_5 * x
            + math.sqrt(561) * sh_7_9 * z
        )
    )
    sh_8_12 = (
        (1 / 40) * math.sqrt(935) * sh_7_10 * z
        + (1 / 10) * math.sqrt(85) * sh_7_11 * y
        - 1 / 40 * math.sqrt(85) * sh_7_12 * z
        - 1 / 40 * math.sqrt(85) * sh_7_2 * x
        - 1 / 40 * math.sqrt(935) * sh_7_4 * x
    )
    sh_8_13 = (
        (1 / 80)
        * math.sqrt(2)
        * (
            -math.sqrt(85) * sh_7_1 * x
            + math.sqrt(2210) * sh_7_11 * z
            + math.sqrt(2210) * sh_7_12 * y
            - math.sqrt(85) * sh_7_13 * z
            - math.sqrt(2210) * sh_7_3 * x
        )
    )
    sh_8_14 = (
        -1 / 240 * math.sqrt(510) * sh_7_0 * x
        + (1 / 240) * math.sqrt(46410) * sh_7_12 * z
        + (1 / 60) * math.sqrt(1785) * sh_7_13 * y
        - 1 / 240 * math.sqrt(510) * sh_7_14 * z
        - 1 / 240 * math.sqrt(46410) * sh_7_2 * x
    )
    sh_8_15 = (
        -1 / 16 * math.sqrt(238) * sh_7_1 * x
        + (1 / 16) * math.sqrt(238) * sh_7_13 * z
        + (1 / 8) * math.sqrt(17) * sh_7_14 * y
    )
    sh_8_16 = (1 / 4) * math.sqrt(17) * (-sh_7_0 * x + sh_7_14 * z)
    if lmax == 8:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
                sh_4_0,
                sh_4_1,
                sh_4_2,
                sh_4_3,
                sh_4_4,
                sh_4_5,
                sh_4_6,
                sh_4_7,
                sh_4_8,
                sh_5_0,
                sh_5_1,
                sh_5_2,
                sh_5_3,
                sh_5_4,
                sh_5_5,
                sh_5_6,
                sh_5_7,
                sh_5_8,
                sh_5_9,
                sh_5_10,
                sh_6_0,
                sh_6_1,
                sh_6_2,
                sh_6_3,
                sh_6_4,
                sh_6_5,
                sh_6_6,
                sh_6_7,
                sh_6_8,
                sh_6_9,
                sh_6_10,
                sh_6_11,
                sh_6_12,
                sh_7_0,
                sh_7_1,
                sh_7_2,
                sh_7_3,
                sh_7_4,
                sh_7_5,
                sh_7_6,
                sh_7_7,
                sh_7_8,
                sh_7_9,
                sh_7_10,
                sh_7_11,
                sh_7_12,
                sh_7_13,
                sh_7_14,
                sh_8_0,
                sh_8_1,
                sh_8_2,
                sh_8_3,
                sh_8_4,
                sh_8_5,
                sh_8_6,
                sh_8_7,
                sh_8_8,
                sh_8_9,
                sh_8_10,
                sh_8_11,
                sh_8_12,
                sh_8_13,
                sh_8_14,
                sh_8_15,
                sh_8_16,
            ],
            dim=-1,
        )

    sh_9_0 = (1 / 6) * math.sqrt(38) * (sh_8_0 * z + sh_8_16 * x)
    sh_9_1 = (1 / 9) * math.sqrt(19) * (sh_8_0 * y + 2 * sh_8_1 * z + 2 * sh_8_15 * x)
    sh_9_2 = (
        -1 / 306 * math.sqrt(646) * sh_8_0 * z
        + (4 / 153) * math.sqrt(646) * sh_8_1 * y
        + (2 / 153) * math.sqrt(4845) * sh_8_14 * x
        + (1 / 306) * math.sqrt(646) * sh_8_16 * x
        + (2 / 153) * math.sqrt(4845) * sh_8_2 * z
    )
    sh_9_3 = (
        -1 / 306 * math.sqrt(1938) * sh_8_1 * z
        + (1 / 306) * math.sqrt(67830) * sh_8_13 * x
        + (1 / 306) * math.sqrt(1938) * sh_8_15 * x
        + (1 / 51) * math.sqrt(1615) * sh_8_2 * y
        + (1 / 306) * math.sqrt(67830) * sh_8_3 * z
    )
    sh_9_4 = (
        (1 / 306) * math.sqrt(58786) * sh_8_12 * x
        + (1 / 153) * math.sqrt(969) * sh_8_14 * x
        - 1 / 153 * math.sqrt(969) * sh_8_2 * z
        + (2 / 153) * math.sqrt(4522) * sh_8_3 * y
        + (1 / 306) * math.sqrt(58786) * sh_8_4 * z
    )
    sh_9_5 = (
        (1 / 153) * math.sqrt(12597) * sh_8_11 * x
        + (1 / 153) * math.sqrt(1615) * sh_8_13 * x
        - 1 / 153 * math.sqrt(1615) * sh_8_3 * z
        + (1 / 153) * math.sqrt(20995) * sh_8_4 * y
        + (1 / 153) * math.sqrt(12597) * sh_8_5 * z
    )
    sh_9_6 = (
        (1 / 153) * math.sqrt(10659) * sh_8_10 * x
        + (1 / 306) * math.sqrt(9690) * sh_8_12 * x
        - 1 / 306 * math.sqrt(9690) * sh_8_4 * z
        + (2 / 51) * math.sqrt(646) * sh_8_5 * y
        + (1 / 153) * math.sqrt(10659) * sh_8_6 * z
    )
    sh_9_7 = (
        (1 / 306) * math.sqrt(13566) * sh_8_11 * x
        - 1 / 306 * math.sqrt(13566) * sh_8_5 * z
        + (1 / 153) * math.sqrt(24871) * sh_8_6 * y
        + (1 / 306) * math.sqrt(35530) * sh_8_7 * z
        + (1 / 306) * math.sqrt(35530) * sh_8_9 * x
    )
    sh_9_8 = (
        (1 / 153) * math.sqrt(4522) * sh_8_10 * x
        - 1 / 153 * math.sqrt(4522) * sh_8_6 * z
        + (4 / 153) * math.sqrt(1615) * sh_8_7 * y
        + (1 / 51) * math.sqrt(1615) * sh_8_8 * x
    )
    sh_9_9 = (1 / 51) * math.sqrt(323) * (-2 * sh_8_7 * x + 3 * sh_8_8 * y - 2 * sh_8_9 * z)
    sh_9_10 = (
        -1 / 153 * math.sqrt(4522) * sh_8_10 * z
        - 1 / 153 * math.sqrt(4522) * sh_8_6 * x
        + (1 / 51) * math.sqrt(1615) * sh_8_8 * z
        + (4 / 153) * math.sqrt(1615) * sh_8_9 * y
    )
    sh_9_11 = (
        (1 / 153) * math.sqrt(24871) * sh_8_10 * y
        - 1 / 306 * math.sqrt(13566) * sh_8_11 * z
        - 1 / 306 * math.sqrt(13566) * sh_8_5 * x
        - 1 / 306 * math.sqrt(35530) * sh_8_7 * x
        + (1 / 306) * math.sqrt(35530) * sh_8_9 * z
    )
    sh_9_12 = (
        (1 / 153) * math.sqrt(10659) * sh_8_10 * z
        + (2 / 51) * math.sqrt(646) * sh_8_11 * y
        - 1 / 306 * math.sqrt(9690) * sh_8_12 * z
        - 1 / 306 * math.sqrt(9690) * sh_8_4 * x
        - 1 / 153 * math.sqrt(10659) * sh_8_6 * x
    )
    sh_9_13 = (
        (1 / 153) * math.sqrt(12597) * sh_8_11 * z
        + (1 / 153) * math.sqrt(20995) * sh_8_12 * y
        - 1 / 153 * math.sqrt(1615) * sh_8_13 * z
        - 1 / 153 * math.sqrt(1615) * sh_8_3 * x
        - 1 / 153 * math.sqrt(12597) * sh_8_5 * x
    )
    sh_9_14 = (
        (1 / 306) * math.sqrt(58786) * sh_8_12 * z
        + (2 / 153) * math.sqrt(4522) * sh_8_13 * y
        - 1 / 153 * math.sqrt(969) * sh_8_14 * z
        - 1 / 153 * math.sqrt(969) * sh_8_2 * x
        - 1 / 306 * math.sqrt(58786) * sh_8_4 * x
    )
    sh_9_15 = (
        -1 / 306 * math.sqrt(1938) * sh_8_1 * x
        + (1 / 306) * math.sqrt(67830) * sh_8_13 * z
        + (1 / 51) * math.sqrt(1615) * sh_8_14 * y
        - 1 / 306 * math.sqrt(1938) * sh_8_15 * z
        - 1 / 306 * math.sqrt(67830) * sh_8_3 * x
    )
    sh_9_16 = (
        -1 / 306 * math.sqrt(646) * sh_8_0 * x
        + (2 / 153) * math.sqrt(4845) * sh_8_14 * z
        + (4 / 153) * math.sqrt(646) * sh_8_15 * y
        - 1 / 306 * math.sqrt(646) * sh_8_16 * z
        - 2 / 153 * math.sqrt(4845) * sh_8_2 * x
    )
    sh_9_17 = (1 / 9) * math.sqrt(19) * (-2 * sh_8_1 * x + 2 * sh_8_15 * z + sh_8_16 * y)
    sh_9_18 = (1 / 6) * math.sqrt(38) * (-sh_8_0 * x + sh_8_16 * z)
    if lmax == 9:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
                sh_4_0,
                sh_4_1,
                sh_4_2,
                sh_4_3,
                sh_4_4,
                sh_4_5,
                sh_4_6,
                sh_4_7,
                sh_4_8,
                sh_5_0,
                sh_5_1,
                sh_5_2,
                sh_5_3,
                sh_5_4,
                sh_5_5,
                sh_5_6,
                sh_5_7,
                sh_5_8,
                sh_5_9,
                sh_5_10,
                sh_6_0,
                sh_6_1,
                sh_6_2,
                sh_6_3,
                sh_6_4,
                sh_6_5,
                sh_6_6,
                sh_6_7,
                sh_6_8,
                sh_6_9,
                sh_6_10,
                sh_6_11,
                sh_6_12,
                sh_7_0,
                sh_7_1,
                sh_7_2,
                sh_7_3,
                sh_7_4,
                sh_7_5,
                sh_7_6,
                sh_7_7,
                sh_7_8,
                sh_7_9,
                sh_7_10,
                sh_7_11,
                sh_7_12,
                sh_7_13,
                sh_7_14,
                sh_8_0,
                sh_8_1,
                sh_8_2,
                sh_8_3,
                sh_8_4,
                sh_8_5,
                sh_8_6,
                sh_8_7,
                sh_8_8,
                sh_8_9,
                sh_8_10,
                sh_8_11,
                sh_8_12,
                sh_8_13,
                sh_8_14,
                sh_8_15,
                sh_8_16,
                sh_9_0,
                sh_9_1,
                sh_9_2,
                sh_9_3,
                sh_9_4,
                sh_9_5,
                sh_9_6,
                sh_9_7,
                sh_9_8,
                sh_9_9,
                sh_9_10,
                sh_9_11,
                sh_9_12,
                sh_9_13,
                sh_9_14,
                sh_9_15,
                sh_9_16,
                sh_9_17,
                sh_9_18,
            ],
            dim=-1,
        )

    sh_10_0 = (1 / 10) * math.sqrt(105) * (sh_9_0 * z + sh_9_18 * x)
    sh_10_1 = (
        (1 / 10) * math.sqrt(21) * sh_9_0 * y
        + (3 / 20) * math.sqrt(42) * sh_9_1 * z
        + (3 / 20) * math.sqrt(42) * sh_9_17 * x
    )
    sh_10_2 = (
        -1 / 380 * math.sqrt(798) * sh_9_0 * z
        + (3 / 95) * math.sqrt(399) * sh_9_1 * y
        + (3 / 380) * math.sqrt(13566) * sh_9_16 * x
        + (1 / 380) * math.sqrt(798) * sh_9_18 * x
        + (3 / 380) * math.sqrt(13566) * sh_9_2 * z
    )
    sh_10_3 = (
        -3 / 380 * math.sqrt(266) * sh_9_1 * z
        + (1 / 95) * math.sqrt(6783) * sh_9_15 * x
        + (3 / 380) * math.sqrt(266) * sh_9_17 * x
        + (3 / 190) * math.sqrt(2261) * sh_9_2 * y
        + (1 / 95) * math.sqrt(6783) * sh_9_3 * z
    )
    sh_10_4 = (
        (3 / 95) * math.sqrt(665) * sh_9_14 * x
        + (3 / 190) * math.sqrt(133) * sh_9_16 * x
        - 3 / 190 * math.sqrt(133) * sh_9_2 * z
        + (4 / 95) * math.sqrt(399) * sh_9_3 * y
        + (3 / 95) * math.sqrt(665) * sh_9_4 * z
    )
    sh_10_5 = (
        (21 / 380) * math.sqrt(190) * sh_9_13 * x
        + (1 / 190) * math.sqrt(1995) * sh_9_15 * x
        - 1 / 190 * math.sqrt(1995) * sh_9_3 * z
        + (3 / 38) * math.sqrt(133) * sh_9_4 * y
        + (21 / 380) * math.sqrt(190) * sh_9_5 * z
    )
    sh_10_6 = (
        (7 / 380) * math.sqrt(1482) * sh_9_12 * x
        + (3 / 380) * math.sqrt(1330) * sh_9_14 * x
        - 3 / 380 * math.sqrt(1330) * sh_9_4 * z
        + (21 / 95) * math.sqrt(19) * sh_9_5 * y
        + (7 / 380) * math.sqrt(1482) * sh_9_6 * z
    )
    sh_10_7 = (
        (3 / 190) * math.sqrt(1729) * sh_9_11 * x
        + (21 / 380) * math.sqrt(38) * sh_9_13 * x
        - 21 / 380 * math.sqrt(38) * sh_9_5 * z
        + (7 / 190) * math.sqrt(741) * sh_9_6 * y
        + (3 / 190) * math.sqrt(1729) * sh_9_7 * z
    )
    sh_10_8 = (
        (3 / 190) * math.sqrt(1463) * sh_9_10 * x
        + (7 / 190) * math.sqrt(114) * sh_9_12 * x
        - 7 / 190 * math.sqrt(114) * sh_9_6 * z
        + (6 / 95) * math.sqrt(266) * sh_9_7 * y
        + (3 / 190) * math.sqrt(1463) * sh_9_8 * z
    )
    sh_10_9 = (
        (3 / 190) * math.sqrt(798) * sh_9_11 * x
        - 3 / 190 * math.sqrt(798) * sh_9_7 * z
        + (3 / 190) * math.sqrt(4389) * sh_9_8 * y
        + (1 / 190) * math.sqrt(21945) * sh_9_9 * x
    )
    sh_10_10 = (
        -3 / 190 * math.sqrt(1995) * sh_9_10 * z
        - 3 / 190 * math.sqrt(1995) * sh_9_8 * x
        + (1 / 19) * math.sqrt(399) * sh_9_9 * y
    )
    sh_10_11 = (
        (3 / 190) * math.sqrt(4389) * sh_9_10 * y
        - 3 / 190 * math.sqrt(798) * sh_9_11 * z
        - 3 / 190 * math.sqrt(798) * sh_9_7 * x
        + (1 / 190) * math.sqrt(21945) * sh_9_9 * z
    )
    sh_10_12 = (
        (3 / 190) * math.sqrt(1463) * sh_9_10 * z
        + (6 / 95) * math.sqrt(266) * sh_9_11 * y
        - 7 / 190 * math.sqrt(114) * sh_9_12 * z
        - 7 / 190 * math.sqrt(114) * sh_9_6 * x
        - 3 / 190 * math.sqrt(1463) * sh_9_8 * x
    )
    sh_10_13 = (
        (3 / 190) * math.sqrt(1729) * sh_9_11 * z
        + (7 / 190) * math.sqrt(741) * sh_9_12 * y
        - 21 / 380 * math.sqrt(38) * sh_9_13 * z
        - 21 / 380 * math.sqrt(38) * sh_9_5 * x
        - 3 / 190 * math.sqrt(1729) * sh_9_7 * x
    )
    sh_10_14 = (
        (7 / 380) * math.sqrt(1482) * sh_9_12 * z
        + (21 / 95) * math.sqrt(19) * sh_9_13 * y
        - 3 / 380 * math.sqrt(1330) * sh_9_14 * z
        - 3 / 380 * math.sqrt(1330) * sh_9_4 * x
        - 7 / 380 * math.sqrt(1482) * sh_9_6 * x
    )
    sh_10_15 = (
        (21 / 380) * math.sqrt(190) * sh_9_13 * z
        + (3 / 38) * math.sqrt(133) * sh_9_14 * y
        - 1 / 190 * math.sqrt(1995) * sh_9_15 * z
        - 1 / 190 * math.sqrt(1995) * sh_9_3 * x
        - 21 / 380 * math.sqrt(190) * sh_9_5 * x
    )
    sh_10_16 = (
        (3 / 95) * math.sqrt(665) * sh_9_14 * z
        + (4 / 95) * math.sqrt(399) * sh_9_15 * y
        - 3 / 190 * math.sqrt(133) * sh_9_16 * z
        - 3 / 190 * math.sqrt(133) * sh_9_2 * x
        - 3 / 95 * math.sqrt(665) * sh_9_4 * x
    )
    sh_10_17 = (
        -3 / 380 * math.sqrt(266) * sh_9_1 * x
        + (1 / 95) * math.sqrt(6783) * sh_9_15 * z
        + (3 / 190) * math.sqrt(2261) * sh_9_16 * y
        - 3 / 380 * math.sqrt(266) * sh_9_17 * z
        - 1 / 95 * math.sqrt(6783) * sh_9_3 * x
    )
    sh_10_18 = (
        -1 / 380 * math.sqrt(798) * sh_9_0 * x
        + (3 / 380) * math.sqrt(13566) * sh_9_16 * z
        + (3 / 95) * math.sqrt(399) * sh_9_17 * y
        - 1 / 380 * math.sqrt(798) * sh_9_18 * z
        - 3 / 380 * math.sqrt(13566) * sh_9_2 * x
    )
    sh_10_19 = (
        -3 / 20 * math.sqrt(42) * sh_9_1 * x
        + (3 / 20) * math.sqrt(42) * sh_9_17 * z
        + (1 / 10) * math.sqrt(21) * sh_9_18 * y
    )
    sh_10_20 = (1 / 10) * math.sqrt(105) * (-sh_9_0 * x + sh_9_18 * z)
    if lmax == 10:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
                sh_4_0,
                sh_4_1,
                sh_4_2,
                sh_4_3,
                sh_4_4,
                sh_4_5,
                sh_4_6,
                sh_4_7,
                sh_4_8,
                sh_5_0,
                sh_5_1,
                sh_5_2,
                sh_5_3,
                sh_5_4,
                sh_5_5,
                sh_5_6,
                sh_5_7,
                sh_5_8,
                sh_5_9,
                sh_5_10,
                sh_6_0,
                sh_6_1,
                sh_6_2,
                sh_6_3,
                sh_6_4,
                sh_6_5,
                sh_6_6,
                sh_6_7,
                sh_6_8,
                sh_6_9,
                sh_6_10,
                sh_6_11,
                sh_6_12,
                sh_7_0,
                sh_7_1,
                sh_7_2,
                sh_7_3,
                sh_7_4,
                sh_7_5,
                sh_7_6,
                sh_7_7,
                sh_7_8,
                sh_7_9,
                sh_7_10,
                sh_7_11,
                sh_7_12,
                sh_7_13,
                sh_7_14,
                sh_8_0,
                sh_8_1,
                sh_8_2,
                sh_8_3,
                sh_8_4,
                sh_8_5,
                sh_8_6,
                sh_8_7,
                sh_8_8,
                sh_8_9,
                sh_8_10,
                sh_8_11,
                sh_8_12,
                sh_8_13,
                sh_8_14,
                sh_8_15,
                sh_8_16,
                sh_9_0,
                sh_9_1,
                sh_9_2,
                sh_9_3,
                sh_9_4,
                sh_9_5,
                sh_9_6,
                sh_9_7,
                sh_9_8,
                sh_9_9,
                sh_9_10,
                sh_9_11,
                sh_9_12,
                sh_9_13,
                sh_9_14,
                sh_9_15,
                sh_9_16,
                sh_9_17,
                sh_9_18,
                sh_10_0,
                sh_10_1,
                sh_10_2,
                sh_10_3,
                sh_10_4,
                sh_10_5,
                sh_10_6,
                sh_10_7,
                sh_10_8,
                sh_10_9,
                sh_10_10,
                sh_10_11,
                sh_10_12,
                sh_10_13,
                sh_10_14,
                sh_10_15,
                sh_10_16,
                sh_10_17,
                sh_10_18,
                sh_10_19,
                sh_10_20,
            ],
            dim=-1,
        )

    sh_11_0 = (1 / 22) * math.sqrt(506) * (sh_10_0 * z + sh_10_20 * x)
    sh_11_1 = (
        (1 / 11) * math.sqrt(23) * sh_10_0 * y
        + (1 / 11) * math.sqrt(115) * sh_10_1 * z
        + (1 / 11) * math.sqrt(115) * sh_10_19 * x
    )
    sh_11_2 = (
        -1 / 462 * math.sqrt(966) * sh_10_0 * z
        + (2 / 231) * math.sqrt(4830) * sh_10_1 * y
        + (1 / 231) * math.sqrt(45885) * sh_10_18 * x
        + (1 / 231) * math.sqrt(45885) * sh_10_2 * z
        + (1 / 462) * math.sqrt(966) * sh_10_20 * x
    )
    sh_11_3 = (
        -1 / 154 * math.sqrt(322) * sh_10_1 * z
        + (1 / 154) * math.sqrt(18354) * sh_10_17 * x
        + (1 / 154) * math.sqrt(322) * sh_10_19 * x
        + (1 / 77) * math.sqrt(3059) * sh_10_2 * y
        + (1 / 154) * math.sqrt(18354) * sh_10_3 * z
    )
    sh_11_4 = (
        (1 / 154) * math.sqrt(16422) * sh_10_16 * x
        + (1 / 77) * math.sqrt(161) * sh_10_18 * x
        - 1 / 77 * math.sqrt(161) * sh_10_2 * z
        + (2 / 77) * math.sqrt(966) * sh_10_3 * y
        + (1 / 154) * math.sqrt(16422) * sh_10_4 * z
    )
    sh_11_5 = (
        (2 / 231) * math.sqrt(8211) * sh_10_15 * x
        + (1 / 231) * math.sqrt(2415) * sh_10_17 * x
        - 1 / 231 * math.sqrt(2415) * sh_10_3 * z
        + (1 / 231) * math.sqrt(41055) * sh_10_4 * y
        + (2 / 231) * math.sqrt(8211) * sh_10_5 * z
    )
    sh_11_6 = (
        (2 / 77) * math.sqrt(805) * sh_10_14 * x
        + (1 / 154) * math.sqrt(1610) * sh_10_16 * x
        - 1 / 154 * math.sqrt(1610) * sh_10_4 * z
        + (4 / 77) * math.sqrt(322) * sh_10_5 * y
        + (2 / 77) * math.sqrt(805) * sh_10_6 * z
    )
    sh_11_7 = (
        (1 / 22) * math.sqrt(230) * sh_10_13 * x
        + (1 / 22) * math.sqrt(46) * sh_10_15 * x
        - 1 / 22 * math.sqrt(46) * sh_10_5 * z
        + (1 / 11) * math.sqrt(115) * sh_10_6 * y
        + (1 / 22) * math.sqrt(230) * sh_10_7 * z
    )
    sh_11_8 = (
        (1 / 66) * math.sqrt(1794) * sh_10_12 * x
        + (1 / 33) * math.sqrt(138) * sh_10_14 * x
        - 1 / 33 * math.sqrt(138) * sh_10_6 * z
        + (4 / 33) * math.sqrt(69) * sh_10_7 * y
        + (1 / 66) * math.sqrt(1794) * sh_10_8 * z
    )
    sh_11_9 = (
        (1 / 77) * math.sqrt(2093) * sh_10_11 * x
        + (1 / 77) * math.sqrt(966) * sh_10_13 * x
        - 1 / 77 * math.sqrt(966) * sh_10_7 * z
        + (1 / 77) * math.sqrt(6279) * sh_10_8 * y
        + (1 / 77) * math.sqrt(2093) * sh_10_9 * z
    )
    sh_11_10 = (
        (1 / 77) * math.sqrt(3542) * sh_10_10 * x
        + (1 / 154) * math.sqrt(4830) * sh_10_12 * x
        - 1 / 154 * math.sqrt(4830) * sh_10_8 * z
        + (2 / 77) * math.sqrt(1610) * sh_10_9 * y
    )
    sh_11_11 = (
        (1 / 21) * math.sqrt(483) * sh_10_10 * y
        - 1 / 231 * math.sqrt(26565) * sh_10_11 * z
        - 1 / 231 * math.sqrt(26565) * sh_10_9 * x
    )
    sh_11_12 = (
        (1 / 77) * math.sqrt(3542) * sh_10_10 * z
        + (2 / 77) * math.sqrt(1610) * sh_10_11 * y
        - 1 / 154 * math.sqrt(4830) * sh_10_12 * z
        - 1 / 154 * math.sqrt(4830) * sh_10_8 * x
    )
    sh_11_13 = (
        (1 / 77) * math.sqrt(2093) * sh_10_11 * z
        + (1 / 77) * math.sqrt(6279) * sh_10_12 * y
        - 1 / 77 * math.sqrt(966) * sh_10_13 * z
        - 1 / 77 * math.sqrt(966) * sh_10_7 * x
        - 1 / 77 * math.sqrt(2093) * sh_10_9 * x
    )
    sh_11_14 = (
        (1 / 66) * math.sqrt(1794) * sh_10_12 * z
        + (4 / 33) * math.sqrt(69) * sh_10_13 * y
        - 1 / 33 * math.sqrt(138) * sh_10_14 * z
        - 1 / 33 * math.sqrt(138) * sh_10_6 * x
        - 1 / 66 * math.sqrt(1794) * sh_10_8 * x
    )
    sh_11_15 = (
        (1 / 22) * math.sqrt(230) * sh_10_13 * z
        + (1 / 11) * math.sqrt(115) * sh_10_14 * y
        - 1 / 22 * math.sqrt(46) * sh_10_15 * z
        - 1 / 22 * math.sqrt(46) * sh_10_5 * x
        - 1 / 22 * math.sqrt(230) * sh_10_7 * x
    )
    sh_11_16 = (
        (2 / 77) * math.sqrt(805) * sh_10_14 * z
        + (4 / 77) * math.sqrt(322) * sh_10_15 * y
        - 1 / 154 * math.sqrt(1610) * sh_10_16 * z
        - 1 / 154 * math.sqrt(1610) * sh_10_4 * x
        - 2 / 77 * math.sqrt(805) * sh_10_6 * x
    )
    sh_11_17 = (
        (2 / 231) * math.sqrt(8211) * sh_10_15 * z
        + (1 / 231) * math.sqrt(41055) * sh_10_16 * y
        - 1 / 231 * math.sqrt(2415) * sh_10_17 * z
        - 1 / 231 * math.sqrt(2415) * sh_10_3 * x
        - 2 / 231 * math.sqrt(8211) * sh_10_5 * x
    )
    sh_11_18 = (
        (1 / 154) * math.sqrt(16422) * sh_10_16 * z
        + (2 / 77) * math.sqrt(966) * sh_10_17 * y
        - 1 / 77 * math.sqrt(161) * sh_10_18 * z
        - 1 / 77 * math.sqrt(161) * sh_10_2 * x
        - 1 / 154 * math.sqrt(16422) * sh_10_4 * x
    )
    sh_11_19 = (
        -1 / 154 * math.sqrt(322) * sh_10_1 * x
        + (1 / 154) * math.sqrt(18354) * sh_10_17 * z
        + (1 / 77) * math.sqrt(3059) * sh_10_18 * y
        - 1 / 154 * math.sqrt(322) * sh_10_19 * z
        - 1 / 154 * math.sqrt(18354) * sh_10_3 * x
    )
    sh_11_20 = (
        -1 / 462 * math.sqrt(966) * sh_10_0 * x
        + (1 / 231) * math.sqrt(45885) * sh_10_18 * z
        + (2 / 231) * math.sqrt(4830) * sh_10_19 * y
        - 1 / 231 * math.sqrt(45885) * sh_10_2 * x
        - 1 / 462 * math.sqrt(966) * sh_10_20 * z
    )
    sh_11_21 = (
        -1 / 11 * math.sqrt(115) * sh_10_1 * x
        + (1 / 11) * math.sqrt(115) * sh_10_19 * z
        + (1 / 11) * math.sqrt(23) * sh_10_20 * y
    )
    sh_11_22 = (1 / 22) * math.sqrt(506) * (-sh_10_0 * x + sh_10_20 * z)
    if lmax == 11:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
                sh_4_0,
                sh_4_1,
                sh_4_2,
                sh_4_3,
                sh_4_4,
                sh_4_5,
                sh_4_6,
                sh_4_7,
                sh_4_8,
                sh_5_0,
                sh_5_1,
                sh_5_2,
                sh_5_3,
                sh_5_4,
                sh_5_5,
                sh_5_6,
                sh_5_7,
                sh_5_8,
                sh_5_9,
                sh_5_10,
                sh_6_0,
                sh_6_1,
                sh_6_2,
                sh_6_3,
                sh_6_4,
                sh_6_5,
                sh_6_6,
                sh_6_7,
                sh_6_8,
                sh_6_9,
                sh_6_10,
                sh_6_11,
                sh_6_12,
                sh_7_0,
                sh_7_1,
                sh_7_2,
                sh_7_3,
                sh_7_4,
                sh_7_5,
                sh_7_6,
                sh_7_7,
                sh_7_8,
                sh_7_9,
                sh_7_10,
                sh_7_11,
                sh_7_12,
                sh_7_13,
                sh_7_14,
                sh_8_0,
                sh_8_1,
                sh_8_2,
                sh_8_3,
                sh_8_4,
                sh_8_5,
                sh_8_6,
                sh_8_7,
                sh_8_8,
                sh_8_9,
                sh_8_10,
                sh_8_11,
                sh_8_12,
                sh_8_13,
                sh_8_14,
                sh_8_15,
                sh_8_16,
                sh_9_0,
                sh_9_1,
                sh_9_2,
                sh_9_3,
                sh_9_4,
                sh_9_5,
                sh_9_6,
                sh_9_7,
                sh_9_8,
                sh_9_9,
                sh_9_10,
                sh_9_11,
                sh_9_12,
                sh_9_13,
                sh_9_14,
                sh_9_15,
                sh_9_16,
                sh_9_17,
                sh_9_18,
                sh_10_0,
                sh_10_1,
                sh_10_2,
                sh_10_3,
                sh_10_4,
                sh_10_5,
                sh_10_6,
                sh_10_7,
                sh_10_8,
                sh_10_9,
                sh_10_10,
                sh_10_11,
                sh_10_12,
                sh_10_13,
                sh_10_14,
                sh_10_15,
                sh_10_16,
                sh_10_17,
                sh_10_18,
                sh_10_19,
                sh_10_20,
                sh_11_0,
                sh_11_1,
                sh_11_2,
                sh_11_3,
                sh_11_4,
                sh_11_5,
                sh_11_6,
                sh_11_7,
                sh_11_8,
                sh_11_9,
                sh_11_10,
                sh_11_11,
                sh_11_12,
                sh_11_13,
                sh_11_14,
                sh_11_15,
                sh_11_16,
                sh_11_17,
                sh_11_18,
                sh_11_19,
                sh_11_20,
                sh_11_21,
                sh_11_22,
            ],
            dim=-1,
        )

    sh_12_0 = (5 / 12) * math.sqrt(6) * (sh_11_0 * z + sh_11_22 * x)
    sh_12_1 = (5 / 12) * sh_11_0 * y + (5 / 24) * math.sqrt(22) * sh_11_1 * z + (5 / 24) * math.sqrt(22) * sh_11_21 * x
    sh_12_2 = (
        -5 / 552 * math.sqrt(46) * sh_11_0 * z
        + (5 / 138) * math.sqrt(253) * sh_11_1 * y
        + (5 / 552) * math.sqrt(10626) * sh_11_2 * z
        + (5 / 552) * math.sqrt(10626) * sh_11_20 * x
        + (5 / 552) * math.sqrt(46) * sh_11_22 * x
    )
    sh_12_3 = (
        -5 / 552 * math.sqrt(138) * sh_11_1 * z
        + (5 / 276) * math.sqrt(2415) * sh_11_19 * x
        + (5 / 92) * math.sqrt(161) * sh_11_2 * y
        + (5 / 552) * math.sqrt(138) * sh_11_21 * x
        + (5 / 276) * math.sqrt(2415) * sh_11_3 * z
    )
    sh_12_4 = (
        (5 / 276) * math.sqrt(2185) * sh_11_18 * x
        - 5 / 276 * math.sqrt(69) * sh_11_2 * z
        + (5 / 276) * math.sqrt(69) * sh_11_20 * x
        + (5 / 69) * math.sqrt(115) * sh_11_3 * y
        + (5 / 276) * math.sqrt(2185) * sh_11_4 * z
    )
    sh_12_5 = (
        (5 / 184) * math.sqrt(874) * sh_11_17 * x
        + (5 / 276) * math.sqrt(115) * sh_11_19 * x
        - 5 / 276 * math.sqrt(115) * sh_11_3 * z
        + (5 / 276) * math.sqrt(2185) * sh_11_4 * y
        + (5 / 184) * math.sqrt(874) * sh_11_5 * z
    )
    sh_12_6 = (
        (5 / 552)
        * math.sqrt(3)
        * (
            math.sqrt(2346) * sh_11_16 * x
            + math.sqrt(230) * sh_11_18 * x
            - math.sqrt(230) * sh_11_4 * z
            + 12 * math.sqrt(23) * sh_11_5 * y
            + math.sqrt(2346) * sh_11_6 * z
        )
    )
    sh_12_7 = (
        (5 / 138) * math.sqrt(391) * sh_11_15 * x
        + (5 / 552) * math.sqrt(966) * sh_11_17 * x
        - 5 / 552 * math.sqrt(966) * sh_11_5 * z
        + (5 / 276) * math.sqrt(2737) * sh_11_6 * y
        + (5 / 138) * math.sqrt(391) * sh_11_7 * z
    )
    sh_12_8 = (
        (5 / 138) * math.sqrt(345) * sh_11_14 * x
        + (5 / 276) * math.sqrt(322) * sh_11_16 * x
        - 5 / 276 * math.sqrt(322) * sh_11_6 * z
        + (10 / 69) * math.sqrt(46) * sh_11_7 * y
        + (5 / 138) * math.sqrt(345) * sh_11_8 * z
    )
    sh_12_9 = (
        (5 / 552) * math.sqrt(4830) * sh_11_13 * x
        + (5 / 92) * math.sqrt(46) * sh_11_15 * x
        - 5 / 92 * math.sqrt(46) * sh_11_7 * z
        + (5 / 92) * math.sqrt(345) * sh_11_8 * y
        + (5 / 552) * math.sqrt(4830) * sh_11_9 * z
    )
    sh_12_10 = (
        (5 / 552) * math.sqrt(4186) * sh_11_10 * z
        + (5 / 552) * math.sqrt(4186) * sh_11_12 * x
        + (5 / 184) * math.sqrt(230) * sh_11_14 * x
        - 5 / 184 * math.sqrt(230) * sh_11_8 * z
        + (5 / 138) * math.sqrt(805) * sh_11_9 * y
    )
    sh_12_11 = (
        (5 / 276) * math.sqrt(3289) * sh_11_10 * y
        + (5 / 276) * math.sqrt(1794) * sh_11_11 * x
        + (5 / 552) * math.sqrt(2530) * sh_11_13 * x
        - 5 / 552 * math.sqrt(2530) * sh_11_9 * z
    )
    sh_12_12 = (
        -5 / 276 * math.sqrt(1518) * sh_11_10 * x
        + (5 / 23) * math.sqrt(23) * sh_11_11 * y
        - 5 / 276 * math.sqrt(1518) * sh_11_12 * z
    )
    sh_12_13 = (
        (5 / 276) * math.sqrt(1794) * sh_11_11 * z
        + (5 / 276) * math.sqrt(3289) * sh_11_12 * y
        - 5 / 552 * math.sqrt(2530) * sh_11_13 * z
        - 5 / 552 * math.sqrt(2530) * sh_11_9 * x
    )
    sh_12_14 = (
        -5 / 552 * math.sqrt(4186) * sh_11_10 * x
        + (5 / 552) * math.sqrt(4186) * sh_11_12 * z
        + (5 / 138) * math.sqrt(805) * sh_11_13 * y
        - 5 / 184 * math.sqrt(230) * sh_11_14 * z
        - 5 / 184 * math.sqrt(230) * sh_11_8 * x
    )
    sh_12_15 = (
        (5 / 552) * math.sqrt(4830) * sh_11_13 * z
        + (5 / 92) * math.sqrt(345) * sh_11_14 * y
        - 5 / 92 * math.sqrt(46) * sh_11_15 * z
        - 5 / 92 * math.sqrt(46) * sh_11_7 * x
        - 5 / 552 * math.sqrt(4830) * sh_11_9 * x
    )
    sh_12_16 = (
        (5 / 138) * math.sqrt(345) * sh_11_14 * z
        + (10 / 69) * math.sqrt(46) * sh_11_15 * y
        - 5 / 276 * math.sqrt(322) * sh_11_16 * z
        - 5 / 276 * math.sqrt(322) * sh_11_6 * x
        - 5 / 138 * math.sqrt(345) * sh_11_8 * x
    )
    sh_12_17 = (
        (5 / 138) * math.sqrt(391) * sh_11_15 * z
        + (5 / 276) * math.sqrt(2737) * sh_11_16 * y
        - 5 / 552 * math.sqrt(966) * sh_11_17 * z
        - 5 / 552 * math.sqrt(966) * sh_11_5 * x
        - 5 / 138 * math.sqrt(391) * sh_11_7 * x
    )
    sh_12_18 = (
        (5 / 552)
        * math.sqrt(3)
        * (
            math.sqrt(2346) * sh_11_16 * z
            + 12 * math.sqrt(23) * sh_11_17 * y
            - math.sqrt(230) * sh_11_18 * z
            - math.sqrt(230) * sh_11_4 * x
            - math.sqrt(2346) * sh_11_6 * x
        )
    )
    sh_12_19 = (
        (5 / 184) * math.sqrt(874) * sh_11_17 * z
        + (5 / 276) * math.sqrt(2185) * sh_11_18 * y
        - 5 / 276 * math.sqrt(115) * sh_11_19 * z
        - 5 / 276 * math.sqrt(115) * sh_11_3 * x
        - 5 / 184 * math.sqrt(874) * sh_11_5 * x
    )
    sh_12_20 = (
        (5 / 276) * math.sqrt(2185) * sh_11_18 * z
        + (5 / 69) * math.sqrt(115) * sh_11_19 * y
        - 5 / 276 * math.sqrt(69) * sh_11_2 * x
        - 5 / 276 * math.sqrt(69) * sh_11_20 * z
        - 5 / 276 * math.sqrt(2185) * sh_11_4 * x
    )
    sh_12_21 = (
        -5 / 552 * math.sqrt(138) * sh_11_1 * x
        + (5 / 276) * math.sqrt(2415) * sh_11_19 * z
        + (5 / 92) * math.sqrt(161) * sh_11_20 * y
        - 5 / 552 * math.sqrt(138) * sh_11_21 * z
        - 5 / 276 * math.sqrt(2415) * sh_11_3 * x
    )
    sh_12_22 = (
        -5 / 552 * math.sqrt(46) * sh_11_0 * x
        - 5 / 552 * math.sqrt(10626) * sh_11_2 * x
        + (5 / 552) * math.sqrt(10626) * sh_11_20 * z
        + (5 / 138) * math.sqrt(253) * sh_11_21 * y
        - 5 / 552 * math.sqrt(46) * sh_11_22 * z
    )
    sh_12_23 = -5 / 24 * math.sqrt(22) * sh_11_1 * x + (5 / 24) * math.sqrt(22) * sh_11_21 * z + (5 / 12) * sh_11_22 * y
    sh_12_24 = (5 / 12) * math.sqrt(6) * (-sh_11_0 * x + sh_11_22 * z)

    return torch.stack(
        [
            sh_0_0,
            sh_1_0,
            sh_1_1,
            sh_1_2,
            sh_2_0,
            sh_2_1,
            sh_2_2,
            sh_2_3,
            sh_2_4,
            sh_3_0,
            sh_3_1,
            sh_3_2,
            sh_3_3,
            sh_3_4,
            sh_3_5,
            sh_3_6,
            sh_4_0,
            sh_4_1,
            sh_4_2,
            sh_4_3,
            sh_4_4,
            sh_4_5,
            sh_4_6,
            sh_4_7,
            sh_4_8,
            sh_5_0,
            sh_5_1,
            sh_5_2,
            sh_5_3,
            sh_5_4,
            sh_5_5,
            sh_5_6,
            sh_5_7,
            sh_5_8,
            sh_5_9,
            sh_5_10,
            sh_6_0,
            sh_6_1,
            sh_6_2,
            sh_6_3,
            sh_6_4,
            sh_6_5,
            sh_6_6,
            sh_6_7,
            sh_6_8,
            sh_6_9,
            sh_6_10,
            sh_6_11,
            sh_6_12,
            sh_7_0,
            sh_7_1,
            sh_7_2,
            sh_7_3,
            sh_7_4,
            sh_7_5,
            sh_7_6,
            sh_7_7,
            sh_7_8,
            sh_7_9,
            sh_7_10,
            sh_7_11,
            sh_7_12,
            sh_7_13,
            sh_7_14,
            sh_8_0,
            sh_8_1,
            sh_8_2,
            sh_8_3,
            sh_8_4,
            sh_8_5,
            sh_8_6,
            sh_8_7,
            sh_8_8,
            sh_8_9,
            sh_8_10,
            sh_8_11,
            sh_8_12,
            sh_8_13,
            sh_8_14,
            sh_8_15,
            sh_8_16,
            sh_9_0,
            sh_9_1,
            sh_9_2,
            sh_9_3,
            sh_9_4,
            sh_9_5,
            sh_9_6,
            sh_9_7,
            sh_9_8,
            sh_9_9,
            sh_9_10,
            sh_9_11,
            sh_9_12,
            sh_9_13,
            sh_9_14,
            sh_9_15,
            sh_9_16,
            sh_9_17,
            sh_9_18,
            sh_10_0,
            sh_10_1,
            sh_10_2,
            sh_10_3,
            sh_10_4,
            sh_10_5,
            sh_10_6,
            sh_10_7,
            sh_10_8,
            sh_10_9,
            sh_10_10,
            sh_10_11,
            sh_10_12,
            sh_10_13,
            sh_10_14,
            sh_10_15,
            sh_10_16,
            sh_10_17,
            sh_10_18,
            sh_10_19,
            sh_10_20,
            sh_11_0,
            sh_11_1,
            sh_11_2,
            sh_11_3,
            sh_11_4,
            sh_11_5,
            sh_11_6,
            sh_11_7,
            sh_11_8,
            sh_11_9,
            sh_11_10,
            sh_11_11,
            sh_11_12,
            sh_11_13,
            sh_11_14,
            sh_11_15,
            sh_11_16,
            sh_11_17,
            sh_11_18,
            sh_11_19,
            sh_11_20,
            sh_11_21,
            sh_11_22,
            sh_12_0,
            sh_12_1,
            sh_12_2,
            sh_12_3,
            sh_12_4,
            sh_12_5,
            sh_12_6,
            sh_12_7,
            sh_12_8,
            sh_12_9,
            sh_12_10,
            sh_12_11,
            sh_12_12,
            sh_12_13,
            sh_12_14,
            sh_12_15,
            sh_12_16,
            sh_12_17,
            sh_12_18,
            sh_12_19,
            sh_12_20,
            sh_12_21,
            sh_12_22,
            sh_12_23,
            sh_12_24,
        ],
        dim=-1,
    )
