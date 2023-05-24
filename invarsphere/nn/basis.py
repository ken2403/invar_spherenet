from __future__ import annotations

import math
import os
from collections.abc import Callable
from functools import lru_cache

import numpy as np
import sympy
import torch
from torch import Tensor


class SphericalBesselFunction(torch.nn.Module):
    """Calculate the spherical Bessel function based on the sympy + torch
    implementations."""

    def __init__(self, max_l: int = 6, max_n: int = 5, cutoff: float = 5.0, smooth: bool = False):
        """
        Args:
            max_l(int): max order (excluding l)
            max_n (int): max number of roots used in each l
            cutoff (float): cutoff radius
            smooth (bool): whether to use smooth version of spherical Bessel function
        """
        super().__init__()
        self.max_l = max_l
        self.max_n = max_n
        self.cutoff = cutoff
        self.smooth = smooth
        if smooth:
            self.funcs = self._calculate_smooth_symbolic_funcs()
        else:
            self.funcs = self._calculate_symbolic_funcs()

        self.register_buffer("sb_roots", self._get_spherical_bessel_roots())

    @lru_cache(maxsize=128)
    def _get_spherical_bessel_roots(self) -> Tensor:
        return torch.tensor(np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sb_roots.txt")))

    @lru_cache(maxsize=128)
    def _calculate_symbolic_funcs(self) -> list[Callable]:
        """Spherical basis functions based on Rayleigh formula. This function
        generates symbolic formula.

        Returns:
            symbolic_funcs (list): list of symbolic functions
        """
        r = sympy.symbols("r")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt, "exp": torch.exp}
        funcs = [sympy.expand_func(sympy.functions.special.bessel.jn(i, r)) for i in range(self.max_l + 1)]
        return [sympy.lambdify(r, sympy.simplify(f).evalf(), modules) for f in funcs]

    @lru_cache(maxsize=128)
    def _calculate_smooth_symbolic_funcs(self) -> list[Callable]:
        r = sympy.symbols("r")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt, "exp": torch.exp}

        d0 = 1.0
        en = []
        for i in range(self.max_n):
            en.append(i**2 * (i + 2) ** 2 / (4 * (i + 1) ** 4 + 1))

        dn = [d0]
        for i in range(1, self.max_n):
            dn.append(1 - en[i] / dn[-1])

        fnr = []
        for i in range(self.max_n):
            fnr.append(
                (-1) ** i
                * sympy.sqrt(2.0)
                * sympy.pi
                / self.cutoff**1.5
                * (i + 1)
                * (i + 2)
                / sympy.sqrt(1.0 * (i + 1) ** 2 + (i + 2) ** 2)
                * (
                    sympy.sin(r * (i + 1) * sympy.pi / self.cutoff) / (r * (i + 1) * sympy.pi / self.cutoff)
                    + sympy.sin(r * (i + 2) * sympy.pi / self.cutoff) / (r * (i + 2) * sympy.pi / self.cutoff)
                )
            )

        gnr = [fnr[0]]
        for i in range(1, self.max_n):
            gnr.append(1 / sympy.sqrt(dn[i]) * (fnr[i] + sympy.sqrt(en[i] / dn[i - 1]) * gnr[-1]))

        return [sympy.lambdify([r], sympy.simplify(f).evalf(), modules) for f in gnr]

    def _call_sbf(self, r: Tensor) -> Tensor:
        roots = self.sb_roots[: self.max_l, : self.max_n]  # type: ignore # Since mypy cannnot determine sb_roots is Tensor # noqa: E501

        results = []
        factor = math.sqrt(2.0 / self.cutoff**3)
        for i in range(self.max_l):
            root = roots[i]
            func = self.funcs[i]
            func_add1 = self.funcs[i + 1]
            results.append(
                func(r[:, None] * root[None, :] / self.cutoff) * factor / torch.abs(func_add1(root[None, :]))
            )
        return torch.stack(results, dim=-1)

    def _call_smooth_sbf(self, r: Tensor) -> Tensor:
        return torch.stack([i(r) for i in self.funcs], dim=-1)

    def forward(self, r: Tensor) -> Tensor:
        """
        Args:
            r (torch.Tensor): distance Tensor with (*) shape

        Returns:
            sbb (torch.Tensor): spherical Bessel basis with (*, max_n * max_l) shape
        """
        if self.smooth:
            return self._call_smooth_sbf(r)
        return self._call_sbf(r)

    @staticmethod
    def rbf_j0(r: Tensor, cutoff: float = 5.0, max_n: int = 3) -> Tensor:
        """Spherical Bessel function of order 0, ensuring the function value
        vanishes at cutoff.

        Args:
            r (torch.Tensor): distance Tensor with (E) shape
            cutoff (float): the cutoff radius
            max_n (int): max number of basis

        Returns:
            expanded_r (torch.Tensor): basis function expansion using first spherical Bessel function with (E, max_n) shape
        """  # noqa: E501
        n = torch.arange(1, max_n + 1, device=r.device, dtype=r.dtype)[None, :]
        r = r[:, None]
        return math.sqrt(2.0 / cutoff) * torch.sin(n * math.pi / cutoff * r) / r


class SphericalHarmonicsFunction(torch.nn.Module):
    """Spherical Harmonics function based on sympy + torch implementations."""

    def __init__(self, max_l: int, use_phi: bool = True):
        """
        Args:
            max_l (int): max degree of spherical harmonics (excluding l)
            use_phi (bool): whether to use the polar angle. If not, the function will compute `Y_l^0` only
        """
        super().__init__()
        self.max_l = max_l
        self.use_phi = use_phi
        self.funcs = self._calculate_symbolic_funcs()

    @staticmethod
    def _y00(theta: Tensor, phi: Tensor) -> Tensor:
        r"""
        Spherical Harmonics with `l=m=0`.
        ..math::
            Y_0^0 = \frac{1}{2} \sqrt{\frac{1}{\pi}}

        Args:
            theta: the azimuthal angle.
            phi: the polar angle.

        Returns:
            `Y_0^0`: the spherical harmonics with `l=m=0`.
        """
        dtype = theta.dtype
        return (0.5 * torch.ones_like(theta) * math.sqrt(1.0 / math.pi)).to(dtype)

    def _calculate_symbolic_funcs(self) -> list[Callable]:
        theta, phi = sympy.symbols("theta phi")
        modules = {"sin": torch.sin, "cos": torch.cos, "conjugate": torch.conj, "sqrt": torch.sqrt, "exp": torch.exp}

        funcs = []
        for lval in range(self.max_l):
            if self.use_phi:
                m_list = list(range(-lval, lval + 1))
            else:
                m_list = [0]
            for m in m_list:
                func = sympy.expand_func(sympy.functions.special.spherical_harmonics.Znm(lval, m, theta, phi))
                funcs.append(func)
        results = [sympy.lambdify([theta, phi], sympy.simplify(f).evalf(), modules) for f in funcs]
        results[0] = SphericalHarmonicsFunction._y00
        return results

    def forward(self, theta: Tensor, phi: Tensor | None = None) -> Tensor:
        """Forward calculation of SphericalHarmonicsBasis.

        Args:
            theta (torch.Tensor): the azimuthal angle with (*) shape
            phi (torch.Tensor | None): the polar angle with (*) shape

        Returns:
            shb (torch.Tensor): spherical harmonics basis with (*, max_l) shape if not use_phi, (*, max_l*max_l) shape if use_phi
        """  # noqa: E501
        shb = torch.stack([f(theta, phi) for f in self.funcs], dim=-1)
        return shb


def combine_sbb_shb(sbb: Tensor, shb: Tensor, max_n: int, max_l: int, use_phi: bool):
    """Combine the spherical Bessel basis and the spherical Harmonics basis.
    For the spherical Bessel function, the column is ordered by.

        [n=[0, ..., max_n-1], n=[0, ..., max_n-1], ...], max_l blocks,

    For the spherical Harmonics function, the column is ordered by
        [m=[0], m=[-1, 0, 1], m=[-2, -1, 0, 1, 2], ...] max_l blocks, and each block has 2*l + 1
        if use_phi is False, then the columns become
        [m=[0], m=[0], ...] max_l columns

    Args:
        sbb (Tensor): spherical bessel basis results with (*, max_n*max_l) shape.
        shb (Tensor): spherical harmonics basis results with (*, max_l) shape if not use_phi, (*, max_l*max_l) shape if use_phi.
        max_n (int): max number of n.
        max_l (int): max number of l.
        use_phi (bool): whether to use phi.

    Returns:
        combined_sbf_shf (Tensor): combination of spherical bessel and spherical harmonics.
    """  # noqa: E501
    if sbb.size(0) == 0:
        return sbb

    device = sbb.device
    if not use_phi:
        repeats_sbb = torch.tensor([1] * max_l * max_n, device=device)
        block_size = np.array([1] * max_l)
    else:
        # [1, 1, 1, ..., 1, 3, 3, 3, ..., 3, ... 5, ..., 2*max_l-1,...]
        repeats_sbb = torch.tensor(np.repeat(2 * np.arange(max_l) + 1, repeats=max_n), device=device)
        block_size = 2 * np.arange(max_l) + 1
    expanded_sbb = torch.repeat_interleave(sbb, repeats=repeats_sbb, dim=-1)
    expanded_shb = _block_repeat(shb, block_size=block_size, repeats=np.array([max_n] * max_l))
    shape = max_n * max_l
    if use_phi:
        shape *= max_l
    return (expanded_sbb * expanded_shb).view(-1, shape)


def _block_repeat(t: Tensor, block_size: np.ndarray, repeats: np.ndarray) -> Tensor:
    col_index = torch.arange(t.size(1))
    indices = []
    start = 0

    for i, b in enumerate(block_size):
        indices.append(torch.tile(col_index[start : start + b], (repeats[i],)))  # noqa: E203
        start += b
    indices_tensor = torch.cat(indices, dim=0).to(t.device).long()
    return t[..., indices_tensor]
