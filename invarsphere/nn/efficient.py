from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor


class EfficientInteractionDownProjection(nn.Module):
    """Down projection in the efficient reformulation.

    Args:
        emb_size_interm (int): Intermediate embedding size (down-projection size).
    """

    def __init__(self, n_spherical: int, n_radial: int, emb_size_interm: int, weight_init: Callable[[Tensor], Tensor]):
        super().__init__()

        self.n_spherical = n_spherical
        self.n_radial = n_radial
        self.emb_size_interm = emb_size_interm
        self.wi = weight_init

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = nn.Parameter(
            torch.empty((self.n_spherical, self.n_radial, self.emb_size_interm)),
            requires_grad=True,
        )
        self.wi(self.weight)

    def forward(self, rbf: Tensor, sph: Tensor, id_ca: Tensor, id_ragged_idx: Tensor):
        """
        Args:
            rbf (torch.Tensor): shape=(1, E, n_radial)
            sph (torch.Tensor): shape=(E, Kmax, n_spherical)
            id_ca (torch.Tensor): shape=(E,)
            id_ragged_idx (torch.Tensor): shape=(E,)

        Returns:
            rbf_W1 (torch.Tensor): shape=(E, emb_size_interm, n_spherical)
            sph (torch.Tensor): shape=(E, Kmax, n_spherical)
                Kmax = maximum number of neighbors of the edges
        """
        E = rbf.size(1)

        # MatMul: mul + sum over num_radial
        rbf_W1 = torch.matmul(rbf, self.weight)
        # (n_spherical, E, emb_size_interm)
        rbf_W1 = rbf_W1.permute(1, 2, 0)
        # (E, emb_size_interm, n_spherical)

        # Zero padded dense matrix
        # maximum number of neighbors, catch empty id_ca with maximum
        if sph.size(0) == 0:
            Kmax = 0
        else:
            Kmax = int(
                torch.max(
                    torch.max(id_ragged_idx + 1),
                    torch.tensor(0).to(id_ragged_idx.device),
                ).item()
            )

        sph2 = sph.new_zeros(E, Kmax, self.n_spherical)
        sph2[id_ca, id_ragged_idx] = sph

        sph2 = torch.transpose(sph2, 1, 2)
        # (E, n_spherical/emb_size_interm, Kmax)

        return rbf_W1, sph2


class EfficientInteractionBilinear(nn.Module):
    """Efficient reformulation of the bilinear layer and subsequent summation.

    Args:
        units_out (int): Embedding output size of the bilinear layer.
        weight_init (callable): Initializer of the weight matrix.
    """

    def __init__(
        self,
        emb_size: int,
        emb_size_interm: int,
        units_out: int,
        weight_init: Callable[[Tensor], Tensor],
    ):
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_interm = emb_size_interm
        self.units_out = units_out
        self.wi = weight_init

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.emb_size, self.emb_size_interm, self.units_out),
                requires_grad=True,
            )
        )
        self.wi(self.weight)

    def forward(self, basis: Tensor, m: Tensor, id_reduce: Tensor, id_ragged_idx: Tensor):
        """
        Args
            basis
            m: quadruplets: m = m_db , triplets: m = m_ba
            id_reduce
            id_ragged_idx

        Returns:
            m_st (torch.Tensor): shape=(E, units_out) Edge embeddings.
        """
        # n_spherical is actually n_spherical**2 for quadruplets
        (rbf_W1, sph) = basis
        # (E, emb_size_interm, n_spherical), (E, n_spherical, Kmax)
        E = rbf_W1.size(0)

        # Create (zero-padded) dense matrix of the neighboring edge embeddings.
        Kmax = int(
            torch.max(
                torch.max(id_ragged_idx) + 1,
                torch.tensor(0).to(id_ragged_idx.device),
            ).item()
        )
        # maximum number of neighbors, catch empty id_reduce_ji with maximum
        m2 = m.new_zeros(E, Kmax, self.emb_size)
        m2[id_reduce, id_ragged_idx] = m
        # (n_quadruplets or n_triplets, emb_size) -> (E, Kmax, emb_size)

        sum_k = torch.matmul(sph, m2)  # (E, n_spherical, emb_size)

        # MatMul: mul + sum over num_spherical
        rbf_W1_sum_k = torch.matmul(rbf_W1, sum_k)
        # (E, emb_size_interm, emb_size)

        # Bilinear: Sum over emb_size_interm and emb_size
        m_st = torch.matmul(rbf_W1_sum_k.permute(2, 0, 1), self.weight)
        # (emb_size, E, units_out)
        m_st = torch.sum(m_st, dim=0)
        # (E, units_out)

        return m_st
