from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_scatter import scatter

from ..data.keys import GraphKeys
from ..nn.base import Dense, ResidualLayer
from ..nn.basis import (
    SphericalBesselFunction,
    SphericalHarmonicsFunction,
    combine_sbb_shb,
)
from ..nn.cutoff import BaseCutoff
from ..nn.scaling import ScaleFactor
from ..utils.resolve import activation_resolver, cutoffnet_resolver, init_resolver
from .base import BaseMPNN


class InvarianceSphereNet(BaseMPNN):
    def __init__(
        self,
        emb_size: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        n_neighbor_basis: int,
        n_blocks: int,
        n_targets: int,
        max_n: int,
        max_l: int,
        rbf_smooth: bool = True,
        cutoff: float = 6.0,
        cutoff_net: str | type[BaseCutoff] = "envelope",
        cutoff_kwargs: dict[str, Any] = {},
        n_residual_output: int = 2,
        max_z: int | None = None,
        extensive: bool = True,
        regress_forces: bool = True,
        direct_forces: bool = True,
        activation: str | nn.Module = "scaledsilu",
        weight_init: str | Callable[[Tensor], Tensor] | None = None,
        scale_file: str | None = None,
    ):
        super().__init__()
        act = activation_resolver(activation)
        wi = init_resolver(weight_init) if weight_init is not None else None

        self.n_blocks = n_blocks
        self.n_targets = n_targets
        self.max_n = max_n
        self.max_l = max_l
        self.rbf_smooth = rbf_smooth
        self.extensive = extensive
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces

        # basis layers
        self.rbf = SphericalBesselFunction(max_l, max_n, cutoff, rbf_smooth)
        self.cbf = SphericalHarmonicsFunction(max_l, False)
        self.sbf = SphericalHarmonicsFunction(max_l, True)
        cutoff_kwargs["cutoff"] = cutoff
        self.cn = cutoffnet_resolver(cutoff_net, **cutoff_kwargs)

        # shared layers
        self.mlp_rbf = Dense(max_n, emb_size_rbf, bias=False, weight_init=wi)
        self.mlp_rbf_proj = Dense(max_n, emb_size_rbf, bias=False, weight_init=wi)
        self.mlp_cbf = Dense(max_n * max_l, emb_size_cbf, bias=False, weight_init=wi)
        self.mlp_cbf_proj = Dense(max_n * max_l, emb_size_cbf, bias=False, weight_init=wi)
        self.mlp_sbf = Dense(max_n * max_l * max_l, emb_size_sbf, bias=False, weight_init=wi)
        self.mlp_rbf_h = Dense(max_n, emb_size_rbf, bias=False, weight_init=wi)
        self.mlp_rbf_out = Dense(max_n, emb_size_rbf, bias=False, weight_init=wi)

        # embedding block
        self.emb_block = EmbeddingBlock(emb_size, emb_size, emb_size, max_z, True, act, wi)

        # interaction and output blocks
        self.int_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    emb_size,
                    emb_size,
                    emb_size_rbf,
                    emb_size_cbf,
                    emb_size_sbf,
                    n_neighbor_basis,
                    n_before_skip=1,
                    n_after_skip=1,
                    n_after_atom_self=1,
                    n_atom_emb=1,
                    activation=act,
                    weight_init=wi,
                )
                for _ in range(n_blocks)
            ]
        )
        self.out_blocks = nn.ModuleList(
            [
                OutputBlock(emb_size, emb_size, emb_size_rbf, n_residual_output, n_targets, direct_forces, act, wi)
                for _ in range(n_blocks + 1)
            ]
        )

    def forward(self, graph: Batch):
        if self.regress_forces and not self.direct_forces:
            graph[GraphKeys.Pos].requires_grad_(True)

        graph = self.calc_atomic_distances(graph, return_vec=True)
        graph = self.rot_transform(graph)

        z: Tensor = graph[GraphKeys.Z]
        r_ij: Tensor = graph[GraphKeys.Edge_dist]
        if graph.get(GraphKeys.Batch_idx):
            batch_idx: Tensor = graph[GraphKeys.Batch_idx]
        else:
            batch_idx = torch.zeros_like(z, dtype=torch.long, device=z.device)
        # order is "source_to_target" i.e. [index_j, index_i]
        idx: Tensor = graph[GraphKeys.Edge_idx]
        idx_j = idx[0]
        idx_i = idx[1]
        idx_swap: Tensor = graph[GraphKeys.Edge_idx_swap]

        # ---------- Basis layers ----------
        # rbf
        rbf = self.rbf(r_ij)  # (E, n_rbf)
        rbf_h = self.mlp_rbf_h(rbf)  # (E, emb_size_rbf)
        rbf_out = self.mlp_rbf_out(rbf)  # (E, emb_size_rbf)
        rbf_mp = self.mlp_rbf(rbf)  # (E, emb_size_rbf)

        # cbf & sbf
        phi: Tensor = graph[GraphKeys.Phi]  # (NB, E)
        theta: Tensor = graph[GraphKeys.Theta]  # (NB, E)
        NB, E = phi.size()

        r_ij_proj = torch.sin(phi) * r_ij[None, ...]  # (NB, E)
        rbf_proj = self.rbf(r_ij_proj.view(-1)).view(NB, E, -1)  # (NB, E, n_rbf)
        rbf_proj_mp = self.mlp_rbf_proj(rbf_proj)  # (NB, E, emb_size_rbf)
        theta, phi = theta.view(-1), phi.view(-1)
        cbf = self.cbf(theta).view(NB, E, -1)  # (NB, E, n_cbf)
        cbf = combine_sbb_shb(rbf, cbf, self.max_n, self.max_l, False)  # (NB, E, max_n*max_l)
        cbf_proj = combine_sbb_shb(rbf_proj, cbf, self.max_n, self.max_l, False)  # (NB, E, max_n*max_l)
        cbf_mp = self.mlp_cbf(cbf)  # (NB, E, emb_size_cbf)
        cbf_proj_mp = self.mlp_cbf_proj(cbf_proj)  # (NB, E, emb_size_cbf)

        sbf = self.sbf(theta, phi).view(NB, E, -1)  # (NB, E, n_sbf)
        sbf = combine_sbb_shb(rbf, sbf, self.max_n, self.max_l, True)  # (NB, E, max_n*max_l**2)
        sbf_mp = self.mlp_sbf(sbf)  # (NB, E, emb_size_sbf)

        # ---------- EmbeddingBlock and OutputBlock----------
        # (N, emb_size) & (E, emb_size)
        h, m_ij = self.emb_block(z, rbf, idx_i, idx_j)
        # (B, n_targets) & (E, n_targets)
        E_i, F_ij = self.out_blocks[0](h, m_ij, rbf_out, idx_i)

        # ---------- InteractionBlock and OutputBlock ----------
        for i in range(self.n_blocks):
            # interacton
            h, m_ij = self.int_blocks[i](
                h,
                m_ij,
                rbf_h,
                rbf_mp,
                rbf_proj_mp,
                cbf_mp,
                cbf_proj_mp,
                sbf_mp,
                idx_i,
                idx_j,
                idx_swap,
            )

            # output
            E, F = self.out_blocks[i](h, m_ij, rbf_out, idx_i)
            E_i += E
            F_ij += F

        # ---------- Output calculation ----------
        B = torch.max(batch_idx) + 1
        # (B, n_targets)
        if self.extensive:
            E_i = scatter(E_i, batch_idx, dim=0, dim_size=B, reduce="add")
        else:
            E_i = scatter(E_i, batch_idx, dim=0, dim_size=B, reduce="mean")

        if self.regress_forces:
            if self.direct_forces:
                # map forces in edge directions
                dir_ij = graph[GraphKeys.Edge_dir_ij]
                F_ij = F_ij[:, :, None] * dir_ij[:, None, :]  # (E, n_targets, 3)
                F_ij = scatter(F_ij, idx_i, dim=0, dim_size=z.size(0), reduce="add")  # (N, n_targets, 3)
                F_ij = F_ij.squeeze(1)  # (N, 3)
            else:
                if self.n_targets > 1:
                    # maybe this can be solved differently
                    F_ij = torch.stack(
                        [
                            -torch.autograd.grad(E_i[:, i].sum(), graph[GraphKeys.Pos], create_graph=True)[0]
                            for i in range(self.n_targets)
                        ],
                        dim=1,
                    )  # (N, n_targets, 3)
                else:
                    F_ij = -torch.autograd.grad(E_i.sum(), graph[GraphKeys.Pos], create_graph=True)[0]  # (N, 3)

                graph[GraphKeys.Pos].requires_grad = False

        return E_i, F_ij


class EmbeddingBlock(nn.Module):
    """Atom and Edge embedding with atomic number and RBF of edge ji."""

    def __init__(
        self,
        atom_features: int,
        edge_features: int,
        out_features: int,
        max_z: int | None,
        bias: bool,
        activation: nn.Module,
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            atom_features (int): Embedding size of the atom embeddings.
            edge_features (int): Embedding size of the edge embeddings.
            out_features(int): Embedding size after the embedding block.
            max_z (int | None): Maximum atomic number. If `None`, set 93 as max_z.
            bias (bool): Whether to use bias term.
            activation (nn.Module): Activation function.
            weight_init (Callble[[Tensor], Tensor] | None): weight init function.
        """
        super().__init__()
        if max_z is None:
            max_z = 93
        self.atom_embedding = nn.Embedding(max_z, atom_features)
        in_features = 2 * atom_features + edge_features
        self.mlp_embed = nn.Sequential(
            Dense(in_features, out_features, bias, weight_init),
            activation,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.atom_embedding.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))

    def forward(self, z: Tensor, rbf: Tensor, idx_i: Tensor, idx_j: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            z (torch.Tensor): Atomic number with (N) shape.
            rbf (torch.Tensor): RBF of edge ji with (E) shape.
            idx_i (torch.Tensor): Edge index of target atom i with (E) shape.
            idx_j (torch.Tensor): Edge index of source atom j with (E) shape.

        Returns:
            h (torch.Tensor): Atom embedding with (N, atom_features) shape.
            m_ij (torch.Tensor): Edge embedding with (E, edge_features) shape.
        """
        h = self.atom_embedding(z - 1)
        h_i = h[idx_i]  # (E, atom_features)
        h_j = h[idx_j]  # (E, atom_features)

        m_ij = torch.cat([h_i, h_j, rbf], dim=-1)  # (E, 2*atom_features+edge_features)
        m_ij = self.mlp_embed(m_ij)  # (E, out_features)
        return h, m_ij


class InteractionBlock(nn.Module):
    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        n_neighbor_basis: int,
        n_before_skip: int,
        n_after_skip: int,
        n_after_atom_self: int,
        n_atom_emb: int,
        activation: nn.Module,
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()

        # ---------- Geometric MP ----------
        self.mlp_ij = Dense(emb_size_edge, emb_size_edge, False, weight_init=weight_init)
        self.q_mp = QuadrupletInteraction(
            emb_size_edge,
            emb_size_rbf,
            emb_size_cbf,
            emb_size_sbf,
            n_neighbor_basis,
            activation,
            weight_init,
        )
        self.t_mp = TripletInteraction(
            emb_size_edge,
            emb_size_rbf,
            emb_size_cbf,
            n_neighbor_basis,
            activation,
            weight_init,
        )

        # ---------- Update Edge Embeddings ----------
        # Residual layers before skip connection
        self.residual_before_skip = nn.ModuleList(
            [ResidualLayer(emb_size_edge, 2, False, activation, weight_init) for _ in range(n_before_skip)]
        )
        # Residual layers after skip connection
        self.residual_after_skip = torch.nn.ModuleList(
            [ResidualLayer(emb_size_edge, 2, False, activation, weight_init) for _ in range(n_after_skip)]
        )

        # ---------- Update Atom Embeddings ----------
        self.atom_emb = AtomEmbedding(
            emb_size_atom,
            emb_size_edge,
            emb_size_rbf,
            n_residual=n_atom_emb,
            activation=activation,
            weight_init=weight_init,
        )

        # ---------- Update Edge Embeddings with Atom Embeddings ----------
        self.atom_self_interaction = AtomSelfInteracion(emb_size_atom, emb_size_edge, activation, weight_init)
        self.residual_m = nn.ModuleList(
            [ResidualLayer(emb_size_edge, 2, False, activation, weight_init) for _ in range(n_after_atom_self)]
        )

        self.inv_sqrt_2 = 1 / (2.0**0.5)
        self.inv_sqrt_3 = 1 / (3.0**0.5)

    def forward(
        self,
        h: Tensor,
        m_ij: Tensor,
        rbf_h: Tensor,
        rbf: Tensor,
        rbf_proj: Tensor,
        cbf: Tensor,
        cbf_proj: Tensor,
        sbf: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        idx_swap: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # ---------- Geometric MP ----------
        # Initial transformation
        x_ij_skip = self.mlp_ij(m_ij)  # (E, emb_size_edge)

        x4 = self.q_mp(m_ij, rbf, cbf, sbf, idx_swap)
        x3 = self.t_mp(m_ij, rbf_proj, cbf_proj, idx_swap)

        # ---------- Merge Embeddings after Quadruplet and Triplet Interaction ----------
        x = x_ij_skip + x3 + x4  # (E, emb_size_edge)
        x = x * self.inv_sqrt_3

        # ---------- Update Edge Embeddings ----------
        for layer in self.residual_before_skip:
            x = layer(x)  # (E, emb_size_edge)

        # Skip connection
        m_ij = m_ij + x  # (E, emb_size_edge)
        m_ij = m_ij * self.inv_sqrt_2

        for layer in self.residual_after_skip:
            m_ij = layer(m_ij)  # (E, emb_size_edge)

        # ---------- Update Atom Embeddings ----------
        h2 = self.atom_emb(h, m_ij, rbf_h, idx_i)  # (N, emb_size_atom)

        # Skip connection
        h = h + h2  # (N, emb_size_atom)
        h = h * self.inv_sqrt_2

        # ---------- Update Edge Embeddings with Atom Embeddings ----------
        m2 = self.atom_self_interaction(h, m_ij, idx_i, idx_j)  # (E, emb_size_edge)

        for layer in self.residual_m:
            m2 = layer(m2)  # (E, emb_size_edge)

        # Skip connection
        m_ij = m_ij + m2  # (E, emb_size_edge)
        m_ij = m_ij * self.inv_sqrt_2

        return h, m_ij


class OutputBlock(nn.Module):
    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        n_residual: int,
        n_targets: int,
        direct_forces: bool,
        activation: nn.Module,
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        self.emb_size_atom = emb_size_atom
        self.emb_size_edge = emb_size_edge
        self.emb_size_rbf = emb_size_rbf
        self.n_residual = n_residual
        self.n_targets = n_targets

        self.direct_forces = direct_forces
        self.dense_rbf = Dense(emb_size_rbf, emb_size_edge, bias=False, weight_init=weight_init)

        self.mlp_energy = self.get_mlp(emb_size_atom, n_residual, activation, weight_init)
        self.scale_sum = ScaleFactor()
        # do not add bias to final layer to enforce that prediction for an atom
        # without any edge embeddings is zero
        self.mlp_out_energy = Dense(emb_size_atom, n_targets, bias=False, weight_init=weight_init)

        if self.direct_forces:
            self.scale_rbf = ScaleFactor()
            self.mlp_forces = self.get_mlp(emb_size_edge, n_residual, activation, weight_init)
            # no bias in final layer to ensure continuity
            self.mlp_out_forces = Dense(emb_size_edge, n_targets, bias=False, weight_init=weight_init)

    def get_mlp(
        self,
        units: int,
        n_residual: int,
        activation: nn.Module,
        weight_init: Callable[[Tensor], Tensor] | None,
    ):
        mlp: list[nn.Module] = []
        mlp.append(Dense(self.emb_size_edge, units, False, weight_init))
        mlp.append(activation)
        for _ in range(n_residual):
            mlp.append(ResidualLayer(units, n_layers=2, activation=activation, bias=False))
        return nn.ModuleList(mlp)

    def forward(self, h: Tensor, m_ij: Tensor, rbf: Tensor, idx_i: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            h (torch.Tensor): Atom embedding with (N, emb_size_atom) shape.
            m_ij (torch.Tensor): edge embedding of ji with (E, emb_size_edge) shape.
            rbf (torch.Tensor): RBF of ji with (E, emb_size_rbf) shape.
            idx_i (torch.Tensor): Edge index of target atom i with (E) shape.

        Returns:
            E (torch.Tensor): Output energy with (N, n_targets) shape.
            F (torch.Tensor): Output forces with (E, n_targets) shape.
        """
        N = h.size(0)

        rbf_mlp = self.dense_rbf(rbf)  # (E, emb_size_edge)
        x = m_ij * rbf_mlp

        # ---------- Energy Prediction ----------
        x_E = scatter(x, idx_i, dim=0, dim_size=N, reduce="add")  # (N, emb_size_edge)
        x_E = self.scale_sum(x_E, ref=m_ij)

        for layer in self.mlp_energy:
            x_E = layer(x_E)  # (N, emb_size_atom)

        x_E = self.mlp_out_energy(x_E)  # (N, num_targets)

        # ---------- Force Prediction ----------
        if self.direct_forces:
            x_F = self.scale_rbf(x, ref=m_ij)

            for layer in self.mlp_forces:
                x_F = layer(x_F)  # (E, emb_size_edge)

            x_F = self.mlp_out_forces(x_F)  # (E, num_targets)

        else:
            x_F = 0

        return x_E, x_F


class QuadrupletInteraction(nn.Module):
    def __init__(
        self,
        emb_size_edge: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        n_neighbor_basis: int,
        activation: nn.Module,
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()

        self.mlp_m_rbf = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_rbf = Dense(emb_size_rbf, emb_size_edge, bias=False, weight_init=weight_init)
        self.scale_rbf = ScaleFactor()

        self.mlp_m_cbf = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_cbf = Dense(emb_size_cbf, emb_size_edge, bias=False, weight_init=weight_init)
        self.scale_cbf = ScaleFactor()

        self.mlp_m_sbf = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_sbf = Dense(emb_size_sbf, emb_size_edge, bias=False, weight_init=weight_init)
        self.scale_sbf = ScaleFactor()

        self.mlp_direction = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, False, weight_init),
            activation,
        )

        self.mlp_ij = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_ji = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )

        self.inv_sqrt_2 = 1 / (2.0**0.5)
        self.inv_sqrt_neighbor = 1 / (n_neighbor_basis**0.5)

    def forward(
        self,
        m_ij: Tensor,
        rbf: Tensor,
        cbf: Tensor,
        sbf: Tensor,
        idx_swap: Tensor,
    ) -> Tensor:
        # ---------- Geometric MP ----------
        NB, E, _ = rbf.size()
        m_ij = m_ij.unsqueeze(0).expand(NB, E, -1)  # (NB, E, emb_size_edge)

        m_ij = self.mlp_m_rbf(m_ij)
        m_ij2 = m_ij * self.mlp_rbf(rbf)
        m_ij = self.scale_rbf(m_ij2, ref=m_ij)  # (NB, E, emb_size_edge)

        m_ij = self.mlp_m_cbf(m_ij)
        m_ij2 = m_ij * self.mlp_cbf(cbf)
        m_ij = self.scale_cbf(m_ij2, m_ij)  # (NB, E, emb_size_edge)

        m_ij = self.mlp_m_sbf(m_ij)
        m_ij2 = m_ij * self.mlp_sbf(sbf)
        m_ij = self.scale_sbf(m_ij2, m_ij)  # (NB, E, emb_size_edge)

        # ---------- Basis MP ----------
        # x = torch.stack([m_ij[i] for i in range(self.n_neighbor_basis)], dim=-1)
        x = m_ij.sum(0)  # (E, emb_size_edge)
        x = x * self.inv_sqrt_neighbor
        x = self.mlp_direction(x)

        # ---------- Update embeddings ----------
        x_ij = self.mlp_ij(x)  # (E, emb_size_edge)
        x_ji = self.mlp_ji(x)  # (E, emb_size_edge)

        # Merge interaction of i->j and j->i
        x_ji = x_ji[idx_swap]  # swap to add to edge j->i and not i->j
        x4 = x_ji + x_ij
        x4 = x4 * self.inv_sqrt_2

        return x4


class TripletInteraction(nn.Module):
    def __init__(
        self,
        emb_size_edge: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        n_neighbor_basis: int,
        activation: nn.Module,
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()

        self.mlp_m_rbf = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_rbf = Dense(emb_size_rbf, emb_size_edge, bias=False, weight_init=weight_init)
        self.scale_rbf = ScaleFactor()

        self.mlp_m_cbf = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_cbf = Dense(emb_size_cbf, emb_size_edge, bias=False, weight_init=weight_init)
        self.scale_cbf = ScaleFactor()

        self.mlp_direction = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, False, weight_init),
            activation,
        )

        self.mlp_ij = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_ji = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )

        self.inv_sqrt_2 = 1 / (2.0**0.5)
        self.inv_sqrt_neighbor = 1 / (n_neighbor_basis**0.5)

    def forward(
        self,
        m_ij: Tensor,
        rbf: Tensor,
        cbf: Tensor,
        idx_swap: Tensor,
    ) -> Tensor:
        # ---------- Geometric MP ----------
        NB, E, _ = rbf.size()
        m_ij = m_ij.unsqueeze(0).expand(NB, E, -1)  # (NB, E, emb_size_edge)

        m_ij = self.mlp_m_rbf(m_ij)
        m_ij2 = m_ij * self.mlp_rbf(rbf)
        m_ij = self.scale_rbf(m_ij2, ref=m_ij)  # (NB, E, emb_size_edge)

        m_ij = self.mlp_m_cbf(m_ij)
        m_ij2 = m_ij * self.mlp_cbf(cbf)
        m_ij = self.scale_cbf(m_ij2, ref=m_ij)  # (NB, E, emb_size_edge)

        # ---------- Basis MP ----------
        # x = torch.stack([m_ij[i] for i in range(self.n_neighbor_basis)], dim=-1)
        x = m_ij.sum(0)  # (E, emb_size_edge)
        x = x * self.inv_sqrt_neighbor
        x = self.mlp_direction(x)

        # ---------- Update embeddings ----------
        x_ij = self.mlp_ij(x)  # (E, emb_size_edge)
        x_ji = self.mlp_ji(x)  # (E, emb_size_edge)

        # Merge interaction of i->j and j->i
        x_ji = x_ji[idx_swap]  # swap to add to edge j->i and not i->j
        x3 = x_ji + x_ij
        x3 = x3 * self.inv_sqrt_2

        return x3


class AtomEmbedding(nn.Module):
    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        n_residual: int,
        activation: nn.Module,
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        self.emb_size_atom = emb_size_atom
        self.emb_size_edge = emb_size_edge
        self.emb_size_rbf = emb_size_rbf
        self.n_residual = n_residual

        self.mlp_rbf = Dense(emb_size_rbf, emb_size_edge, False, weight_init)
        self.scale_sum = ScaleFactor()

        self.mlp = self.get_mlp(emb_size_atom, n_residual, activation, weight_init)

    def get_mlp(
        self,
        units: int,
        n_residual: int,
        activation: nn.Module,
        weight_init: Callable[[Tensor], Tensor] | None,
    ):
        mlp: list[nn.Module] = []
        mlp.append(Dense(self.emb_size_edge, units, False, weight_init))
        mlp.append(activation)
        for _ in range(n_residual):
            mlp.append(ResidualLayer(units, n_layers=2, activation=activation, bias=False))
        return nn.ModuleList(mlp)

    def forward(self, h: Tensor, m_ij: Tensor, rbf: Tensor, idx_i: Tensor) -> Tensor:
        """
        Args:
            h (torch.Tensor): Atom embedding with (N, emb_size_atom) shape.
            m_ij (torch.Tensor): edge embedding of ji with (E, emb_size_edge) shape.
            rbf (torch.Tensor): RBF of ji with (E, emb_size_rbf) shape.
            idx_i (torch.Tensor): Edge index of target atom i with (E) shape.

        Returns:
            h (torch.Tensor): Updated atom embedding with (N, emb_size_atom) shape.
        """
        N = h.size(0)

        mlp_rbf = self.mlp_rbf(rbf)  # (E, emb_size_edge)
        x = m_ij * mlp_rbf

        x2 = scatter(x, idx_i, dim=0, dim_size=N, reduce="add")
        x = self.scale_sum(x2, ref=m_ij)  # (N, emb_size_edge)

        for layer in self.mlp:
            x = layer(x)  # (N, emb_size_atom)
        return x


class AtomSelfInteracion(nn.Module):
    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        activation: nn.Module,
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        in_size = 2 * emb_size_atom + emb_size_edge
        self.mlp_embed = nn.Sequential(
            Dense(in_size, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )

    def forward(self, h: Tensor, m_ij: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        """
        Args:
            h (torch.Tensor): Atom embedding with (N, emb_size_atom) shape.
            m_ij (torch.Tensor): edge embedding of ji with (E, emb_size_edge) shape.
            idx_i (torch.Tensor): Edge index of target atom i with (E) shape.
            idx_j (torch.Tensor): Edge index of source atom j with (E) shape.

        Returns:
            m_ij (torch.Tensor): Updated edge embedding with (E, emb_size_edge) shape.
        """
        h_i = h[idx_i]
        h_j = h[idx_j]

        m_ij = torch.cat([h_i, h_j, m_ij], dim=-1)  # (E, 2*emb_size_atom+emb_size_edge)
        m_ij = self.mlp_embed(m_ij)  # (E, emb_size_edge)
        return m_ij
