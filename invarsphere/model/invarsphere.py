from __future__ import annotations

import copy
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
from ..nn.basis import SphericalBesselFunction, SphericalHarmonicsWithBessel
from ..nn.cutoff import BaseCutoff
from ..nn.scaling.compat import load_scales_compat
from ..nn.scaling.scale_factor import ScaleFactor
from ..utils.calc import repeat_blocks
from ..utils.resolve import activation_resolver, cutoffnet_resolver, init_resolver
from .base import BaseMPNN


class InvarianceSphereNet(BaseMPNN):
    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        emb_quad: int,
        emb_triplet: int,
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
        align_initial_weight: bool = True,
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
        self.align_initial_weight = align_initial_weight

        # basis layers
        self.rbf = SphericalBesselFunction(max_n, max_l, cutoff, rbf_smooth)
        self.cbf = SphericalHarmonicsWithBessel(max_n, max_l, cutoff, use_phi=False)
        self.sbf = SphericalHarmonicsWithBessel(max_n, max_l, cutoff, use_phi=True)
        cutoff_kwargs["cutoff"] = cutoff
        self.cutoff = cutoffnet_resolver(cutoff_net, **cutoff_kwargs)

        # shared layers
        self.mlp_rbf_h = Dense(max_n, emb_size_rbf, bias=False, weight_init=wi)
        self.mlp_rbf_out = Dense(max_n, emb_size_rbf, bias=False, weight_init=wi)
        self.mlp_rbf3 = Dense(max_n, emb_size_rbf, bias=False, weight_init=wi)
        self.mlp_rbf4 = Dense(max_n, emb_size_rbf, bias=False, weight_init=wi)
        self.mlp_cbf3 = Dense(max_n * max_l, emb_size_cbf, bias=False, weight_init=wi)
        self.mlp_cbf4 = Dense(max_n * max_l, emb_size_cbf, bias=False, weight_init=wi)
        self.mlp_sbf4 = Dense(max_n * max_l * max_l, emb_size_sbf, bias=False, weight_init=wi)

        # embedding block
        self.emb_block = EmbeddingBlock(emb_size_atom, max_n, emb_size_edge, max_z, True, act, wi)

        # interaction and output blocks
        self.int_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    emb_size_atom,
                    emb_size_edge,
                    emb_size_rbf,
                    emb_size_cbf,
                    emb_size_sbf,
                    emb_quad,
                    emb_triplet,
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
                OutputBlock(
                    emb_size_atom,
                    emb_size_edge,
                    emb_size_rbf,
                    n_residual_output,
                    n_targets,
                    regress_forces,
                    direct_forces,
                    activation=act,
                    weight_init=wi,
                )
                for _ in range(n_blocks + 1)
            ]
        )

        if align_initial_weight:
            int_state_dict = self.int_blocks[0].state_dict()
            out_state_dict = self.out_blocks[0].state_dict()
            for i in range(1, n_blocks):
                self.int_blocks[i].load_state_dict(copy.deepcopy(int_state_dict))
                self.out_blocks[i].load_state_dict(copy.deepcopy(out_state_dict))
            self.out_blocks[n_blocks].load_state_dict(copy.deepcopy(out_state_dict))

        load_scales_compat(self, scale_file)

    def _select_symmetric_edges(self, tensor: Tensor, mask: Tensor, reorder_idx: Tensor, inverse_neg: bool) -> Tensor:
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def _reorder_symmetric_edges(
        self,
        edge_index: Tensor,
        cell_offsets: Tensor,
        neighbors: Tensor,
        edge_dist: Tensor,
        edge_vector: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data, since
        every atom has a maximum number of neighbors. Since we only use
        i->j edges here, we lose some j->i edges and add others by
        making it symmetric. We could fix this by merging edge_index
        with its counter-edges, including the cell_offsets, and then
        running torch.unique. But this does not seem worth it.
        """
        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] == 0) & (cell_offsets[:, 2] < 0))
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(batch_edge, minlength=neighbors.size(0))

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self._select_symmetric_edges(cell_offsets, mask, edge_reorder_idx, True)
        edge_dist_new = self._select_symmetric_edges(edge_dist, mask, edge_reorder_idx, False)
        edge_vector_new = self._select_symmetric_edges(edge_vector, mask, edge_reorder_idx, True)

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
        )

    def _rot_transform(self, rot_mat: Tensor, vec_ij: Tensor, idx_s: Tensor) -> tuple[Tensor, Tensor]:
        """Cartesian to polar transform of edge vector.

        Args:
            rot_mat (torch.Tensor): atom rotation matrix with (N, NB, 3, 3) shape.
            vec_ij (torch.Tensor): cartesian edge vector with (E, 3) shape.
            idx_s (torch.Tensor): edge index of source atom with (E) shape.

        Returns:
            theta (torch.Tensor): the azimuthal angle with (E, NB) shape.
            phi (torch.Tensor): the polar angle with (E, NB) shape.
        """
        rot_mat = rot_mat[idx_s]  # (E, NB, 3, 3)

        # ---------- rotation transform ----------
        rot_vec = torch.einsum("ebnm,em->ebn", rot_mat, vec_ij)  # (E, NB, 3)

        # ---------- cart to polar transform ----------
        rot_vec = rot_vec / rot_vec.norm(dim=-1, keepdim=True)
        # Define azimuthal angle as counterclockwise from the y-axis of the second proximity
        theta = torch.atan2(rot_vec[..., 2], rot_vec[..., 1])  # (E, NB)
        # The angle of first proximity is the polar angle
        phi = torch.acos(rot_vec[..., 0])  # (E, NB)

        return theta, phi  # (E, NB)

    def generate_interaction_graph(self, graph: Batch) -> Batch:
        graph = self.calc_atomic_distances(graph, return_vec=True)

        edge_index: Tensor = graph[GraphKeys.Edge_idx]
        cell_offsets: Tensor = graph[GraphKeys.Edge_shift]
        neighbors: Tensor = graph[GraphKeys.Neighbors]
        d_st: Tensor = graph[GraphKeys.Edge_dist_st]
        v_st: Tensor = graph[GraphKeys.Edge_vec_st]
        (
            edge_index,
            cell_offsets,
            neighbors,
            d_st,
            v_st,
        ) = self._reorder_symmetric_edges(edge_index, cell_offsets, neighbors, d_st, v_st)
        graph[GraphKeys.Edge_idx] = edge_index
        graph[GraphKeys.Edge_shift] = cell_offsets
        graph[GraphKeys.Neighbors] = neighbors
        graph[GraphKeys.Edge_dist_st] = d_st
        graph[GraphKeys.Edge_vec_st] = v_st

        # Indices for swapping c->a and a->c (for symmetric MP)
        block_sizes = neighbors // 2
        idx_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )
        graph[GraphKeys.Edge_idx_swap] = idx_swap

        theta, phi = self._rot_transform(
            graph[GraphKeys.Rot_mat],
            v_st,
            edge_index[0],  # order is "source_to_target"
        )
        graph[GraphKeys.Theta] = theta
        graph[GraphKeys.Phi] = phi

        return graph

    def forward(self, graph: Batch) -> tuple[Tensor, Tensor]:
        if self.regress_forces and not self.direct_forces:
            graph[GraphKeys.Pos].requires_grad_(True)

        graph = self.generate_interaction_graph(graph)

        z: Tensor = graph[GraphKeys.Z]
        d_st: Tensor = graph[GraphKeys.Edge_dist_st]
        if graph.get(GraphKeys.Batch_idx) is None:
            graph[GraphKeys.Batch_idx] = z.new_zeros(z.size(0), dtype=torch.long)
        batch_idx: Tensor = graph[GraphKeys.Batch_idx]
        # order is "source_to_target"
        idx: Tensor = graph[GraphKeys.Edge_idx]
        idx_s = idx[0]
        idx_t = idx[1]
        idx_swap: Tensor = graph[GraphKeys.Edge_idx_swap]
        basis_idx1: Tensor = graph[GraphKeys.Basis_edge_idx1]
        basis_idx2: Tensor = graph[GraphKeys.Basis_edge_idx2]

        # ---------- Basis layers ----------
        # --- rbf ---
        rbf = self.rbf(d_st)  # (E, max_n)
        rbf3 = self.mlp_rbf3(rbf)  # (E, emb_size_rbf)
        rbf4 = self.mlp_rbf4(rbf)  # (E, emb_size_rbf)
        rbf_h = self.mlp_rbf_h(rbf)  # (E, emb_size_rbf)
        rbf_out = self.mlp_rbf_out(rbf)  # (E, emb_size_rbf)

        # --- cbf & sbf ---
        phi: Tensor = graph[GraphKeys.Phi]  # (E, NB)
        theta: Tensor = graph[GraphKeys.Theta]  # (E, NB)

        # expand with NB dimension
        E, NB = phi.size()
        d_st = d_st.unsqueeze(1).expand(E, NB)  # (E, NB)
        # reshape to calculate basis
        d_st, theta, phi = d_st.flatten(), theta.flatten(), phi.flatten()
        # cbf
        # phi is the angle with the first proximity edge
        cbf3 = self.cbf(d_st, phi).view(E, NB, -1)  # (E, NB, max_n*max_l)
        cbf3 = self.mlp_cbf3(cbf3)  # (E, NB, emb_size_cbf)
        # theta is the angle between m_st and the plane made by the first and second proximity
        cbf4 = self.cbf(d_st, theta).view(E, NB, -1)  # (E, NB, max_n*max_l)
        cbf4 = self.mlp_cbf4(cbf4)  # (E, NB, emb_size_cbf)
        # sbf
        sbf4 = self.sbf(d_st, phi, theta).view(E, NB, -1)  # (E, NB, max_n*max_l*max_l)
        sbf4 = self.mlp_sbf4(sbf4)  # (E, NB, emb_size_sbf)

        # ---------- EmbeddingBlock and OutputBlock----------
        # (N, emb_size) & (E, emb_size)
        h, m_st = self.emb_block(z, rbf, idx_s, idx_t)
        # (B, n_targets) & (E, n_targets)
        E_t, F_st = self.out_blocks[0](h, m_st, rbf_out, idx_s)

        # ---------- InteractionBlock and OutputBlock ----------
        for i in range(self.n_blocks):
            # interacton
            h, m_st = self.int_blocks[i](
                h,
                m_st,
                rbf_h,
                rbf3,
                rbf4,
                cbf3,
                cbf4,
                sbf4,
                idx_s,
                idx_t,
                idx_swap,
                basis_idx1,
                basis_idx2,
            )

            # output
            E, F = self.out_blocks[i](h, m_st, rbf_out, idx_t)
            E_t += E
            F_st += F

        # ---------- Output calculation ----------
        B = torch.max(batch_idx) + 1
        # (B, n_targets)
        if self.extensive:
            E_b = scatter(E_t, batch_idx, dim=0, dim_size=B, reduce="add")
        else:
            E_b = scatter(E_t, batch_idx, dim=0, dim_size=B, reduce="mean")

        if self.regress_forces:
            if self.direct_forces:
                # map forces in edge directions
                dir_st = graph[GraphKeys.Edge_vec_st] / graph[GraphKeys.Edge_vec_st].norm(dim=-1, keepdim=True)
                F_st = F_st[:, :, None] * dir_st[:, None, :]  # (E, n_targets, 3)
                F_n = scatter(F_st, idx_t, dim=0, dim_size=z.size(0), reduce="add")  # (N, n_targets, 3)
                F_n = F_n.squeeze(1)  # (N, 3)
            else:
                if self.n_targets > 1:
                    # maybe this can be solved differently
                    F_n = torch.stack(
                        [
                            -torch.autograd.grad(E_b[:, i].sum(), graph[GraphKeys.Pos], create_graph=True)[0]
                            for i in range(self.n_targets)
                        ],
                        dim=1,
                    )  # (N, n_targets, 3)
                else:
                    F_n = -torch.autograd.grad(E_b.sum(), graph[GraphKeys.Pos], create_graph=True)[0]  # (N, 3)

                graph[GraphKeys.Pos].requires_grad = False
        else:
            F_n = z.new_zeros(1)

        return E_b, F_n


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

    def forward(self, z: Tensor, rbf: Tensor, idx_s: Tensor, idx_t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            z (torch.Tensor): Atomic number with (N) shape.
            rbf (torch.Tensor): RBF of edge ji with (E) shape.
            idx_s (torch.Tensor): Edge index of source atom i with (E) shape.
            idx_t (torch.Tensor): Edge index of target atom j with (E) shape.

        Returns:
            h (torch.Tensor): Atom embedding with (N, atom_features) shape.
            m_st (torch.Tensor): Edge embedding with (E, edge_features) shape.
        """
        h = self.atom_embedding(z - 1)
        h_s = h[idx_s]  # (E, atom_features)
        h_t = h[idx_t]  # (E, atom_features)

        m_st = torch.cat([h_s, h_t, rbf], dim=-1)  # (E, 2*atom_features+edge_features)
        m_st = self.mlp_embed(m_st)  # (E, out_features)
        return h, m_st


class InteractionBlock(nn.Module):
    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        emb_quad: int,
        emb_triplet: int,
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
        self.mlp_st = Dense(emb_size_edge, emb_size_edge, False, weight_init=weight_init)
        self.q_mp = QuadrupletInteraction(
            emb_size_edge,
            emb_size_rbf,
            emb_size_cbf,
            emb_size_sbf,
            emb_quad,
            n_neighbor_basis,
            activation,
            weight_init,
        )
        self.t_mp = TripletInteraction(
            emb_size_edge,
            emb_size_rbf,
            emb_size_cbf,
            emb_triplet,
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
        m_st: Tensor,
        rbf_h: Tensor,
        rbf3: Tensor,
        rbf4: Tensor,
        cbf3: Tensor,
        cbf4: Tensor,
        sbf4: Tensor,
        idx_s: Tensor,
        idx_t: Tensor,
        idx_swap: Tensor,
        basis_idx1: Tensor,
        basis_idx2: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # ---------- Geometric MP ----------
        # Initial transformation
        x_st_skip = self.mlp_st(m_st)  # (E, emb_size_edge)

        x4 = self.q_mp(m_st, rbf4, cbf4, sbf4, idx_s, idx_swap, basis_idx1, basis_idx2)
        x3 = self.t_mp(m_st, rbf3, cbf3, idx_s, idx_swap, basis_idx1)

        # ---------- Merge Embeddings after Quadruplet and Triplet Interaction ----------
        x = x_st_skip + x3 + x4  # (E, emb_size_edge)
        x = x * self.inv_sqrt_3

        # ---------- Update Edge Embeddings ----------
        for layer in self.residual_before_skip:
            x = layer(x)  # (E, emb_size_edge)

        # Skip connection
        m_st = m_st + x  # (E, emb_size_edge)
        m_st = m_st * self.inv_sqrt_2

        for layer in self.residual_after_skip:
            m_st = layer(m_st)  # (E, emb_size_edge)

        # ---------- Update Atom Embeddings ----------
        h2 = self.atom_emb(h, m_st, rbf_h, idx_t)  # (N, emb_size_atom)

        # Skip connection
        h = h + h2  # (N, emb_size_atom)
        h = h * self.inv_sqrt_2

        # ---------- Update Edge Embeddings with Atom Embeddings ----------
        m2 = self.atom_self_interaction(h, m_st, idx_s, idx_t)  # (E, emb_size_edge)

        for layer in self.residual_m:
            m2 = layer(m2)  # (E, emb_size_edge)

        # Skip connection
        m_st = m_st + m2  # (E, emb_size_edge)
        m_st = m_st * self.inv_sqrt_2

        return h, m_st


class OutputBlock(nn.Module):
    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        n_residual: int,
        n_targets: int,
        regress_forces: bool,
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
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces

        self.dense_rbf = Dense(emb_size_rbf, emb_size_edge, bias=False, weight_init=weight_init)

        self.mlp_energy = self.get_mlp(emb_size_atom, n_residual, activation, weight_init)
        self.scale_sum = ScaleFactor()
        # do not add bias to final layer to enforce that prediction for an atom
        # without any edge embeddings is zero
        self.mlp_out_energy = Dense(emb_size_atom, n_targets, bias=False, weight_init=weight_init)

        if self.regress_forces and self.direct_forces:
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
            mlp.append(ResidualLayer(units, 2, False, activation=activation, weight_init=weight_init))
        return nn.ModuleList(mlp)

    def forward(self, h: Tensor, m_st: Tensor, rbf: Tensor, idx_t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            h (torch.Tensor): Atom embedding with (N, emb_size_atom) shape.
            m_st (torch.Tensor): edge embedding with (E, emb_size_edge) shape.
            rbf (torch.Tensor): RBF of ji with (E, emb_size_rbf) shape.
            idx_t (torch.Tensor): Edge index of target atom with (E) shape.

        Returns:
            E (torch.Tensor): Output energy with (N, n_targets) shape.
            F (torch.Tensor): Output forces with (E, n_targets) shape.
        """
        N = h.size(0)

        rbf_mlp = self.dense_rbf(rbf)  # (E, emb_size_edge)
        x = m_st * rbf_mlp

        # ---------- Energy Prediction ----------
        x_E = scatter(x, idx_t, dim=0, dim_size=N, reduce="add")  # (N, emb_size_edge)
        x_E = self.scale_sum(x_E, ref=m_st)

        for layer in self.mlp_energy:
            x_E = layer(x_E)  # (N, emb_size_atom)

        x_E = self.mlp_out_energy(x_E)  # (N, num_targets)

        # ---------- Force Prediction ----------
        if self.regress_forces and self.direct_forces:
            x_F = self.scale_rbf(x, ref=m_st)

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
        emb_quad: int,
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
            Dense(emb_size_edge, emb_quad, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_cbf = Dense(emb_size_cbf, emb_quad, bias=False, weight_init=weight_init)
        self.scale_cbf = ScaleFactor()

        self.mlp_m_sbf = nn.Sequential(
            Dense(emb_quad, emb_quad, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_sbf = Dense(emb_size_sbf, emb_quad, bias=False, weight_init=weight_init)
        self.scale_sbf = ScaleFactor()

        self.mlp_direction = nn.Sequential(
            Dense(emb_quad, emb_size_edge, False, weight_init),
            activation,
        )

        self.mlp_st = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_ts = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )

        self.inv_sqrt_2 = 1 / (2.0**0.5)
        self.inv_sqrt_neighbor = 1 / (n_neighbor_basis**0.5)

    def forward(
        self,
        m_st: Tensor,
        rbf: Tensor,
        cbf: Tensor,
        sbf: Tensor,
        idx_s: Tensor,
        idx_swap: Tensor,
        basis_idx1: Tensor,
        basis_idx2: Tensor,
    ) -> Tensor:
        """
        Args:
            m_st (Tensor): Edge embedding with (E, emb_size_edge) shape.
            rbf (Tensor): RBF with (E, emb_size_rbf) shape.
            cbf (Tensor): CBF with (E, NB, emb_size_cbf) shape.
            sbf (Tensor): SBF with (E, NB, emb_size_sbf) shape.
            idx_s (Tensor): index of source atom with (E) shape.
            idx_swap (Tensor): swap index of edge with (E) shape.
            basis_idx1 (Tensor): basis index of first proximity edge with (N, NB) shape.
            basis_idx2 (Tensor): basis index of second proximity edge with (N, NB) shape.

        Returns:
            x4 (Tensor): Qudruplet interaction embedding with (E, emb_size_edge) shape.
        """
        basis_idx1 = basis_idx1[idx_s].flatten()  # (E*NB)
        basis_idx2 = basis_idx2[idx_s].flatten()  # (E*NB)

        # ---------- Geometric MP ----------
        m_st = self.mlp_m_rbf(m_st)  # (E, emb_size_edge)
        m_st_rbf = m_st * self.mlp_rbf(rbf)
        m_st = self.scale_rbf(m_st_rbf, ref=m_st)  # (E, emb_size_edge)

        E, NB, _ = cbf.size()
        cbf = cbf.reshape(E * NB, -1)  # (E*NB, emb_size_cbf)
        m_st_nb: Tensor = self.mlp_m_cbf(m_st)  # (E, emb_size_edge)
        m_st_nb = m_st_nb[basis_idx1] + m_st_nb[basis_idx2]  # (E*NB, emb_size_edge)
        m_st_nb = m_st_nb * self.inv_sqrt_2
        m_st_cbf = m_st_nb * self.mlp_cbf(cbf)  # (E*NB, emb_size_edge)
        m_st_nb = self.scale_cbf(m_st_cbf, ref=m_st_nb)  # (E*NB, emb_size_edge)

        sbf = sbf.reshape(E * NB, -1)  # (E*NB, emb_size_sbf)
        m_st_nb = self.mlp_m_sbf(m_st_nb)  # (E*NB, emb_size_edge)
        m_st_sbf = m_st_nb * self.mlp_sbf(sbf)  # (E*NB, emb_size_edge)
        m_st_nb = self.scale_sbf(m_st_sbf, ref=m_st_nb)  # (E*NB, emb_size_edge)

        m_st_nb = m_st_nb.reshape(E, NB, -1)  # (E, NB, emb_size_edge)

        # ---------- Basis MP ----------
        x = m_st_nb.sum(1)  # (E, emb_size_edge)
        x = x * self.inv_sqrt_neighbor
        x = self.mlp_direction(x)

        # ---------- Update embeddings ----------
        x_st = self.mlp_st(x)  # (E, emb_size_edge)
        x_ts = self.mlp_ts(x)  # (E, emb_size_edge)

        # Merge interaction of s->t and t->s
        x_ts = x_ts[idx_swap]  # swap to add to edge s->t and not t->s
        x4 = x_st + x_ts
        x4 = x4 * self.inv_sqrt_2

        return x4


class TripletInteraction(nn.Module):
    def __init__(
        self,
        emb_size_edge: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_triplet: int,
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
            Dense(emb_size_edge, emb_triplet, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_cbf = Dense(emb_size_cbf, emb_triplet, bias=False, weight_init=weight_init)
        self.scale_cbf = ScaleFactor()

        self.mlp_direction = nn.Sequential(
            Dense(emb_triplet, emb_size_edge, False, weight_init),
            activation,
        )

        self.mlp_st = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_ts = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )

        self.inv_sqrt_2 = 1 / (2.0**0.5)
        self.inv_sqrt_neighbor = 1 / (n_neighbor_basis**0.5)

    def forward(
        self,
        m_st: Tensor,
        rbf: Tensor,
        cbf: Tensor,
        idx_s: Tensor,
        idx_swap: Tensor,
        basis_idx1: Tensor,
    ) -> Tensor:
        """
        Args:
            m_st (Tensor): Edge embedding with (E, emb_size_edge) shape.
            rbf (Tensor): RBF with (E, emb_size_rbf) shape.
            cbf (Tensor): CBF with (E, NB, emb_size_cbf) shape.
            idx_s (Tensor): index of source atom with (E) shape.
            idx_swap (Tensor): swap index of edge with (E) shape.
            basis_idx1 (Tensor): basis index of first proximity edge with (N, NB) shape.

        Returns:
            x3 (Tensor): Triplet interaction embedding with (E, emb_size_edge) shape.
        """
        basis_idx1 = basis_idx1[idx_s].flatten()  # (E*NB)

        # ---------- Geometric MP ----------
        m_st = self.mlp_m_rbf(m_st)
        m_st_rbf = m_st * self.mlp_rbf(rbf)
        m_st = self.scale_rbf(m_st_rbf, ref=m_st)  # (E, emb_size_edge)

        E, NB, _ = cbf.size()
        cbf = cbf.reshape(E * NB, -1)  # (E*NB, emb_size_cbf)
        m_st_nb: Tensor = self.mlp_m_cbf(m_st)  # (E, emb_size_edge)
        m_st_nb = m_st_nb[basis_idx1]  # (E*NB, emb_size_edge)
        m_st_cbf = m_st_nb * self.mlp_cbf(cbf)
        m_st_nb = self.scale_cbf(m_st_cbf, ref=m_st_nb)  # (E, NB, emb_size_edge)

        m_st_nb = m_st_nb.reshape(E, NB, -1)  # (E, NB, emb_size_edge)

        # ---------- Basis MP ----------
        x = m_st_nb.sum(1)  # (E, emb_size_edge)
        x = x * self.inv_sqrt_neighbor
        x = self.mlp_direction(x)

        # ---------- Update embeddings ----------
        x_st = self.mlp_st(x)  # (E, emb_size_edge)
        x_ts = self.mlp_ts(x)  # (E, emb_size_edge)

        # Merge interaction of s->t and t->s
        x_ts = x_ts[idx_swap]  # swap to add to edge s->t and not t->s
        x3 = x_st + x_ts
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
            mlp.append(ResidualLayer(units, 2, False, activation=activation, weight_init=weight_init))
        return nn.ModuleList(mlp)

    def forward(self, h: Tensor, m_st: Tensor, rbf: Tensor, idx_t: Tensor) -> Tensor:
        """
        Args:
            h (torch.Tensor): Atom embedding with (N, emb_size_atom) shape.
            m_st (torch.Tensor): edge embedding with (E, emb_size_edge) shape.
            rbf (torch.Tensor): RBF of ji with (E, emb_size_rbf) shape.
            idx_t (torch.Tensor): Edge index of target atom with (E) shape.

        Returns:
            h (torch.Tensor): Updated atom embedding with (N, emb_size_atom) shape.
        """
        N = h.size(0)

        mlp_rbf = self.mlp_rbf(rbf)  # (E, emb_size_edge)
        x = m_st * mlp_rbf

        x2 = scatter(x, idx_t, dim=0, dim_size=N, reduce="add")
        x = self.scale_sum(x2, ref=m_st)  # (N, emb_size_edge)

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

    def forward(self, h: Tensor, m_st: Tensor, idx_s: Tensor, idx_t: Tensor) -> Tensor:
        """
        Args:
            h (torch.Tensor): Atom embedding with (N, emb_size_atom) shape.
            m_st (torch.Tensor): edge embedding with (E, emb_size_edge) shape.
            idx_s (torch.Tensor): Edge index of source atom with (E) shape.
            idx_t (torch.Tensor): Edge index of target atom with (E) shape.

        Returns:
            m_st (torch.Tensor): Updated edge embedding with (E, emb_size_edge) shape.
        """
        h_s = h[idx_s]
        h_t = h[idx_t]

        m_st = torch.cat([h_s, h_t, m_st], dim=-1)  # (E, 2*emb_size_atom+emb_size_edge)
        m_st = self.mlp_embed(m_st)  # (E, emb_size_edge)
        return m_st
