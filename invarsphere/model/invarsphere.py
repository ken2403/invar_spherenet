from __future__ import annotations

import copy
import logging
import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_scatter import scatter, segment_csr
from torch_sparse import SparseTensor

from ..data.keys import GraphKeys
from ..nn.base import Dense, ResidualLayer
from ..nn.basis import SphericalBesselFunction
from ..nn.cutoff import BaseCutoff
from ..nn.direct_basis import SphericalHarmonicsWithBesselDirect
from ..nn.efficient import (
    EfficientInteractionBilinear,
    EfficientInteractionDownProjection,
)
from ..nn.scaling.compat import load_scales_compat
from ..nn.scaling.scale_factor import ScaleFactor
from ..utils.math import inner_product_normalized
from ..utils.repeat_tensor import (
    block_repeat,
    block_repeat_each,
    ragged_range,
    repeat_blocks,
)
from ..utils.resolve import activation_resolver, cutoffnet_resolver, init_resolver
from .base import BaseMPNN

logger = logging.getLogger(__name__)


class InvarianceSphereNet(BaseMPNN):
    def __init__(
        self,
        emb_size_atom: int = 128,
        emb_size_edge: int = 128,
        emb_size_rbf: int = 64,
        emb_size_cbf: int = 64,
        emb_size_sbf: int = 64,
        emb_triplet: int = 64,
        emb_quad: int | None = 64,
        n_blocks: int = 4,
        n_targets: int = 1,
        max_n: int = 6,
        max_l: int = 5,
        triplets_only: bool = False,
        nb_only: bool = False,
        rbf_smooth: bool = True,
        cutoff: float = 6.0,
        cutoff_net: str | type[BaseCutoff] = "envelope",
        cutoff_kwargs: dict[str, Any] = {"p": 5.0},
        n_residual_output: int = 1,
        max_z: int | None = None,
        extensive: bool = True,
        regress_forces: bool = True,
        direct_forces: bool = True,
        activation: str | nn.Module = "scaledsilu",
        weight_init: str | Callable[[Tensor], Tensor] | None = None,
        align_initial_weight: bool = False,
        scale_file: str | None = None,
    ):
        super().__init__()
        act = activation_resolver(activation)
        wi = init_resolver(weight_init) if weight_init is not None else None

        self.n_blocks = n_blocks
        self.n_targets = n_targets
        self.max_n = max_n
        self.max_l = max_l
        self.triplets_only = triplets_only
        self.nb_only = nb_only
        self.rbf_smooth = rbf_smooth
        self.cutoff = cutoff
        self.extensive = extensive
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.align_initial_weight = align_initial_weight

        if triplets_only and nb_only:
            raise ValueError("Triplets only and neighbor basis only cannot be set simultaneously.")

        # basis layers
        cutoff_kwargs["cutoff"] = cutoff
        cn = cutoffnet_resolver(cutoff_net, **cutoff_kwargs)
        self.rbf = SphericalBesselFunction(max_n, 1, cutoff, None if rbf_smooth else cn, rbf_smooth)
        if not nb_only:
            self.cbf = SphericalHarmonicsWithBesselDirect(max_n, max_l, cutoff, cn, use_phi=False, efficient=True)
        if not triplets_only:
            self.sbf = SphericalHarmonicsWithBesselDirect(max_n, max_l, cutoff, cn, use_phi=True, efficient=True)

        # shared layers
        self.mlp_rbf_h = Dense(max_n, emb_size_rbf, bias=False, weight_init=wi)
        self.mlp_rbf_out = Dense(max_n, emb_size_rbf, bias=False, weight_init=wi)
        if not nb_only:
            self.mlp_rbf3 = Dense(max_n, emb_size_rbf, bias=False, weight_init=wi)
            self.mlp_cbf3 = EfficientInteractionDownProjection(max_l, max_n * max_l, emb_size_cbf)
        if not triplets_only:
            self.mlp_sbf4 = EfficientInteractionDownProjection(max_l * max_l, max_n * max_l, emb_size_sbf)

        # embedding block
        self.emb_block = EmbeddingBlock(max_n, emb_size_atom, emb_size_edge, max_z, act, wi)

        # interaction and output blocks
        self.int_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    emb_size_atom,
                    emb_size_edge,
                    emb_size_rbf,
                    emb_size_cbf,
                    emb_triplet,
                    emb_size_sbf,
                    emb_quad,
                    n_before_skip=1,
                    n_after_skip=1,
                    n_after_atom_self=1,
                    n_atom_emb=1,
                    triplets_only=triplets_only,
                    nb_only=nb_only,
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

    def _mask_neighbors(self, neighbors: Tensor, edge_mask: Tensor) -> Tensor:
        neighbors_old_indptr = torch.cat([neighbors.new_zeros(1), neighbors])
        neighbors_old_indptr = torch.cumsum(neighbors_old_indptr, dim=0)
        neighbors = segment_csr(edge_mask.long(), neighbors_old_indptr)
        return neighbors

    def _select_edges(
        self,
        edge_index: Tensor,
        cell_offsets: Tensor,
        neighbors: Tensor,
        edge_dist: Tensor,
        edge_vector: Tensor,
        cutoff: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        edge_mask = edge_dist <= cutoff

        edge_index = edge_index[:, edge_mask]
        cell_offsets = cell_offsets[edge_mask]
        neighbors = self._mask_neighbors(neighbors, edge_mask)
        edge_dist = edge_dist[edge_mask]
        edge_vector = edge_vector[edge_mask]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError("Empty images are existed in the dataset.")
        return edge_index, cell_offsets, neighbors, edge_dist, edge_vector

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

    def _get_triplets(self, edge_index: Tensor, n_node: int) -> tuple[Tensor, Tensor, Tensor]:
        """Get all k->t for each edge s->t. It is possible that k=s, as long as
        the edges are distinct.

        Returns:
            id3_kt (torch.Tensor): shape (T)
                Indices of input edge k->t of each triplet k->t<-s
            id3_st (torch.Tensor): shape (T)
                Indices of output edge s->t of each triplet b->t<-s
            id3_ragged_idx (torch.Tensor): shape (T)
                Indices enumerating the copies of id3_st for creating a padded matrix
        """
        idx_s, idx_t = edge_index  # edge_index order is "source_to_target"

        value = torch.arange(idx_s.size(0), device=idx_s.device, dtype=idx_s.dtype)
        # Possibly contains multiple copies of the same edge (for periodic interactions)
        adj = SparseTensor(
            row=idx_t,
            col=idx_s,
            value=value,
            sparse_sizes=(n_node, n_node),
        )
        adj_edges = adj[idx_t]

        # Edge indices (k->t, s->t) for triplets.
        id3_kt = adj_edges.storage.value()
        id3_st = adj_edges.storage.row()

        # Remove self-loop triplets
        # Compare edge indices, not atom indices to correctly handle periodic interactions
        mask = id3_kt != id3_st
        id3_kt = id3_kt[mask]
        id3_st = id3_st[mask]

        # Get indices to reshape the neighbor indices k->t into a dense matrix.
        # id3_st has to be sorted for this to work.
        num_triplets = torch.bincount(id3_st, minlength=idx_s.size(0))
        id3_ragged_idx = ragged_range(num_triplets)

        return id3_kt, id3_st, id3_ragged_idx

    def _get_edge_nb_info(
        self,
        idx_s: Tensor,
        rot_mat: Tensor,
        basis_node_idx: Tensor,
        n_node: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cnt_bnode = torch.bincount(basis_node_idx, minlength=n_node)
        cnt_edge_s = torch.bincount(idx_s, minlength=n_node)

        edge_nb_idx = block_repeat_each(idx_s, cnt_edge_s, cnt_bnode, return_index=True)
        nb_edge_idx = block_repeat(rot_mat, cnt_bnode, cnt_edge_s, return_index=True)

        n_edge_nb = torch.bincount(edge_nb_idx, minlength=idx_s.size(0))
        edge_nb_ragged_idx = ragged_range(n_edge_nb)

        return edge_nb_idx, nb_edge_idx, edge_nb_ragged_idx

    def _rot_transform(
        self,
        rot_mat: Tensor,
        vec_ij: Tensor,
        nb_edge_idx: Tensor,
        edge_nb_idx: Tensor,
    ) -> Tensor:
        """Cartesian to polar transform of edge vector.

        Args:
            rot_mat (torch.Tensor): atom rotation matrix with (NB, 3, 3) shape.
            vec_ij (torch.Tensor): cartesian edge vector with (E, 3) shape.
            nb_edge_idx (torch.Tensor):  edge neighbor index of (E_NB) shape, used to extend NB->E_NB.
            edge_nb_idx (torch.Tensor): edge index of (E_NB) shape, used to extend E->E_NB.

        Returns:
            rotated_vec (torch.Tesor): the normalized rotated vector with (E_NB, 3) shape.
        """
        rot_mat = rot_mat[nb_edge_idx]  # (E_NB, 3, 3)

        # ---------- rotation transform ----------
        rot_vec = torch.einsum("enm,em->en", rot_mat, vec_ij[edge_nb_idx])  # (E_NB, 3)
        rot_vec = rot_vec / rot_vec.norm(dim=-1, keepdim=True)  # (E_NB, 3)
        return rot_vec  # (E_NB, 3)

        # # ---------- cart to polar transform ----------
        # rot_vec = rot_vec / rot_vec.norm(dim=-1, keepdim=True)
        # # Define azimuthal angle as counterclockwise from the y-axis of the second proximity
        # cosθ = torch.cos(torch.atan2(rot_vec[:, 2], rot_vec[:, 1]))  # (E_NB)
        # # The angle of first proximity is the polar angle
        # cosφ_b1 = rot_vec[:, 0]  # (E_NB)

        # # The angle with second proimity
        # cosφ_b2 = inner_product_normalized(rot_vec, vec_ij[basis_edge_idx2][nb_edge_idx])  # (E_NB)

        # return cosθ, cosφ_b1, cosφ_b2  # (E_NB)

    def generate_interaction_graph(self, graph: Batch) -> Batch:
        # batch index
        if graph.get(GraphKeys.Batch_idx) is None:
            z = graph[GraphKeys.Z]
            graph[GraphKeys.Batch_idx] = z.new_zeros(z.size(0), dtype=torch.long)

        # interatomic distances
        graph = self.calc_atomic_distances(graph, return_vec=True)

        # modify edge_info
        edge_index: Tensor = graph[GraphKeys.Edge_idx]
        cell_offsets: Tensor = graph[GraphKeys.Edge_shift]
        neighbors: Tensor = graph[GraphKeys.Neighbors]
        d_st: Tensor = graph[GraphKeys.Edge_dist_st]
        v_st: Tensor = graph[GraphKeys.Edge_vec_st]
        # (
        #     edge_index,
        #     cell_offsets,
        #     neighbors,
        #     d_st,
        #     v_st,
        # ) = self._select_edges(edge_index, cell_offsets, neighbors, d_st, v_st, self.cutoff)
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

        # indices for swapping c->a and a->c (for symmetric MP)
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

        # triplets
        if not self.nb_only:
            id3_kt, id3_st, id3_ragged_idx = self._get_triplets(edge_index, n_node=graph[GraphKeys.Z].size(0))
            graph[GraphKeys.T_edge_idx_kt] = id3_kt
            graph[GraphKeys.T_edge_idx_st] = id3_st
            graph[GraphKeys.T_ragged_idx] = id3_ragged_idx

        # edge neighbor basis information
        if not self.triplets_only:
            rot_mat = graph[GraphKeys.Rot_mat]
            basis_node_idx = graph[GraphKeys.Basis_node_idx]
            n_node = graph[GraphKeys.Z].size(0)
            edge_nb_idx, nb_edge_idx, edge_nb_ragged_idx = self._get_edge_nb_info(
                edge_index[0], rot_mat, basis_node_idx, n_node
            )
            graph[GraphKeys.Edge_nb_idx] = edge_nb_idx
            graph[GraphKeys.Edge_nb_ragged_idx] = edge_nb_ragged_idx

            # rotation transform with node rot_matrix
            rot_vec = self._rot_transform(rot_mat, v_st, nb_edge_idx, edge_nb_idx)
            graph[GraphKeys.Rotated_vec] = rot_vec

        return graph

    def forward(self, graph: Batch) -> tuple[Tensor, Tensor]:
        if self.regress_forces and not self.direct_forces:
            graph[GraphKeys.Pos].requires_grad_(True)

        graph = self.generate_interaction_graph(graph)

        # ---------- Get tensors ----------
        z: Tensor = graph[GraphKeys.Z]
        d_st: Tensor = graph[GraphKeys.Edge_dist_st]
        v_st: Tensor = graph[GraphKeys.Edge_vec_st]
        batch_idx: Tensor = graph[GraphKeys.Batch_idx]
        # order is "source_to_target"
        idx: Tensor = graph[GraphKeys.Edge_idx]
        idx_s = idx[0]
        idx_t = idx[1]
        idx_swap: Tensor = graph[GraphKeys.Edge_idx_swap]
        if not self.nb_only:
            id3_kt: Tensor = graph[GraphKeys.T_edge_idx_kt]
            id3_st: Tensor = graph[GraphKeys.T_edge_idx_st]
            id3_ragged_idx: Tensor = graph[GraphKeys.T_ragged_idx]
        else:
            id3_kt = id3_st = id3_ragged_idx = None
        if not self.triplets_only:
            edge_nb_idx = graph[GraphKeys.Edge_nb_idx]
            edge_nb_ragged_idx = graph[GraphKeys.Edge_nb_ragged_idx]
        else:
            edge_nb_idx = edge_nb_ragged_idx = None

        # ---------- Basis layers ----------
        # --- rbf ---
        rbf = self.rbf(d_st)
        # transform rbf to (E, emb_size_rbf)
        rbf_h = self.mlp_rbf_h(rbf)  # (E, emb_size_rbf)
        rbf_out = self.mlp_rbf_out(rbf)  # (E, emb_size_rbf)
        if not self.nb_only:
            rbf3 = self.mlp_rbf3(rbf)  # (E, emb_size_rbf)
        else:
            rbf3 = None

        # --- cbf ---
        if not self.nb_only:
            cosφ_stk = inner_product_normalized(v_st[id3_st], v_st[id3_kt])
            rad_cbf3, cbf3 = self.cbf(d_st, costheta=cosφ_stk)
            # transform cbf to (T, emb_size_cbf)
            cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_kt, id3_ragged_idx)  # (T, emb_size_cbf)
        else:
            cbf3 = None

        # --- Neighbor basis sbf ---
        if not self.triplets_only:
            rot_vec = graph[GraphKeys.Rotated_vec]
            rad_sbf4, sbf4 = self.sbf(d_st, vec=rot_vec)  # (1, E, max_n) and (E_NB, max_n*max_l*max_l)
            # transform sbf to (E_NB, emb_size_sbf)
            sbf4 = self.mlp_sbf4(rad_sbf4, sbf4, edge_nb_idx, edge_nb_ragged_idx)  # (E_NB, emb_size_sbf)
        else:
            sbf4 = None

        # ---------- EmbeddingBlock and OutputBlock----------
        # (N, emb_size) & (E, emb_size)
        h, m_st = self.emb_block(z, rbf, idx_s, idx_t)
        # (B, n_targets) & (E, n_targets)
        E_t, F_st = self.out_blocks[0](h, m_st, rbf_out, idx_t)

        # ---------- InteractionBlock and OutputBlock ----------
        for i in range(self.n_blocks):
            # interacton
            h, m_st = self.int_blocks[i](
                h,
                m_st,
                rbf_h,
                rbf3,
                cbf3,
                sbf4,
                idx_s,
                idx_t,
                idx_swap,
                id3_kt,
                id3_st,
                id3_ragged_idx,
                edge_nb_idx,
                edge_nb_ragged_idx,
            )

            # output
            E, F = self.out_blocks[i + 1](h, m_st, rbf_out, idx_t)
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
                F_st = F_st[:, :, None] * v_st[:, None, :]  # (E, n_targets, 3)
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
        n_edge_features: int,
        emb_size_atom: int,
        emb_size_edge: int,
        max_z: int | None,
        activation: nn.Module,
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        """
        Args:
            n_edge_features (int): the size of the edge features (RBF).
            emb_size_atom (int): The atom embedding size after the embedding block.
            emb_size_edge (int): The edge embedding size after the embedding block.
            max_z (int | None): Maximum atomic number. If `None`, set 93 as max_z.
            activation (nn.Module): Activation function.
            weight_init (Callble[[Tensor], Tensor] | None): weight init function.
        """
        super().__init__()
        if max_z is None:
            max_z = 93
        self.atom_embedding = nn.Embedding(max_z, emb_size_atom)
        in_features = 2 * emb_size_atom + n_edge_features
        self.mlp_embed = nn.Sequential(
            Dense(in_features, emb_size_edge, False, weight_init),
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
            h (torch.Tensor): Atom embedding with (N, emb_size_atom) shape.
            m_st (torch.Tensor): Edge embedding with (E, emb_size_edge) shape.
        """
        h = self.atom_embedding(z - 1)
        h_s = h[idx_s]  # (E, emb_size_atom)
        h_t = h[idx_t]  # (E, emb_size_atom)

        m_st = torch.cat([h_s, h_t, rbf], dim=-1)  # (E, 2*emb_size_atom+n_edge_features)
        m_st = self.mlp_embed(m_st)  # (E, emb_size_edge)
        return h, m_st


class InteractionBlock(nn.Module):
    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_rbf: int,
        emb_size_cbf: int | None,
        emb_triplet: int | None,
        emb_size_sbf: int | None,
        emb_quad: int | None,
        n_before_skip: int,
        n_after_skip: int,
        n_after_atom_self: int,
        n_atom_emb: int,
        triplets_only: bool,
        nb_only: bool,
        activation: nn.Module,
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        self.triplets_only = triplets_only
        self.nb_only = nb_only

        if triplets_only and nb_only:
            raise ValueError("Triplets_only and nb_only cannot be True at the same time.")

        # ---------- Geometric MP ----------
        self.mlp_st = Dense(emb_size_edge, emb_size_edge, False, weight_init=weight_init)
        if not triplets_only:
            assert emb_size_sbf is not None
            assert emb_quad is not None
            self.nb_mp = NearestBasisInteraction(
                emb_size_atom,
                emb_size_edge,
                emb_size_sbf,
                emb_quad,
                activation,
                weight_init,
            )
        if not nb_only:
            assert emb_size_cbf is not None
            assert emb_triplet is not None
            self.t_mp = TripletInteraction(
                emb_size_edge,
                emb_size_rbf,
                emb_size_cbf,
                emb_triplet,
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
        rbf3: Tensor | None,
        cbf3: tuple[Tensor, Tensor] | None,
        sbf4: tuple[Tensor, Tensor] | None,
        idx_s: Tensor,
        idx_t: Tensor,
        idx_swap: Tensor,
        id3_kt: Tensor,
        id3_st: Tensor,
        id3_ragged_idx: Tensor,
        edge_nb_idx: Tensor | None,
        edge_nb_ragged_idx: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        # ---------- Geometric MP ----------
        # Initial transformation
        x_st_skip = self.mlp_st(m_st)  # (E, emb_size_edge)

        if not self.triplets_only:
            assert sbf4 is not None and edge_nb_idx is not None and edge_nb_ragged_idx is not None
            x_nb = self.nb_mp(h, sbf4, idx_s, idx_t, idx_swap, edge_nb_idx, edge_nb_ragged_idx)
        if not self.nb_only:
            assert rbf3 is not None and cbf3 is not None
            x3 = self.t_mp(m_st, rbf3, cbf3, idx_swap, id3_kt, id3_st, id3_ragged_idx)

        # ---------- Merge Embeddings after Quadruplet and Triplet Interaction ----------
        if self.triplets_only:
            x = x_st_skip + x3  # (E, emb_size_edge)
            x = x * self.inv_sqrt_2
        elif self.nb_only:
            x = x_st_skip + x_nb
            x = x * self.inv_sqrt_2
        else:
            x = x_st_skip + x3 + x_nb
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


class NearestBasisInteraction(nn.Module):
    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_sbf: int,
        emb_quad: int,
        activation: nn.Module,
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()

        self.mlp_down = nn.Sequential(
            Dense(emb_size_atom, emb_quad, bias=False, weight_init=weight_init),
            activation,
        )

        self.mlp_sbf = EfficientInteractionBilinear(emb_quad, emb_size_sbf, emb_quad)
        self.scale_sbf_sum = ScaleFactor()

        self.mlp_m_st = nn.Sequential(
            Dense(2 * emb_quad, emb_quad, bias=False, weight_init=weight_init),
            activation,
        )

        self.mlp_st = nn.Sequential(
            Dense(emb_quad, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_ts = nn.Sequential(
            Dense(emb_quad, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )

        self.inv_sqrt_2 = 1 / (2.0**0.5)

    def forward(
        self,
        h: Tensor,
        sbf: tuple[Tensor, Tensor],
        idx_s: Tensor,
        idx_t: Tensor,
        idx_swap: Tensor,
        edge_nb_idx: Tensor,
        edge_nb_ragged_idx: Tensor,
    ) -> Tensor:
        """
        Args:
            h (Tensor): Atom embedding with (N, emb_size_atom) shape.
            m_st (Tensor): Edge embedding with (E, emb_size_edge) shape.
            sbf (tuple[Tensor, Tensor]): the weighted RBF and SBF with (E_NB, emb_size_sbf) shape.
            idx_swap (Tensor): swap index of edge with (E) shape.
            edge_nb_idx (Tensor): edge index of neighbor basis with (E_NB) shape.
            edge_nb_ragged_idx (Tensor): ragged edge index of neighbor basis with (E_nb) shape.

        Returns:
            x4 (Tensor): Qudruplet interaction embedding with (E, emb_size_edge) shape.
        """
        # ---------- Geometric MP ----------
        h_t = self.mlp_down(h)  # (N, emb_quad)

        h_t = h_t[idx_t][edge_nb_idx]  # (E_NB, emb_quad)

        h_sbf = self.mlp_sbf(sbf, h_t, edge_nb_idx, edge_nb_ragged_idx)  # (E, emb_quad)
        h_sbf = scatter(h_sbf, idx_s, dim=0, dim_size=h.size(0), reduce="add")  # (N, emb_quad)
        h_mp = self.scale_sbf_sum(h_sbf, ref=h_t)  # (N, emb_quad)

        x = self.mlp_m_st(torch.cat([h_mp[idx_s], h_mp[idx_t]], dim=-1))  # (E, emb_quad)

        # ---------- Update embeddings ----------
        x_st = self.mlp_st(x)  # (E, emb_size_edge)
        x_ts = self.mlp_ts(x)  # (E, emb_size_edge)

        # Merge interaction of s->t and t->s
        x_ts = x_ts[idx_swap]  # swap to add to edge s->t and not t->s
        x_nb = x_st + x_ts
        x_nb = x_nb * self.inv_sqrt_2
        return x_nb


class TripletInteraction(nn.Module):
    def __init__(
        self,
        emb_size_edge: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_triplet: int,
        activation: nn.Module,
        weight_init: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()

        self.mlp_m_kt = nn.Sequential(
            Dense(emb_size_edge, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )

        self.mlp_rbf = Dense(emb_size_rbf, emb_size_edge, bias=False, weight_init=weight_init)
        self.scale_rbf = ScaleFactor()

        self.mlp_down = nn.Sequential(
            Dense(emb_size_edge, emb_triplet, bias=False, weight_init=weight_init),
            activation,
        )

        self.mlp_cbf = EfficientInteractionBilinear(emb_triplet, emb_size_cbf, emb_triplet)
        self.scale_cbf_sum = ScaleFactor()

        self.mlp_st = nn.Sequential(
            Dense(emb_triplet, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )
        self.mlp_ts = nn.Sequential(
            Dense(emb_triplet, emb_size_edge, bias=False, weight_init=weight_init),
            activation,
        )

        self.inv_sqrt_2 = 1 / (2.0**0.5)

    def forward(
        self,
        m_st: Tensor,
        rbf: Tensor,
        cbf: tuple[Tensor, Tensor],
        idx_swap: Tensor,
        id3_kt: Tensor,
        id3_st: Tensor,
        id3_ragged_idx: Tensor,
    ) -> Tensor:
        """
        Args:
            m_st (Tensor): Edge embedding with (E, emb_size_edge) shape.
            rbf (Tensor): RBF with (E, emb_size_rbf) shape.
            cbf (tuple[Tensor, Tensor]): the weighted RBF and CBF with (T, emb_size_cbf) shape.
            idx_swap (Tensor): swap index of edge with (E) shape.
            id3_kt (Tensor): Triplet edge index of k->t with (T) shape.
            id3_st (Tensor): Triplet edge index of s->t with (T) shape.
            id3_ragged_idx (Tensor): ragged index of triplet edge with (T) shape.

        Returns:
            x3 (Tensor): Triplet interaction embedding with (E, emb_size_edge) shape.
        """
        m_kt = self.mlp_m_kt(m_st)  # (E, emb_size_edge)

        # ---------- Geometric MP ----------
        # basis representation
        # rbf(d_kt)
        # cbf(d_kt, angle_stk)
        # --- rbf ---
        m_kt_rbf = m_kt * self.mlp_rbf(rbf)  # (E, emb_size_edge)
        m_kt = self.scale_rbf(m_kt_rbf, ref=m_kt)  # (E, emb_size_edge)

        # --- cbf ---
        m_kt = self.mlp_down(m_kt)  # (E, emb_triplet)
        # check triplets for diatomic molecules
        if id3_st.numel() != 0:
            m_kt = m_kt[id3_kt]  # (T, emb_triplet)
            x = self.mlp_cbf(cbf, m_kt, id3_st, id3_ragged_idx)  # (E, emb_triplet)
            x = self.scale_cbf_sum(x, ref=m_kt)  # (E, emb_triplet)
        else:
            x = m_kt  # (E, emb_triplet)

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

        x = m_st * self.mlp_rbf(rbf)  # (E, emb_size_edge)

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
