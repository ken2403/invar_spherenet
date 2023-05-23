from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from ..data.keys import GraphKeys


class BaseMPNN(nn.Module):
    def __init__(self):
        super().__init__()

    def calc_atomic_distances(self, graph: Batch, return_vec: bool = False) -> Batch:
        """calculate atomic distances with PBC.

        Args:
            graph (torch_geometric.data.Batch): material graph batch.
            return_vec (bool, optional): return distance vector from i to j atom. Defaults to `False`.

        Returns:
            graph (torch_geometric.data.Batch): material graph batch with edge information:
                edge_dist_ij (torch.Tensor): inter atomic distances of (E) shape.
                edge_vec_ij (torch.Tensor): inter atomic vector from i to j atom of (E, 3) shape.
                edge_dir_ij (torch.Tensor): edge direction from i to j atom of (E, 3) shape.
        """
        if graph.get(GraphKeys.Batch_idx) is not None:
            batch_ind = graph[GraphKeys.Batch_idx]
        else:
            batch_ind = graph[GraphKeys.Pos].new_zeros(graph[GraphKeys.Pos].shape[0], dtype=torch.long)

        # order is "source_to_traget" i.e. [index_j, index_i]
        edge_j, edge_i = graph[GraphKeys.Edge_idx]
        edge_batch = batch_ind[edge_i]
        edge_vec = (
            graph[GraphKeys.Pos][edge_j]
            - graph[GraphKeys.Pos][edge_i]
            + torch.einsum("ni,nij->nj", graph[GraphKeys.Edge_shift], graph[GraphKeys.Lattice][edge_batch]).contiguous()
        )

        graph[GraphKeys.Edge_dist] = torch.norm(edge_vec, dim=1)
        if return_vec:
            graph[GraphKeys.Edge_vec_ij] = edge_vec
            edge_dir = edge_vec / edge_vec.norm(dim=-1, keepdim=True)
            graph[GraphKeys.Edge_dir_ij] = edge_dir
        return graph

    def rot_transform(self, graph: Batch) -> Batch:
        """Cartesian to polar transform of edge vector.

        Args:
            graph (torch_geometric.data.Batch): material graph batch with following attributes:
                rotation_matrix (torch.Tensor): atom rotation matrix with (N, NB, 3, 3) shape.
                edge_vec_ij (torch.Tensor): cartesian edge vector with (E, 3) shape.

        Returns:
            graph (torch_geometric.data.Batch): material graph batch with following attributes:
                theta (torch.Tensor): the azimuthal angle with (NB, E) shape.
                phi (torch.Tensor): the polar angle with (NB, E) shape.
        """
        rot_mat = graph[GraphKeys.Rot_mat]  # (N, NB, 3, 3)
        vec = graph[GraphKeys.Edge_vec_ij]  # (E, 3)
        idx_i = graph[GraphKeys.Edge_idx][1]  # (E)
        rot_mat = rot_mat[idx_i]  # (E, NB, 3, 3)

        # ---------- rotation transform ----------
        vec = torch.einsum("ebnm,em->ebn", rot_mat, vec)  # (E, NB, 3)

        # ---------- cart to polar transform ----------
        vec = vec / vec.norm(dim=-1, keepdim=True)
        theta = torch.atan2(vec[..., 0], vec[..., 1])  # (E, NB)
        phi = torch.acos(vec[..., 2])  # (E, NB)

        graph[GraphKeys.Theta] = theta.transpose(0, 1)  # (NB, E)
        graph[GraphKeys.Phi] = phi.transpose(0, 1)  # (NB, E)
        return graph

    @property
    def n_param(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
