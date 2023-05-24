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

    @property
    def n_param(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
