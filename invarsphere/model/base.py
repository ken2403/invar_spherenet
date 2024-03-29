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
            return_vec (bool, optional): return normalized distance vector from s to t atom. Defaults to `False`.

        Returns:
            graph (torch_geometric.data.Batch): material graph batch with edge information:
                edge_dist (torch.Tensor): inter atomic distances of (E) shape.
                edge_vec_ji (torch.Tensor): normalized inter atomic vector from s to t atom of (E, 3) shape.
        """
        if graph.get(GraphKeys.Batch_idx) is not None:
            batch_ind = graph[GraphKeys.Batch_idx]
        else:
            batch_ind = graph[GraphKeys.Pos].new_zeros(graph[GraphKeys.Pos].size(0), dtype=torch.long)
            graph[GraphKeys.Batch_idx] = batch_ind

        # order is "source_to_traget"
        idx_s, idx_t = graph[GraphKeys.Edge_idx]
        edge_batch = batch_ind[idx_s]
        edge_vec = (
            graph[GraphKeys.Pos][idx_t]
            - graph[GraphKeys.Pos][idx_s]
            + torch.einsum("ni,nij->nj", graph[GraphKeys.Edge_shift], graph[GraphKeys.Lattice][edge_batch]).contiguous()
        )

        graph[GraphKeys.Edge_dist_st] = torch.norm(edge_vec, dim=1)
        if return_vec:
            graph[GraphKeys.Edge_vec_st] = edge_vec / graph[GraphKeys.Edge_dist_st].unsqueeze(-1)
        return graph

    @property
    def n_param(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
