from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from ..data.keys import GraphKeys


class BaseTransform(ABC, nn.Module):
    def __init__(self):
        super.__init__()

    @abstractmethod
    def transform(self, graph: Batch) -> Batch:
        return graph

    def forward(self, graph: Batch) -> Batch:
        return self.transform(graph)


class RotationPolarTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def transform(self, graph: Batch) -> Batch:
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
