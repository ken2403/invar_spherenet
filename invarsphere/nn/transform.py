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


class RotationTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def transform(self, graph: Batch) -> Batch:
        """Rotation transform of edge vector.

        Args:
            graph (torch_geometric.data.Batch): material graph batch with following attributes:
                rotation_matrix (torch.Tensor): the rotation matrix with (*, 3, 3) shape.
                edge_vec (torch.Tensor): the arbitrary vector of (*, 3) shape.

        Returns:
            graph (torch_geometric.data.Batch): material graph batch with following attributes:
                transformed_vec (torch.Tensor): the vector after rotation with (*, 3) shape.
        """
        rot_mat = graph[GraphKeys.Rot_mat]
        vec = graph[GraphKeys.Edge_vec]

        vec = torch.einsum("enm,em->en", rot_mat, vec)

        graph[GraphKeys.Transformed_vec] = vec
        return vec


class CartToPolarTransform(BaseTransform):
    """The layer that transform cartesian coordinates to polar coordinates."""

    def __init__(self):
        super().__init__()

    def transform(self, graph: Batch) -> Batch:
        """Cartesian to polar transform of edge vector.

        Args:
            graph (torch_geometric.data.Batch): material graph batch with following attributes:
                edge_vec (torch.Tensor): arbitrary cartesian vector with (*, 3) shape.

        Returns:
            graph (torch_geometric.data.Batch): material graph batch with following attributes:
                theta (torch.Tensor): the azimuthal angle with (*) shape.
                phi (torch.Tensor): the polar angle with (*) shape.
        """
        vec = graph[GraphKeys.Edge_vec]

        vec = vec / vec.norm(dim=-1, keepdim=True)
        theta = torch.atan2(vec[..., 0], vec[..., 1])
        phi = torch.acos(vec[..., 2])

        graph[GraphKeys.Theta] = theta
        graph[GraphKeys.Phi] = phi
        return graph
