from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from torch import Tensor

from ..utils.resolve import init_param_resolver


class Dense(nn.Linear):
    """Applies a linear transformation to the incoming data with using weight
    initialize method."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        weight_init: Callable[[Tensor], Tensor] | None = None,
        bias_init: Callable[[Tensor], Tensor] | None = nn.init.zeros_,
        **kwargs,
    ):
        """
        Args:
            in_dim (int): input dimension of tensor.
            out_dim (int): output dimension of tensor.
            bias (bool, optional): if `False`, the layer will not return an additive bias. Defaults to `True`.
            weight_init (Callable, optional): weight initialize methods. Defaults to `None`.
            bias_init (Callable, optional): bias initialize methods. Defaults to `nn.init.zeros_`.
        """
        if bias and bias_init is None:
            raise ValueError("bias_init must not be None if set bias")
        self.bias_init = bias_init
        self.weight_init = weight_init
        self.kwargs = kwargs
        # gain and scale paramer is set to default values
        if self.weight_init is not None:
            params = init_param_resolver(self.weight_init)
            for p in params:
                if p in kwargs:
                    continue
                if p == "gain":
                    kwargs[p] = 1.0
                # for glorot_orthogonal init function
                elif p == "scale":
                    kwargs[p] = 2.0
            self.kwargs = kwargs

        super().__init__(in_dim, out_dim, bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_init is not None:
            self.weight_init(self.weight, **self.kwargs)
        if self.bias is not None:
            self.bias_init(self.bias)

    def extra_repr(self) -> str:
        weight_init_key = ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])
        return "in_features={}, out_features={}, bias={}, weight_init={}({}), bias_init={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.weight_init.__name__ if self.weight_init is not None else None,
            weight_init_key,
            self.bias_init.__name__ if self.bias_init is not None else None,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward calculation of the Dense layer.

        Args:
            x (torch.Tensor): input tensor of (*, in_dim) shape.

        Returns:
            torch.Tensor: output tensor of (*, out_dim) shape.
        """
        # compute linear layer y = xW^T + b
        return super().forward(x)


class ResidualLayer(nn.Module):
    """Residual block with output scaled by 1/sqrt(2).

    Args:
        hidden_channels (int): hidden dimension of dense layer.
        n_layers (int): Number of dense layers.
        activation (torch.nn.Module | None): activation function to use.
        bias (bool): whether to use bias in dense layer.
    """

    def __init__(
        self,
        hidden_channels: int,
        n_layers: int = 2,
        bias: bool = False,
        activation: nn.Module | None = None,
        weight_init: Callable[[Tensor], Tensor] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.bias = bias
        dense: list[nn.Module] = []
        for _ in range(n_layers):
            dense.append(
                Dense(
                    hidden_channels,
                    hidden_channels,
                    bias=bias,
                    activation_name=activation,
                    weight_init=weight_init,
                    **kwargs,
                )
            )
            if activation is not None:
                dense.append(activation)

        self.dense_mlp = nn.Sequential(*dense)
        self.inv_sqrt_2 = 1 / (2.0**0.5)

        self.reset_parameters()

    def reset_parameters(self):
        for ll in self.dense_mlp:
            if hasattr(ll, "reset_parameters"):
                ll.reset_parameters()

    def extra_repr(self) -> str:
        return f"n_layers={self.n_layers}, bias={self.bias}, scale_after_residual={self.inv_sqrt_2:.2f}"

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.dense_mlp(inputs)
        x = inputs + x
        x = x * self.inv_sqrt_2
        return x
