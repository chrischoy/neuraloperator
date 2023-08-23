from typing import Optional, Union, List, Literal, Callable
from jaxtyping import Shaped, Float

import torch
from torch import nn
import torch.nn.functional as F

from torch_scatter import segment_csr
from neuralop.layers.mlp import MLPLinear
from neuralop.layers.neighbor_search import (
    NeighborSearchReturn,
    DistributedNeighborSearchReturn,
)
from neuralop.mpu.mappings import gather_from_model_parallel_region


class IntegralTransform(nn.Module):
    """Integral Kernel Transform (GNO)
    Computes one of the following:
        (a) \int_{A(x)} k(x, y) dy
        (b) \int_{A(x)} k(x, y) * f(y) dy
        (c) \int_{A(x)} k(x, y, f(y)) dy
        (d) \int_{A(x)} k(x, y, f(y)) * f(y) dy

    x : Points for which the output is defined
    y : Points for which the input is defined
    A(x) : A subset of all points y (depending on
           each x) over which to integrate
    k : A kernel parametrized as a MLP
    f : Input function to integrate against given
        on the points y

    If f is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(x) are specified
    as a graph in CRS format.

    Parameters
    ----------
    mlp : torch.nn.Module, default None
        MLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f
    mlp_layers : list, default None
        List of layers sizes speficing a MLP which
        parametrizes the kernel k. The MLP will be
        instansiated by the MLPLinear class
    mlp_non_linearity : callable, default torch.nn.functional.gelu
        Non-linear function used to be used by the
        MLPLinear class. Only used if mlp_layers is
        given and mlp is None
    transform_type : str, default 'linear'
        Which integral transform to compute. The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b)
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter.
    """

    def __init__(
        self,
        mlp: Optional[nn.Module] = None,
        mlp_layers: Optional[nn.Module] = None,
        mlp_non_linearity: Optional[Callable] = F.gelu,
        transform_type: Literal[
            "linear", "linear_kernelonly", "nonlinear", "nonlinear_kernelonly"
        ] = "linear",
    ):
        super().__init__()

        assert mlp is not None or mlp_layers is not None

        self.transform_type = transform_type

        if (
            self.transform_type != "linear_kernelonly"
            and self.transform_type != "linear"
            and self.transform_type != "nonlinear_kernelonly"
            and self.transform_type != "nonlinear"
        ):
            raise ValueError(
                f"Got {transform_type=} but expected one of [linear_kernelonly, linear, nonlinear_kernelonly, nonlinear]"
            )

        if mlp is None:
            self.mlp = MLPLinear(layers=mlp_layers, non_linearity=mlp_non_linearity)
        else:
            self.mlp = mlp

    """"
    

    Assumes x=y if not specified
    Integral is taken w.r.t. the neighbors
    If no weights are given, a Monte-Carlo approximation is made
    NOTE: For transforms of type 0 or 2, out channels must be
    the same as the channels of f
    """

    def forward(
        self, y, neighbors: NeighborSearchReturn, x=None, f_y=None, weights=None
    ):
        """Compute a kernel integral transform

        Parameters
        ----------
        y : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
        neighbors : NeighborSearchReturn
            The sets A(x) given in CRS format. It contains
            "neighbors_index" and "neighbors_row_splits."
            For descriptions of the two, see NeighborSearch.
        x : torch.Tensor of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined. If None,
            x = y.
        f_y : torch.Tensor of shape [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : torch.Tensor of shape [n,], default None
            Weights for each point y proprtional to the
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are y_{j+1} - y_j. If None,
            1/|A(x)| is used.

        Output
        ----------
        out_features : torch.Tensor of shape [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """

        if x is None:
            x = y

        rep_features = y[neighbors.neighbors_index]
        if f_y is not None:
            in_features = f_y[neighbors.neighbors_index]

        num_reps = (
            neighbors.neighbors_row_splits[1:] - neighbors.neighbors_row_splits[:-1]
        )
        self_features = torch.repeat_interleave(x, num_reps, dim=0)

        agg_features = torch.cat([rep_features, self_features], dim=1)
        if f_y is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            agg_features = torch.cat([agg_features, in_features], dim=1)

        rep_features = self.mlp(agg_features)

        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            rep_features = rep_features * in_features

        if weights is not None:
            rep_features = weights[neighbors.neighbors_index] * rep_features
            reduction = "sum"
        else:
            reduction = "mean"

        out_features = segment_csr(
            rep_features, neighbors.neighbors_row_splits, reduce=reduction
        )
        return out_features


class DistributedIntegralTransform(nn.Module):
    def __init__(
        self,
        mlp: nn.Module = None,
        in_channels: int = 8,
        hidden_dim: int = 32,
        out_channels: int = 32,
        reduction: str = "mean",
    ):
        super().__init__(mlp, in_channels, hidden_dim, out_channels, reduction)
        self.rank = torch.distributed.get_rank()
        # TODO check the MLP weights are the same across all GPUs

    def forward(
        self,
        in_features: Float[torch.Tensor, "N C_in"],
        neighbors: DistributedNeighborSearchReturn,
        node_features: Optional[Float[torch.Tensor, "M C_in"]] = None,
        in_weights: Optional[Float[torch.Tensor, "N"]] = None,
    ) -> Float[torch.Tensor, "M C_out"]:
        assert in_features is not None, f"Input features cannot be None"
        assert neighbors is not None, f"Neighbors cannot be None"

        if node_features is None:
            node_features = in_features

        assert (
            in_features.shape[1] + node_features.shape[1]
            == self.mlp.layers[0].in_features
        ), f"In feature shape: {in_features.shape}, node feature shape: {node_features.shape}, MLP in feature shape: {self.mlp.layers[0].in_features}"

        neighbors = neighbors.ddp_distribute()

        neighbor_index = neighbors.neighbors_index_dev.long()
        in_features_dev = in_features[neighbor_index]

        rs = neighbors.neighbors_row_splits_dev.long()
        num_reps = rs[1:] - rs[:-1]

        node_features_dev = node_features[
            neighbors.neighbors_index_start : neighbors.neighbors_index_end
        ]
        # repeat the self features using num_reps
        self_features = torch.repeat_interleave(node_features_dev, num_reps, dim=0)
        agg_features = torch.cat([in_features_dev, self_features], dim=1)
        agg_features = self.mlp(agg_features)

        if in_weights is not None:
            rep_weights = in_weights[neighbor_index].unsqueeze(-1)
            agg_features = rep_weights * agg_features
        # Runs without a problem. Add assertions to make sure the output is correct
        # print(agg_features.shape)
        # print(neighbors.neighbors_row_splits_at(i).shape)
        # print(neighbors.neighbors_row_splits_at(i).max())
        # assert agg_features.shape[0] == neighbors.neighbors_row_splits_at(i).max()
        # torch.cuda.set_device(device) already done in segment_csr_cuda.cu
        out_features_dev = segment_csr(
            agg_features.contiguous(),
            neighbors.neighbors_row_splits_dev,
            reduce=self.reduction,
        )

        out_features = gather_from_model_parallel_region(out_features_dev, dim=0)
        return out_features
