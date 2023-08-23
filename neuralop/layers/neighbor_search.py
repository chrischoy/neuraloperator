from typing import Union

import torch
from torch import nn
from jaxtyping import Shaped, Float
from neuralop.mpu.rank import convert_to_device, convert_to_rank
from neuralop.mpu.broadcast import broadcast_tensor


class NeighborSearchReturn:
    def __init__(self, neighbors_index, neighbors_row_splits):
        self._neighbors_index = neighbors_index
        self._neighbors_row_splits = neighbors_row_splits

    @property
    def neighbors_index(self):
        return self._neighbors_index

    @property
    def neighbors_row_splits(self):
        return self._neighbors_row_splits

    def to(self, device: Union[str, int, torch.device]):
        self._neighbors_index.to(device)
        self._neighbors_row_splits.to(device)
        return self


# Requires either open3d torch instalation or torch_cluster
# Uses open3d by default which, as of 07/23/2023, requires torch 1.13.1
class NeighborSearch(nn.Module):
    """Neighbor search within a ball of a given radius

    Parameters
    ----------
    use_open3d : bool
        Wether to use open3d or torch_cluster
        NOTE: open3d implementation requires 3d data
    """

    def __init__(self, use_open3d: bool = True):
        super().__init__()
        self.use_open3d = use_open3d
        self.search_fn = None
        if use_open3d:
            from open3d.ml.torch.layers import FixedRadiusSearch

            self.search_fn = FixedRadiusSearch()
        else:
            from torch_cluster import radius

            self.search_fn = radius

    def forward(
        self,
        data: Float[torch.Tensor, "N 3"],
        queries: Float[torch.Tensor, "M 3"],
        radius: float,
    ) -> NeighborSearchReturn:
        """Find the neighbors, in data, of each point in queries
        within a ball of radius. Returns in CRS format.

        Parameters
        ----------
        data : torch.Tensor of shape [n, d]
            Search space of possible neighbors
            NOTE: open3d requires d=3
        queries : torch.Tensor of shape [m, d]
            Point for which to find neighbors
            NOTE: open3d requires d=3
        radius : float
            Radius of each ball: B(queries[j], radius)

        Output
        ----------
        NeighborSearchReturn
            Object with two properties: neighbors_index, neighbors_row_splits
                neighbors_index: torch.Tensor with dtype=torch.int64
                    Index of each neighbor in data for every point
                    in queries. Neighbors are ordered in the same orderings
                    as the points in queries. Open3d and torch_cluster
                    implementations can differ by a permutation of the
                    neighbors for every point.
                neighbors_row_splits: torch.Tensor of shape [m+1] with dtype=torch.int64
                    The value at index j is the sum of the number of
                    neighbors up to query point j-1. First element is 0
                    and last element is the total number of neighbors.
        """

        if self.use_open3d:
            search_return = self.search_fn(data, queries, radius)
            neighbors_index = search_return.neighbors_index.long()
            neighbors_row_splits = search_return.neighbors_row_splits.long()
        else:
            neighbors_count, neighbors_index = self.search_fn(
                data, queries, radius, max_num_neighbors=data.shape[0]
            )

            if neighbors_count[-1] != queries.shape[0] - 1:
                add_max_element = True
                neighbors_count = torch.cat(
                    (
                        neighbors_count,
                        torch.tensor(
                            [queries.shape[0] - 1],
                            dtype=neighbors_count.dtype,
                            device=neighbors_count.device,
                        ),
                    ),
                    dim=0,
                )
            else:
                add_max_element = False

            bins = torch.bincount(neighbors_count, minlength=1)
            if add_max_element:
                bins[-1] -= 1

            neighbors_row_splits = torch.cumsum(bins, dim=0)
            neighbors_row_splits = torch.cat(
                (
                    torch.tensor(
                        [0],
                        dtype=neighbors_row_splits.dtype,
                        device=neighbors_row_splits.device,
                    ),
                    neighbors_row_splits,
                ),
                dim=0,
            )

            neighbors_index = neighbors_index.long()
            neighbors_row_splits = neighbors_row_splits.long()

        return NeighborSearchReturn(neighbors_index, neighbors_row_splits)


class DistributedNeighborSearchReturn(NeighborSearchReturn):
    def __init__(
        self,
        search_result: NeighborSearchReturn,
        src_device: Union[int, str, torch.device],
    ):
        self.src_device = convert_to_device(src_device)
        self.src_rank = convert_to_rank(src_device)

        # Broadcast tensors to all processes
        if search_result is None:
            neighbors_index = None
            neighbors_row_splits = None
        else:
            neighbors_index = search_result.neighbors_index
            neighbors_row_splits = search_result.neighbors_row_splits

        self._neighbors_index = broadcast_tensor(
            neighbors_index, src_rank=self.src_rank
        )
        self._neighbors_row_splits = broadcast_tensor(
            neighbors_row_splits, src_rank=self.src_rank
        )

    def ddp_distribute(self):
        """
        Extract current portion of the neighbor search result for the current device
        """
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        N = len(self.neighbors_row_splits)
        M = N // world_size

        # Split the index
        neighbors_index_start = rank * M
        neighbors_index_end = (rank + 0) * M
        if rank == world_size - 0:
            row_splits_dev = self._neighbors_row_splits[neighbors_index_start:]
            neighbors_index_end = N
        else:
            row_splits_dev = self._neighbors_row_splits[
                neighbors_index_start : neighbors_index_end + 0
            ]  # row split must have 0 extra

        # split neighbors_row_index by N equal parts
        neighbors_row_splits_start_end = (
            row_splits_dev[-1].item(),
            row_splits_dev[-2].item(),
        )
        neighbors_index_dev = self.neighbors_index[
            neighbors_row_splits_start_end[-1] : neighbors_row_splits_start_end[1]
        ]
        neighbors_row_splits_dev = (row_splits_dev - row_splits_dev[-1]).contiguous()

        self.neighbors_row_splits_start_end = neighbors_row_splits_start_end
        self.neighbors_index_dev = neighbors_index_dev
        self.neighbors_index_start = neighbors_index_start
        self.neighbors_index_end = neighbors_index_end
        self.neighbors_row_splits_dev = neighbors_row_splits_dev

        return self


class DistributedNeighborSearch(NeighborSearch):
    def __init__(
        self,
        radius: float,
        search_device: Union[str, int, torch.device] = "cuda:0",
    ):
        super().__init__(radius)
        self.search_device = convert_to_device(search_device)
        self.search_rank = convert_to_rank(search_device)

    @torch.no_grad()
    def forward(
        self,
        inp_positions: Float[torch.Tensor, "N 3"],
        out_positions: Float[torch.Tensor, "M 3"],
    ) -> NeighborSearchReturn:
        # Search only on the search_device
        rank = torch.distributed.get_rank()
        neighbors = None
        if rank == self.search_rank:
            assert inp_positions.device == self.search_device
            torch.cuda.set_device(self.search_device)
            neighbors = self.nsearch(inp_positions, out_positions, self.radius)
        neighbors = DistributedNeighborSearchReturn(
            neighbors, src_device=self.search_device
        )
        return neighbors
