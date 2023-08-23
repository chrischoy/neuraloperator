from typing import Union, List, Optional, Tuple, Dict, Any

import torch
import torch.distributed as dist
from torchtyping import TensorType

from .rank import rank_to_device


def broadcast_object(obj: Any, src_rank: int = 0, group=None) -> Any:
    rank = dist.get_rank()
    obj_list = [0]
    if rank == src_rank:
        obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src_rank, group=group)
    obj = obj_list[0]
    return obj


def broadcast_size_type(
    tensor: TensorType = None, src_rank: int = 0, group=None
) -> Tuple[List[int], torch.dtype]:
    rank = dist.get_rank()
    if rank == src_rank:
        tensor_size = [s for s in tensor.size()]
        dtype = tensor.dtype
    else:
        tensor_size = [0]
        dtype = None

    # Get size of the tensor
    object_list = [tensor_size, dtype]
    # Broadcasting tensor size to all processes
    dist.broadcast_object_list(object_list, src=src_rank, group=group)
    tensor_size, dtype = object_list
    return tensor_size, dtype


def broadcast_list(list: List[Any], src_rank: int = 0, group=None) -> List[Any]:
    rank = dist.get_rank()
    if rank == src_rank:
        list_size = [len(list)]
    else:
        list_size = [0]

    # Get list_size
    dist.broadcast_object_list(list_size, src=src_rank, group=group)
    list_size = list_size[0]

    if rank != src_rank:
        list = [None for _ in range(list_size)]

    dist.broadcast_object_list(list, src=src_rank, group=group)
    return list


@torch.no_grad()
def broadcast_tensor(
    tensor: TensorType = None, src_rank: int = 0, group=None
) -> TensorType:
    size, dtype = broadcast_size_type(tensor, src_rank, group=group)
    rank = dist.get_rank()

    # On other ranks, this will be the received tensor. We use tensor_size to initialize it.
    if rank != src_rank:
        tensor = torch.empty(tuple(size), dtype=dtype, device=rank_to_device(rank))

    if dtype == torch.complex64:
        tensor = torch.view_as_real(tensor)

    dist.broadcast(tensor, src=src_rank, group=group)

    if dtype == torch.complex64:
        tensor = torch.view_as_complex(tensor)
    return tensor


def broadcast_tensor_list(
    tensors: List[TensorType], src_rank: int = 0, group=None
) -> List[TensorType]:
    rank = dist.get_rank()
    if rank == src_rank:
        for i, tensor in enumerate(tensors):
            assert isinstance(
                tensor, torch.Tensor
            ), f"tensors[{i}] {tensor} is not a torch.Tensor"
        tensor_sizes = [tensor.size() for tensor in tensors]
        tensor_dtypes = [tensor.dtype for tensor in tensors]
    else:
        tensor_sizes = None
        tensor_dtypes = None

    tensor_sizes = broadcast_list(tensor_sizes, src_rank, group=group)
    tensor_dtypes = broadcast_list(tensor_dtypes, src_rank, group=group)

    if rank != src_rank:
        tensors = [None for _ in range(len(tensor_sizes))]

    for i, (tensor_size, tensor_dtype) in enumerate(zip(tensor_sizes, tensor_dtypes)):
        if rank == src_rank:
            tensor = tensors[i].cuda(rank)
        else:
            tensor = torch.empty(
                tuple(tensor_size), dtype=tensor_dtype, device=rank_to_device(rank)
            )
        tensor = broadcast_tensor(tensor, src_rank, group=group)
        tensors[i] = tensor
    return tensors


def broadcast_dict(in_dict: Dict, src_rank: int = 0, group=None) -> Dict:
    rank = dist.get_rank()

    def get_type(value):
        if isinstance(value, torch.Tensor):
            return "Tensor"
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                return "TensorList"
            else:
                return "List"
        elif isinstance(value, dict):
            return "Dict"
        else:
            print(f"Unsupported type. value: {value}")
            raise ValueError(f"Unsupported type. value: {value}")

    # broadcast keys from src_rank
    if rank == src_rank:
        keys = list(in_dict.keys())
        types = [get_type(in_dict[key]) for key in keys]
    else:
        keys = None
        types = None
    keys = broadcast_list(keys, src_rank, group=group)
    types = broadcast_list(types, src_rank, group=group)

    if rank != src_rank:
        in_dict = {}

    for key, vtype in zip(keys, types):
        value = None
        if vtype == "Tensor":
            if rank == src_rank:
                value = in_dict[key].cuda(rank)
            value = broadcast_tensor(value, src_rank, group=group)
        elif vtype == "List":
            if rank == src_rank:
                value = in_dict[key]
            value = broadcast_list(value, src_rank, group=group)
        elif vtype == "TensorList":
            if rank == src_rank:
                value = in_dict[key]
            value = broadcast_tensor_list(value, src_rank, group=group)
        elif vtype == "Dict":
            if rank == src_rank:
                value = in_dict[key]
            value = broadcast_object(value, src_rank, group=group)
        else:
            raise ValueError(f"Unsupported type: {vtype(value)}. value: {value}")
        in_dict[key] = value
    return in_dict
