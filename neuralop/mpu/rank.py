from typing import Union, Optional, List

import torch


def convert_to_rank(device: Union[int, str, torch.device]) -> int:
    if isinstance(device, int):
        rank = device
    elif isinstance(device, str):
        rank = int(device.split(":")[-1])
    elif isinstance(device, torch.device):
        rank = device.index
    else:
        raise ValueError(
            f"Invalid device type: {device}. Must be one of int, str, torch.device"
        )

    return rank


def convert_to_device(device: Union[int, str, torch.device]) -> torch.device:
    if isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, torch.device):
        pass
    else:
        raise ValueError(
            f"Invalid device type: {device}. Must be one of int, str, torch.device"
        )

    return device


def default_devices(devices: Optional[List[Union[str, int, torch.device]]] = None):
    if devices is None:
        N_devices = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(N_devices)]

    devices = [convert_to_device(device) for device in devices]
    return devices


def rank_to_device(rank: int) -> torch.device:
    return torch.device(f"cuda:{rank}")


def device_to_rank(device: Union[int, str, torch.device]) -> int:
    if isinstance(device, int):
        rank = device
    elif isinstance(device, str):
        rank = int(device.split(":")[-1])
    elif isinstance(device, torch.device):
        rank = device.index
    else:
        raise ValueError(
            f"Invalid device type: {device}. Must be one of int, str, torch.device"
        )

    return rank
