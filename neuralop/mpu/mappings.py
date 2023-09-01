# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.distributed as dist
import neuralop.mpu.comm as comm

# helper functions
from .helpers import _reduce
from .helpers import _split
from .helpers import _gather


# generalized
class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region."""
    @staticmethod
    def symbolic(graph, input_, comm_id_):
        return input_

    @staticmethod
    def forward(ctx, input_, comm_id_):
        ctx.comm_id = comm_id_
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if comm.is_distributed(ctx.comm_id):
            return _reduce(grad_output, group=comm.get_group(ctx.comm_id)), None
        else:
            return grad_output, None 


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region."""
    
    @staticmethod
    def symbolic(graph, input_, comm_id_):
        if comm.is_distributed(comm_id_):
            return _reduce(input_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def forward(ctx, input_, comm_id_):
        if comm.is_distributed(comm_id_):
            return _reduce(input_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

    
class _ScatterToParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_, comm_id_):
        return _split(input_, dim_, group=comm.get_group(comm_id_))

    @staticmethod
    def forward(ctx, input_, dim_, comm_id_):
        ctx.dim = dim_
        ctx.comm_id = comm_id_
        if comm.is_distributed(comm_id_):
            return _split(input_, dim_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output):
        if comm.is_distributed(ctx.comm_id):
            return _gather(grad_output, ctx.dim, group=comm.get_group(ctx.comm_id)), None, None
        else:
            return grad_output, None, None


class _GatherFromParallelRegion(torch.autograd.Function):
    """Gather the input from parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, dim_, comm_id_):
        if comm.is_distributed(comm_id_):
            return _gather(input_, dim_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def forward(ctx, input_, dim_, comm_id_):
        ctx.dim = dim_
        ctx.comm_id = comm_id_
        if comm.is_distributed(comm_id_):
            return _gather(input_, dim_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output):
        if comm.is_distributed(ctx.comm_id):
            return _split(grad_output, ctx.dim, group=comm.get_group(ctx.comm_id)), None, None
        else:
            return grad_output, None, None

    
# -----------------
# Helper functions.
# -----------------
# general
def copy_to_parallel_region(input_, comm_name):
    return _CopyToParallelRegion.apply(input_, comm_name)

def reduce_from_parallel_region(input_, comm_name):
    return _ReduceFromParallelRegion.apply(input_, comm_name)

def scatter_to_parallel_region(input_, dim, comm_name):
    return _ScatterToParallelRegion.apply(input_, dim, comm_name)

def gather_from_parallel_region(input_, dim, comm_name):
    return _GatherFromParallelRegion.apply(input_, dim, comm_name)