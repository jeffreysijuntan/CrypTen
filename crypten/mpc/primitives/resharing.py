#!/usr/bin/env python3

import crypten.communicator as comm
import torch


def replicate_shares(x_share):
    rank = comm.get().get_rank()
    world_size = comm.get().get_world_size()
    prev_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size

    x_share = x_share.contiguous()
    x_rep = torch.zeros_like(x_share)

    send_group = getattr(comm.get(), f"group{rank}{next_rank}")
    recv_group = getattr(comm.get(), f"group{prev_rank}{rank}")

    comm.get().broadcast(x_share, src=rank, group=send_group)
    comm.get().broadcast(x_rep, src=prev_rank, group=recv_group)

    return x_rep


def __replicated_secret_sharing_protocol(op, x, y, *args, **kwargs):
    assert op in {
        "mul",
        "matmul",
        "conv1d",
        "conv2d",
        "conv_transpose1d",
        "conv_transpose2d",
    }
    from .arithmetic import ArithmeticSharedTensor
    from .binary import BinarySharedTensor

    x1, x2 = x.share, replicate_shares(x.share)
    y1, y2 = y.share, replicate_shares(y.share)

    z = getattr(torch, op)(x1, y1, *args, **kwargs)
    z += getattr(torch, op)(x1, y2, *args, **kwargs)
    z += getattr(torch, op)(x2, y1, *args, **kwargs)

    rank = comm.get().get_rank()
    if isinstance(x, BinarySharedTensor):
        z = BinarySharedTensor.from_shares(z, src=rank)
    elif isinstance(x, ArithmeticSharedTensor):
        z = ArithmeticSharedTensor.from_shares(z, src=rank)

    return z


def mul(x, y):
    return __replicated_secret_sharing_protocol("mul", x, y)


def matmul(x, y):
    return __replicated_secret_sharing_protocol("matmul", x, y)


def conv1d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv1d", x, y, **kwargs)


def conv2d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv2d", x, y, **kwargs)


def conv_transpose1d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv_transpose1d", x, y, **kwargs)


def conv_transpose2d(x, y, **kwargs):
    return __replicated_secret_sharing_protocol("conv_transpose2d", x, y, **kwargs)


def square(x):
    from .arithmetic import ArithmeticSharedTensor

    x1, x2 = x.share, replicate_shares(x.share)
    x_square = x1 ** 2 + 2 * x1 * x2

    return ArithmeticSharedTensor.from_shares(x_square, src=comm.get().get_rank())


def AND(x, y):
    from .binary import BinarySharedTensor

    x1, x2 = x.share, replicate_shares(x.share)
    y1, y2 = y.share, replicate_shares(y.share)

    res = (x1 & y1) ^ (x2 & y1) ^ (x1 & y2)
    return BinarySharedTensor.from_shares(res, src=comm.get().get_rank())
