# -*- coding: utf-8 -*-
import os

import torch

from torch import distributed as dist


# init process group
def init_dist(backend='nccl', **kwargs):
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    dist.init_process_group(backend, **kwargs)


# get rank and world_size in DDP mode
def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_dict_from_list(src_list):
    # for example, src_list = ['a', 0.5]
    assert len(src_list) % 2 == 0, "list has odd length: {}; it must be a list of pairs".format(src_list)

    dst_dict = {}
    for src_key, src_value in zip(src_list[0::2], src_list[1::2]):
        dst_dict[src_key] = src_value

    return dst_dict
