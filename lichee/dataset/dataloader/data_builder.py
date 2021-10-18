# -*- coding: utf-8 -*-
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from lichee import plugin
from lichee.config import get_cfg


class ChunkFileDataLoader:
    """Base dataloader implementation.

    Params:
    -------
    data_config: Dict
        dataloader config.
    data_path_list: List[str]
        list of dataset path.
    desc_path: str
        data description file.
    use_cuda: bool
        train with cuda.
    training: bool
        whether is in training.
    shuffle: bool
        should shuffle.
    """
    def __init__(self, data_config, data_path_list, desc_path, use_cuda, training, shuffle):
        cfg = get_cfg()
        self.cfg = cfg
        if shuffle:
            np.random.shuffle(data_path_list)
        self.dataset_cls = plugin.get_plugin(plugin.PluginType.DATA_LOADER, cfg.DATASET.NAME)
        self.data_config = data_config
        self.data_path_list = data_path_list
        self.desc_path = desc_path
        self.use_cuda = use_cuda
        self.shuffle = shuffle
        self.training = training
        self.num_files = len(self.data_path_list)
        # first initialize to get data nums
        dataset = self.dataset_cls(cfg,
                                   data_path_list,
                                   desc_path,
                                   training=training)
        self.dataset_num = len(dataset)
        del dataset
        self.current_loader = None
        self.current_idx = 0
        if 'CHUNK_SIZE' not in cfg.DATASET:
            self.chunk_size = self.num_files
        else:
            self.chunk_size = cfg.DATASET.CHUNK_SIZE
        assert self.chunk_size > 0

    def init_new_loader(self):
        if self.current_idx >= self.num_files:
            if self.shuffle:
                np.random.shuffle(self.data_path_list)
            self.current_idx = 0
            raise StopIteration
        to_load_files = self.data_path_list[self.current_idx: self.current_idx + self.chunk_size]
        dataset = self.dataset_cls(
            self.cfg,
            to_load_files,
            self.desc_path,
            training=self.training
        )
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             pin_memory=self.use_cuda,
                                             batch_size=self.data_config.BATCH_SIZE,
                                             collate_fn=dataset.collate,
                                             num_workers=self.data_config.WORKER_NUM,
                                             shuffle=self.shuffle
                                             )
        self.current_loader = loader.__iter__()
        self.current_idx += self.chunk_size

    def __next__(self):
        if self.current_loader is None:
            self.init_new_loader()
        try:
            return next(self.current_loader)
        except StopIteration:
            self.init_new_loader()
            return next(self.current_loader)

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.dataset_num / self.data_config.BATCH_SIZE)


def build_dataloader(
        data_config,
        data_path_list: List[str],
        desc_path: str,
        use_cuda: bool,
        training: bool,
        shuffle: bool) -> ChunkFileDataLoader:
    """
    build dp dataloader

    :param data_config: Dict
        dataloader config.
    :param data_path_list: List[str]
        list of dataset path.
    :param desc_path: str
        data description file.
    :param use_cuda: bool
        train with cuda.
    :param training: bool
        whether is in training.
    :param shuffle: bool
        should shuffle.
    :return: torch.utils.data.DataLoader
    """
    return ChunkFileDataLoader(data_config, data_path_list, desc_path, use_cuda, training, shuffle)


def build_ddp_dataloader(data_path_list: List[str],
                         desc_path: str,
                         use_cuda: bool,
                         training: bool,
                         shuffle: bool) -> torch.utils.data.DataLoader:
    """
    build ddp dataloader

    :param data_config: Dict
        dataloader config.
    :param data_path_list: List[str]
        list of dataset path.
    :param desc_path: str
        data description file.
    :param use_cuda: bool
        train with cuda.
    :param training: bool
        whether is in training.
    :param shuffle: bool
        should shuffle.
    :return: torch.utils.data.DataLoader
    """
    cfg = get_cfg()

    dataset_cls = plugin.get_plugin(plugin.PluginType.DATA_LOADER, cfg.DATASET.NAME)
    dataset = dataset_cls(cfg,
                          data_path_list,
                          desc_path,
                          shuffle=shuffle,
                          training=training)

    # TODO
    # sampler = None
    # if 'SAMPLER' in cfg.DATASET:
    #     sampler_cls = plugin.get_plugin(plugin.PluginType.SAMPLER, cfg.DATASET.SAMPLER)
    #     sampler = sampler_cls(dataset)

    sampler = torch.utils.data.DistributedSampler(dataset)

    return torch.utils.data.DataLoader(dataset=dataset,
                                       sampler=sampler,
                                       pin_memory=use_cuda,
                                       batch_size=cfg.DATASET.TRAIN_DATA.BATCH_SIZE,
                                       collate_fn=dataset.collate,
                                       num_workers=cfg.DATASET.TRAIN_DATA.WORKER_NUM,
                                       shuffle=False)
