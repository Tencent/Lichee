# -*- coding: utf-8 -*-
import logging
import torch

from lichee.dataset.dataloader import data_builder
from lichee.utils import storage


# common init gpu setting
def init_gpu_setting_default(self):
    # cpu or cuda
    self.use_cuda = torch.cuda.is_available()

    if self.cfg.RUNTIME.GPU_IDS == "auto":
        gpu_num = torch.cuda.device_count()
        self.gpu_ids = list(range(gpu_num))
        logging.info("gpu set auto, finding %s gpus, let's use them", gpu_num)
    elif self.cfg.RUNTIME.GPU_IDS != "auto" and self.cfg.RUNTIME.GPU_IDS != "":
        self.gpu_ids = [int(gpu_id) for gpu_id in self.cfg.RUNTIME.GPU_IDS.split(",")]
        logging.info("gpu is set to %s, let's use them", self.gpu_ids)
    else:
        self.use_cuda = False


# common init dataloader
def init_dataloader_default(self):
    eval_data_path_list = []
    # use eval data
    for eval_data_path in self.cfg.DATASET.EVAL_DATA.DATA_PATH:
        eval_data_path_list.append(storage.get_storage_file(eval_data_path))

    eval_desc_data_path = storage.get_storage_file(self.cfg.DATASET.DESC_PATH)
    if 'DESC_PATH' in self.cfg.DATASET.EVAL_DATA:
        eval_desc_data_path = storage.get_storage_file(self.cfg.DATASET.EVAL_DATA.DESC_PATH)

    self.eval_dataloader = data_builder.build_dataloader(self.cfg.DATASET.EVAL_DATA,
                                                         eval_data_path_list,
                                                         eval_desc_data_path,
                                                         use_cuda=True,
                                                         training=False,
                                                         shuffle=False)


# common get inputs batch
def get_inputs_batch_default(self, batch):
    inputs = {}
    for key in self.model_inputs:
        if key not in batch:
            continue
        inputs[key] = batch[key].cuda(self.master_gpu_id) if self.use_cuda else batch[key]
    return inputs


# common get label batch
def get_label_batch_default(self, batch):
    label_keys = []
    # graph
    for graph_cfg in self.cfg.MODEL.GRAPH:
        if "LABELS" in graph_cfg:
            label_keys.extend(graph_cfg["LABELS"].split(','))

    labels = []
    for key in label_keys:
        label = batch[key].cuda(self.master_gpu_id) if self.use_cuda else batch[key]
        labels.append(label)
    return label_keys, labels
