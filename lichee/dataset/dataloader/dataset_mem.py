# -*- coding: utf-8 -*-
from abc import ABC

import torch.utils.data

from lichee import plugin
from lichee.utils.tfrecord.reader import read_single_record_with_spec_index
from .dataset_base import BaseDataset


@plugin.register_plugin(plugin.PluginType.DATA_LOADER, "dataset_mem")
class DatasetMem(torch.utils.data.Dataset, BaseDataset, ABC):
    def __init__(self, cfg, data_file, desc_file, training=True):
        super().__init__(cfg, data_file, desc_file, training)

    def __getitem__(self, index):
        """
        get transformed data with index

        :param index: data index
        :return: transformed data
        """
        data_file_index, (start_offset, end_offset) = self.get_nth_data_file(index)
        tfrecord_data_file = self.tfrecord_data_file_list[data_file_index]
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset, self.description)
        return self.transform(row)
