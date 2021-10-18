# -*- coding: utf-8 -*-
import concurrent.futures
from abc import ABCMeta
from concurrent.futures import as_completed
from typing import List

import cgroup_parser

from lichee import plugin
from lichee.dataset.io_reader.io_reader_base import BaseIOReader


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, cfg, data_path_list: List[str], desc_file, training=True):
        """
        Base dataset implementation

        :param cfg: dataset config
        :param data_path_list: dataset file path list
        :param desc_file: dataset description file
        :param training: training or not => shuffle or not
        """
        self.cfg = cfg
        self.data_path_list = data_path_list
        self.desc_file = desc_file
        self.training = training
        # 初始化数据
        self.tfrecord_data_file_list = self.try_convert_to_tfrecord()
        # init description config
        self.description = self.get_desc()
        # init data index
        self.data_index_list = self.get_indexes()
        # init dataset length
        self.data_len = self.get_data_len()
        # 初始化parser
        self.parsers = []
        self.init_parser()
        # 初始化标签
        self.label_keys = []
        self.init_labels()

    def init_parser(self):
        for field_parser in self.cfg.DATASET.FIELD:
            parser_cls = plugin.get_plugin(plugin.PluginType.FIELD_PARSER, field_parser["NAME"])
            parser = parser_cls()
            parser.init(field_parser) # init with field config

            if "KEY" not in field_parser or not field_parser["KEY"]:
                raise Exception("KEY missing for field: {}".format(field_parser['NAME']))

            keys_ori = field_parser["KEY"]
            if "ALIAS" in field_parser:
                alias_str = field_parser["ALIAS"]
            else:
                alias_str = field_parser["KEY"]

            if not alias_str:
                raise Exception('KEY or ALIAS misssing for field: {}'.format(field_parser['NAME']))

            parser.set_key(alias_str, keys_ori)
            self.parsers.append(parser)

    def init_labels(self):
        for graph_cfg in self.cfg.MODEL.GRAPH:
            if "LABELS" in graph_cfg:
                self.label_keys.extend(graph_cfg["LABELS"].split(','))

    def get_indexes(self):
        if "INDEX_LOADER_NUM" not in self.cfg.DATASET or self.cfg.DATASET.INDEX_LOADER_NUM == 0:
            self.cfg.DATASET.INDEX_LOADER_NUM = cgroup_parser.get_max_procs()
        max_workers = min(self.cfg.DATASET.INDEX_LOADER_NUM, len(self.data_path_list))
        reader_cls: BaseIOReader = plugin.get_plugin(plugin.PluginType.DATA_IO_READER, self.cfg.DATASET.FORMAT)

        data_index_list = [None] * len(self.data_path_list)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            fs = {executor.submit(reader_cls.get_index, data_path, self.desc_file): i for i, data_path in
                  enumerate(self.data_path_list)}
            for future in as_completed(fs):
                data_index_list[fs[future]] = future.result()
        return data_index_list

    def get_nth_data_file(self, index):
        '''
        :param index: index of target item
        :return: item file index, and item start & end offset
        '''
        for i, data_index in enumerate(self.data_index_list):
            if index < len(data_index):
                break
            index -= len(data_index)
        start_offset = data_index[index]
        end_offset = data_index[index + 1] if index + 1 < len(data_index) else None
        return i, (start_offset, end_offset)

    def get_data_len(self):
        data_len = 0
        for data_index in self.data_index_list:
            data_len += len(data_index)
        return data_len

    def get_desc(self):
        reader_cls: BaseIOReader = plugin.get_plugin(plugin.PluginType.DATA_IO_READER, self.cfg.DATASET.FORMAT)
        return reader_cls.get_desc(self.desc_file)

    def try_convert_to_tfrecord(self):
        tfrecord_data_file_list = []
        for data_path in self.data_path_list:
            reader_cls: BaseIOReader = plugin.get_plugin(plugin.PluginType.DATA_IO_READER, self.cfg.DATASET.FORMAT)
            tfrecord_data_file_list.append(reader_cls.convert_to_tfrecord(data_path, self.desc_file))
        return tfrecord_data_file_list

    def __len__(self):
        return self.data_len

    def transform(self, row):
        """
        transform data with field parsers

        :param row: the raw feature map {key1: raw_feature1, key2:raw_feature2...}
        :return: parsed feature map {key1:feature1, key2:feature2...}
        """
        record = {}

        for parser in self.parsers:
            parser_result = parser.parse(row, self.training)
            if parser_result is not None:
                record.update(parser_result)

        return record

    def collate(self, batch):
        """
        collate data in a batch

        :param batch: list of item feature map [item1, item2, item3 ...], item with {key1:feature1, key2:feature2}
        :return: batched feature map for model with format {key1: batch_feature1, key2: batch_feature2...}
        """
        record = {}

        for parser in self.parsers:
            collate_result = parser.collate(batch)
            if collate_result is not None:
                record.update(collate_result)

        return record
