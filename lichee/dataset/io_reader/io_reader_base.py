# -*- coding: utf-8 -*-
import json
import os
from abc import ABC
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from lichee.utils import common
from lichee.utils.tfrecord import tfrecord_loader
from lichee.utils.tfrecord.tools import create_index


class BaseIOReader:
    """
    Base IO reader: transfer data files to tfrecord files
    """
    @classmethod
    def get_index(cls, data_file, description_file, index_file: str = "") -> List:
        """
        get index list

        :param data_file: str
        :param description_file: str
        :param index_file: str
        :return: index list
        """
        tfrecord_data_file = cls.convert_to_tfrecord(data_file, description_file)
        return TFRecordReader.get_index(tfrecord_data_file, description_file)

    @classmethod
    def get_iter(cls, data_file, description_file, index_file: str = "",
                 shard: Optional[Tuple[int, int]] = None) -> Iterable[Dict[str, np.ndarray]]:
        """
        get data iter with shard

        :param data_file: str
        :param description_file: str
        :param index_file: str(not be used by default)
        :param shard: Optional[Tuple[int, int]](None by default)
        :return: Iterable[Dict[str, np.ndarray]]
        """
        tfrecord_data_file = cls.convert_to_tfrecord(data_file, description_file)
        return TFRecordReader.get_iter(tfrecord_data_file, description_file, shard=shard)

    @classmethod
    def get_data(cls, data_file, description_file, index_file: str = "") -> List:
        """
        load the whole data in memory and return

        :param data_file: str
        :param description_file: str
        :param index_file: str
        :return: List
        """
        return list(cls.get_iter(data_file, description_file, index_file))

    @classmethod
    def get_desc(cls, description_file) -> Dict:
        """
        get data description

        :param description_file: str
        :return:
        """
        with open(description_file) as fh:
            description = json.load(fh)
        return description

    @classmethod
    def convert_to_tfrecord(cls, data_file, description_file):
        """
        转换 用户数据文件 为 tfrecord 文件

        :param data_file: 用户数据文件
        :param description_file: 描述文件
        :return: tfrecord 文件
        """
        raise NotImplementedError


class TFRecordReader(BaseIOReader):
    @classmethod
    def get_index(cls, data_file, description_file, index_file: str = "") -> List:
        _, index = cls.scan_or_create_index(data_file, index_file)
        return index

    @classmethod
    def get_iter(cls, data_file, description_file, index_file: str = "",
                 shard: Optional[Tuple[int, int]] = None) -> Iterable[Dict[str, np.ndarray]]:
        description = cls.get_desc(description_file)
        index_file, _ = cls.scan_or_create_index(data_file, index_file)
        return tfrecord_loader(data_file, index_file, description, shard)

    @classmethod
    def scan_or_create_index(cls, data_file, index_file: str = "") -> (str, int):
        if index_file == "":
            tmp_index_file = os.path.splitext(data_file)[0] + '.index'
            if common.create_local_file_spinlock(tmp_index_file):
                create_index(data_file, tmp_index_file)
            index_file = tmp_index_file

            if common.get_local_file_spinlock(tmp_index_file):
                pass
        index = np.loadtxt(index_file, dtype=np.int64)[:, 0]
        return index_file, index

    @classmethod
    def convert_to_tfrecord(cls, data_file, description_file):
        return data_file
