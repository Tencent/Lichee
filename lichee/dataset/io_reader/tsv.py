# -*- coding: utf-8 -*-
import os

from lichee import plugin
from lichee.utils import common
from lichee.utils.tfrecord import TFRecordWriter
from .io_reader_base import BaseIOReader


@plugin.register_plugin(plugin.PluginType.DATA_IO_READER, "tsv")
class TsvReader(BaseIOReader):
    @classmethod
    def convert_to_tfrecord(cls, data_file, description_file):
        """
        convert tsv to tfrecord
        :param data_file: str
        :param description_file: str
        :return: tfrecord file path
        """
        description = cls.get_desc(description_file)

        tfrecord_data_file = data_file + ".tfrecord"

        if os.path.exists(tfrecord_data_file):
            return tfrecord_data_file
        if common.create_local_file_spinlock(tfrecord_data_file):
            writer = TFRecordWriter(tfrecord_data_file)
            row_index = 0
            with open(data_file, 'r', newline="\n", encoding='utf8') as f:
                head_arr = f.readline().rstrip("\n").split('\t')
                head_arr = [x.strip() for x in head_arr]  # remove empty character
                row_index += 1
                for line in f:
                    row_index += 1
                    try:
                        data_arr = line.rstrip("\n").split('\t')
                        tfrecord_data = {}
                        for i, head_item in enumerate(head_arr):
                            tfrecord_data[head_item] = (data_arr[i].encode('utf-8'), description[head_item])
                        writer.write(tfrecord_data)
                    except Exception:
                        raise Exception("Exception occurs while scan file %s row-%s" % (data_file, row_index))
            writer.close()

        if common.get_local_file_spinlock(tfrecord_data_file):
            pass

        return tfrecord_data_file
