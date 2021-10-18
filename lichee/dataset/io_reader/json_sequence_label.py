# -*- coding: utf-8 -*-
import json

from lichee import plugin
from lichee.utils import common
from lichee.utils.tfrecord import TFRecordWriter
from .io_reader_base import BaseIOReader


@plugin.register_plugin(plugin.PluginType.DATA_IO_READER, "json_sequence_label")
class JsonSequenceReader(BaseIOReader):
    @classmethod
    def convert_to_tfrecord(cls, data_file, description_file):
        """
        convert json_sequence data to tfrecord
        :param data_file:
        :param description_file:
        :return:
        """
        tfrecord_data_file = data_file + ".tfrecord"

        if common.create_local_file_spinlock(tfrecord_data_file):
            lower = True
            writer = TFRecordWriter(tfrecord_data_file)
            with open(data_file, 'r', newline="\n", encoding='utf8') as f:
                # read in json and process different parts by keys
                for line in f:
                    line = json.loads(line.strip())
                    text = line['text']
                    # Todo
                    if len(text) != len(text.strip()):
                        continue
                    if lower:
                        text = text.lower()
                    label_entities = line.get('label', None)
                    # set the placeholder for labels
                    labels = ['O'] * len(text)
                    if label_entities is not None:
                        for key, value in label_entities.items():
                            for sub_name, sub_index in value.items():
                                for start_index, end_index in sub_index:
                                    if lower:
                                        sub_name = sub_name.lower()
                                    assert text[start_index:end_index + 1] == sub_name
                                    # modify the key content by adding prefix
                                    if start_index == end_index:
                                        labels[start_index] = 'S-' + key
                                    else:
                                        labels[start_index] = 'B-' + key
                                        labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                    # write the tf_record by the keys "text_sl" and "labels_sl"
                    tf_record = {}
                    tf_record['text_sl'] = (text.encode('utf-8'), 'byte')
                    labels_str = '#'.join(labels)
                    tf_record['labels_sl'] = (labels_str.encode('utf-8'), 'byte')
                    writer.write(tf_record)
                writer.close()

        # check the spin lock for output file
        if common.get_local_file_spinlock(tfrecord_data_file):
            pass

        return tfrecord_data_file
