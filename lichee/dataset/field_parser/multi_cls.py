# -*- coding: utf-8 -*-
import numpy as np
import torch
from lichee import plugin
from .field_parser_base import BaseFieldParser


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "multi_cls")
class MultiLabelFieldParser(BaseFieldParser):
    """
    field parser for multiple classification label

    """
    def __init__(self):
        super().__init__()

    def parse(self, row, training=False):
        record = {}
        if self.key not in row:
            raise Exception("Cannot find key %s in row by multi_cls" % self.key)

        label_raw = bytes(row[self.key]).decode("utf-8")
        multi_label_raw = label_raw.split(',')
        labels_one_hot = np.zeros([self.cfg['NUM_CLASS']])
        for label in multi_label_raw:
            label = int(label)
            if label > self.cfg['NUM_CLASS'] - 1:
                raise RuntimeError("data label is illegal, label > class num: " + str(label))
            labels_one_hot[label] = 1
        record[self.alias] = labels_one_hot
        return record

    def collate(self, batch):
        record = {}
        batch_labels = [instance[self.alias] for instance in batch]
        batch_labels = torch.LongTensor(batch_labels)
        record[self.alias] = batch_labels
        return record
