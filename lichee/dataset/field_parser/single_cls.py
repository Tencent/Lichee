# -*- coding: utf-8 -*-
import torch
from lichee import plugin
from .field_parser_base import BaseFieldParser


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "single_cls")
class LabelFieldParser(BaseFieldParser):
    """
    field parser for single classification label

    """
    def __init__(self):
        super().__init__()

    def parse(self, row, training=False):
        record = {}
        if self.key not in row:
            raise Exception("Cannot find key %s in row by single_cls" % self.key)

        label = int(bytes(row[self.key]).decode("utf-8"))
        if label > self.cfg['NUM_CLASS'] - 1:
            raise RuntimeError("data label is illegal, label > class num: " + str(label))
        record[self.alias] = label
        return record

    def collate(self, batch):
        record = {}
        batch_labels = [instance[self.alias] for instance in batch]
        batch_labels = torch.LongTensor(batch_labels)
        record[self.alias] = batch_labels
        return record
