# -*- coding: utf-8 -*-
import torch

from lichee import plugin
from .field_parser_base import BaseFieldParser

# parser used to parse the data with "soft_tgt"
@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "soft_tgt")
class SoftTargetLabelFieldParser(BaseFieldParser):
    def __init__(self):
        super().__init__()

    # parse data for processing
    def parse(self, row, training=False):
        record = {}
        if self.key not in row:
            return record

        # split the data string into list of string
        soft_tgt = bytes(row[self.key]).decode("utf-8").split(',')
        # convert string to float
        soft_tgt = [float(s) for s in soft_tgt]
        record[self.alias] = soft_tgt

        return record

    # collate data to get Dict as batch
    def collate(self, batch):
        record = {}
        for instance in batch:
            if self.alias not in instance:
                return record

        batch_labels = [instance[self.alias] for instance in batch]
        batch_labels = torch.FloatTensor(batch_labels)
        record[self.alias] = batch_labels
        return record
