# -*- coding: utf-8 -*-
from lichee import plugin
from lichee.module.torch.layer.tokenizer import tokenizer_bert_mix_grained
from .bert_common import collate_bert_mix_grained_text, prepare_bert_mix_grained_text
from .field_parser_base import BaseFieldParser

SPLIT_SEG = ","


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "bert_mix_grained_text_pair")
class BertMixGrainedTextFieldParser(BaseFieldParser):
    """field parser for BERT mix-grained input pair

    Attributes
    ----------
    tokenizer: tokenizer_bert_mix_grained.TokenizerBertMixgrained
        BERT mix-grained tokenizer

    """
    def __init__(self):
        super().__init__()
        self.tokenizer = None

    def init(self, cfg):
        super().init(cfg)
        self.tokenizer = tokenizer_bert_mix_grained.TokenizerBertMixgrained(cfg)

    def set_key(self, alias, keys: str):
        self.alias = alias.strip()
        self.key = keys.strip().split(SPLIT_SEG)
        if len(self.key) != 2:
            raise Exception("Unsupported key %s in bert_mix_grained_text_pair" % keys)

    def parse(self, row, training=False):
        record = {}
        value_arr = []
        for key in self.key:
            if key not in row:
                raise Exception("Cannot find key %s in row by bert_mix_grained_text_pair" % key)
            value_arr.append(bytes(row[key]).decode("utf-8"))

        record[self.alias] = prepare_bert_mix_grained_text(self, value_arr, True)
        record["org_" + self.alias] = SPLIT_SEG.join(value_arr)
        return record

    def collate(self, batch):
        return collate_bert_mix_grained_text(self, batch)
