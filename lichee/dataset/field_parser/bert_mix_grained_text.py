# -*- coding: utf-8 -*-
from lichee import plugin
from lichee.module.torch.layer.tokenizer import tokenizer_bert_mix_grained
from .bert_common import collate_bert_mix_grained_text, prepare_bert_mix_grained_text
from .field_parser_base import BaseFieldParser


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "bert_mix_grained_text")
class BertMixGrainedTextFieldParser(BaseFieldParser):
    """field parser for BERT mix-grained input

    Attributes
    ----------
    user_key: str
        key in output fields
    tokenizer: tokenizer_bert_mix_grained.TokenizerBertMixgrained
        BERT mix-grained tokenizer

    """
    def __init__(self):
        super().__init__()
        self.user_key = []
        self.tokenizer = None

    def init(self, cfg):
        super().init(cfg)
        self.tokenizer = tokenizer_bert_mix_grained.TokenizerBertMixgrained(cfg)

    def parse(self, row, training=False):
        record = {}
        if self.key not in row:
            raise Exception("Cannot find key %s in row by bert_mix_grained_text" % self.key)

        text = bytes(row[self.key]).decode("utf-8")
        record[self.alias] = prepare_bert_mix_grained_text(self, text, False)
        record["org_" + self.alias] = text
        return record

    def collate(self, batch):
        return collate_bert_mix_grained_text(self, batch)
