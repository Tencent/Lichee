# -*- coding: utf-8 -*-
from lichee import plugin
from .bert_common import BertTextPairFieldParserCommon, collate_docbert_text


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "docbert_text_pair")
class BertTextFieldParser(BertTextPairFieldParserCommon):
    """field parser for docbert input pair

    """
    def __init__(self):
        super().__init__()
        self.parser_name = "docbert_text_pair"

    def collate(self, batch):
        return collate_docbert_text(self, batch)
