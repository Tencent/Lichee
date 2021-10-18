# -*- coding: utf-8 -*-
from lichee import plugin
from .bert_common import BertTextPairFieldParserCommon


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "bert_text_pair")
class BertTextFieldParser(BertTextPairFieldParserCommon):
    """field parser for BERT input pair

    """
    def __init__(self):
        super().__init__()
        self.parser_name = "bert_text_pair"
