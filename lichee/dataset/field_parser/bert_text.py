# -*- coding: utf-8 -*-
from lichee import plugin
from .bert_common import BertTextFieldParserCommon


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "bert_text")
class BertTextFieldParser(BertTextFieldParserCommon):
    """field parser for BERT input

    """
    def __init__(self):
        super().__init__()
        self.parser_name = "bert_text"
