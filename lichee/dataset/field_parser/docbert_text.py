# -*- coding: utf-8 -*-
from lichee import plugin
from .bert_common import BertTextFieldParserCommon, collate_docbert_text


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "docbert_text")
class BertTextFieldParser(BertTextFieldParserCommon):
    """field parser for docbert input

    """
    def __init__(self):
        super().__init__()
        self.parser_name = "docbert_text"

    def collate(self, batch):
        return collate_docbert_text(self, batch)
