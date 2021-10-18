# -*- coding: utf-8 -*-
import json
import torch
from lichee import plugin
from lichee.module.torch.layer.tokenizer import tokenizer_bert
from .field_parser_base import BaseFieldParser


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, 'sequence_label_parser')
class BertTextFieldParser(BaseFieldParser):
    def __init__(self):
        super().__init__()
        self.user_key = []
        self.tokenizer = None

    def init(self, cfg):
        super().init(cfg)
        self.tokenizer = tokenizer_bert.CharTokenizer(cfg)
        self.inner_key = 'sequence_label_parser'
        # load seq label
        if 'LABEL_DICT' in self.global_config.DATASET.CONFIG:
            with open(self.global_config.DATASET.CONFIG["LABEL_DICT"]["PATH"], mode="r", encoding="utf-8") as f:
                l2i = json.load(f)
                l2i["[PAD]"] = len(l2i)

            self.l2i = l2i
            self.i2l = {}
            for k, v in self.l2i.items():
                self.i2l[v] = k

    def set_key(self, key):
        self.user_key.append(key)

    def parse(self, row, training=False):
        record = {}
        label_map = self.l2i
        MAX_SEQ_LEN = self.global_config.DATASET.CONFIG.MAX_SEQ_LEN
        row_text = bytes(row['text_sl']).decode("utf-8")
        row_label = bytes(row['labels_sl']).decode("utf-8").split('#')
        src_tokens = [t for t in self.tokenizer.tokenize(row_text)]
        src = self.tokenizer.convert_tokens_to_ids(src_tokens)
        labels = row_label
        tgt = [label_map[x] for x in labels]
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(src) > MAX_SEQ_LEN - special_tokens_count:
            src = src[: (MAX_SEQ_LEN - special_tokens_count)]
            tgt = tgt[: (MAX_SEQ_LEN - special_tokens_count)]
        src = self.tokenizer.convert_tokens_to_ids(['[CLS]']) + src + self.tokenizer.convert_tokens_to_ids(['[SEP]'])
        tgt = [label_map['O']] + tgt + [label_map['O']]
        seg = [1] * len(src)
        if len(src) != len(tgt):
            print(row)
            print(src, tgt)
            exit(-1)

        LABELS_NUM = self.global_config.DATASET.CONFIG["LABEL_DICT"]["LABELS_NUM"]
        while len(src) < MAX_SEQ_LEN:
            src.append(0)
            tgt.append(LABELS_NUM - 1)
            seg.append(0)
        record = {}
        record['src'] = src
        record['seg'] = seg
        record['text'] = row_text
        record['tgt'] = tgt

        return record

    def collate(self, batch):
        """Function for collate data batch max length, you can get Dynamic length rather than Static

        Parameters
        ----------
        batch: List
            list of various seq length
        Returns
        -------
        List
            batch with length of orginal max length in various seq length
        """
        record = {}
        batch_src = [instance["src"] for instance in batch]
        batch_seg = [instance["seg"] for instance in batch]
        batch_text = [instance["text"] for instance in batch]
        batch_src = torch.LongTensor(batch_src)
        batch_seg = torch.LongTensor(batch_seg)
        batch_mask = torch.LongTensor(batch_seg)
        batch_token_ids = torch.stack([batch_src, batch_seg, batch_mask])
        batch_token_ids = batch_token_ids.permute([1, 0, 2])
        record = {"text": batch_text, "token_ids": batch_token_ids}

        if 'tgt' in batch[0]:
            batch_tgt = torch.LongTensor([instance['tgt'] for instance in batch])
            record[self.inner_key] = batch_tgt
        return record
