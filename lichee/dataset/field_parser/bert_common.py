# -*- coding: utf-8 -*-
import math

import torch

from lichee.module.torch.layer.tokenizer import tokenizer_bert
from .field_parser_base import BaseFieldParser

SPLIT_SEG = ","


def prepare_bert_mix_grained_text(parser, text, text_pair: bool):
    src, coarse_src, seg = parser.tokenizer.tokenize(text, text_pair=text_pair)

    token_ids = src[:parser.global_config.DATASET.CONFIG.MAX_SEQ_LEN]
    coarse_token_ids = coarse_src[:parser.global_config.DATASET.CONFIG.MAX_SEQ_LEN]
    segment_ids = seg[:parser.global_config.DATASET.CONFIG.MAX_SEQ_LEN]
    attn_masks = [1] * len(token_ids)
    token_ids = torch.LongTensor([token_ids, coarse_token_ids, segment_ids, attn_masks])
    return token_ids


def collate_bert_mix_grained_text(parser, batch):
    record = {}
    key = parser.alias

    batch_token_ids = [instance[key][0] for instance in batch]
    batch_coarse_token_ids = [instance[key][1] for instance in batch]
    batch_segment_ids = [instance[key][2] for instance in batch]
    batch_attn_masks = [instance[key][3] for instance in batch]

    # Dynamci Max Len
    max_len = max([len(token_ids) for token_ids in batch_token_ids])
    for i in range(len(batch)):
        pad_len = max_len - batch_token_ids[i].shape[0]
        if pad_len > 0:
            z = torch.zeros([pad_len], dtype=torch.long)
            batch_token_ids[i] = torch.cat([batch_token_ids[i], z], dim=0)
            batch_coarse_token_ids[i] = torch.cat([batch_coarse_token_ids[i], z], dim=0)
            batch_segment_ids[i] = torch.cat([batch_segment_ids[i], z], dim=0)
            batch_attn_masks[i] = torch.cat([batch_attn_masks[i], z], dim=0)

    batch_token_ids = torch.stack(batch_token_ids)
    batch_coarse_token_ids = torch.stack(batch_coarse_token_ids)
    batch_segment_ids = torch.stack(batch_segment_ids)
    batch_attn_masks = torch.stack(batch_attn_masks)

    batch_token_ids = torch.stack(
        [batch_token_ids, batch_coarse_token_ids, batch_segment_ids, batch_attn_masks])
    batch_token_ids = batch_token_ids.permute([1, 0, 2])

    record[key] = batch_token_ids

    org_key = "org_" + parser.alias
    if org_key in batch[0]:
        batch_texts = [instance[org_key] for instance in batch]
        record[org_key] = batch_texts
    return record


def collate_bert_text(parser, batch):
    record = {}
    key = parser.alias

    batch_token_ids = [instance[key][0] for instance in batch]
    batch_segment_ids = [instance[key][1] for instance in batch]
    batch_attn_masks = [instance[key][2] for instance in batch]

    # Dynamci Max Len
    max_len = max([len(token_ids) for token_ids in batch_token_ids])
    for i in range(len(batch)):
        concate_bert_text(max_len, batch_token_ids, batch_segment_ids, batch_attn_masks, i, False)

    batch_token_ids = torch.stack(batch_token_ids)
    batch_segment_ids = torch.stack(batch_segment_ids)
    batch_attn_masks = torch.stack(batch_attn_masks)

    batch_token_ids = torch.stack([batch_token_ids, batch_segment_ids, batch_attn_masks])
    batch_token_ids = batch_token_ids.permute([1, 0, 2])

    record[key] = batch_token_ids

    org_key = "org_" + parser.alias
    if org_key in batch[0]:
        batch_texts = [instance[org_key] for instance in batch]
        record[org_key] = batch_texts
    return record


def prepare_bert_text(parser, tokens, segment_ids):
    tokens = tokens[:parser.cfg["MAX_SEQ_LEN"]]
    token_ids = parser.tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = segment_ids[:parser.cfg["MAX_SEQ_LEN"]]
    attn_masks = [1] * len(tokens)
    token_ids = torch.LongTensor([token_ids, segment_ids, attn_masks])
    return token_ids


def collate_docbert_text(parser, batch):
    """collate input collections into batched input

    Parameters
    ----------
    batch: List[Dict]
        a collection of model inputs

    Returns
    ------
    record: Dict
        collated input after padding and batching
    """
    record = {}
    key = parser.alias

    batch_token_ids = [instance[key][0] for instance in batch]
    batch_segment_ids = [instance[key][1] for instance in batch]
    batch_attn_masks = [instance[key][2] for instance in batch]

    # Dynamci Max Len
    max_len = max([len(token_ids) for token_ids in batch_token_ids])
    # 求出比max_len 大的最小的512的倍数
    windows_size = parser.global_config.DATASET.CONFIG.DOCBERT_WINDOWS_SIZE
    if max_len <= parser.global_config.DATASET.CONFIG.MAX_SEQ_LEN:
        factor = math.ceil(max_len / windows_size)
    else:
        factor = parser.global_config.DATASET.CONFIG.MAX_SEQ_LEN // windows_size

    max_len = windows_size * factor
    for i in range(len(batch)):
        concate_bert_text(max_len, batch_token_ids, batch_segment_ids, batch_attn_masks, i, True)

    batch_token_ids = torch.stack(batch_token_ids)
    batch_segment_ids = torch.stack(batch_segment_ids)
    batch_attn_masks = torch.stack(batch_attn_masks)

    batch_token_ids = torch.stack([batch_token_ids, batch_segment_ids, batch_attn_masks])
    batch_token_ids = batch_token_ids.permute([1, 0, 2])

    record[key] = batch_token_ids

    org_key = "org_" + parser.alias
    if org_key in batch[0]:
        batch_texts = [instance[org_key] for instance in batch]
        record[org_key] = batch_texts
    return record


def concate_bert_text(max_len, batch_token_ids, batch_segment_ids, batch_attn_masks, i, set_longformer_attention=True):
    pad_len = max_len - batch_token_ids[i].shape[0]
    if pad_len > 0:
        z = torch.zeros([pad_len], dtype=torch.long)
        batch_token_ids[i] = torch.cat([batch_token_ids[i], z], dim=0)
        batch_segment_ids[i] = torch.cat([batch_segment_ids[i], z], dim=0)
        batch_attn_masks[i] = torch.cat([batch_attn_masks[i], z], dim=0)
        if set_longformer_attention:
            batch_attn_masks[i][0] = 2


class BertTextPairFieldParserCommon(BaseFieldParser):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.parser_name = ""

    def init(self, cfg):
        super().init(cfg)
        self.tokenizer = tokenizer_bert.TokenizerBert(cfg)

    def set_key(self, alias, keys: str):
        self.alias = alias.strip()
        self.key = keys.strip().split(SPLIT_SEG)
        if len(self.key) != 2:
            raise Exception("Unsupported key %s in %s" % (keys, self.parser_name))

    def parse(self, row, training=False):
        record = {}
        value_arr = []
        for key in self.key:
            if key not in row:
                raise Exception("Cannot find key %s in row by %s" % (key, self.parser_name))
            value_arr.append(bytes(row[key]).decode("utf-8"))

        record[self.alias] = self.prepare_text(value_arr)
        record["org_" + self.alias] = SPLIT_SEG.join(value_arr)
        return record

    def prepare_text(self, text):
        text_a, text_b = text
        tokens_a = ["[CLS]"] + self.tokenizer.tokenize(text_a) + ["[SEP]"]
        tokens_b = self.tokenizer.tokenize(text_b) + ["[SEP]"]
        tokens = tokens_a + tokens_b
        if self.cfg["TYPE_VOCAB_SIZE"] == 3:
            segment_ids = [1] * len(tokens_a) + [2] * len(tokens_b)
        else:
            segment_ids = [0] * len(tokens_a) + [1] * len(tokens_b)

        return prepare_bert_text(self, tokens, segment_ids)

    def collate(self, batch):
        return collate_bert_text(self, batch)


class BertTextFieldParserCommon(BaseFieldParser):
    """field parser for bert model

    Attributes
    ----------
    tokenizer: tokenizer_bert.TokenizerBert
        bert tokenizer
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.parser_name = ""

    def init(self, cfg):
        super().init(cfg)
        self.tokenizer = tokenizer_bert.TokenizerBert(cfg)

    def parse(self, row, training=False):
        """parse each row into docbert output

        Parameters
        ----------
        row: Dict[str, str]
            input data row
        training: bool
            whether in training or eval mode

        Returns
        ------
        record: Dict
            parsed input data, containing both raw text or tokenized result
        """
        record = {}
        if self.key not in row:
            raise Exception("Cannot find key %s in row by %s" % (self.key, self.parser_name))

        text = bytes(row[self.key]).decode("utf-8")
        record[self.alias] = self.prepare_text(text)
        record["org_" + self.alias] = text
        return record

    def prepare_text(self, text):
        tokens = ["[CLS]"] + self.tokenizer.tokenize(text) + ["[SEP]"]
        segment_ids = [1] * len(tokens)

        return prepare_bert_text(self, tokens, segment_ids)

    def collate(self, batch):
        return collate_bert_text(self, batch)
