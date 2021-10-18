# -*- coding: utf-8 -*-
import logging

from lichee.dataset.bert_constants import *
from lichee.utils import storage
from . import tokenizer_utils
from .seg_utils import SegUtils
from .tokenizer_base import BaseTokenizer
from .tokenizer_bert import BasicTokenizer, WordpieceTokenizer


class TokenizerBertMixgrained(BaseTokenizer):
    """define bert mix-grained tokenizer

    Attributes
    ----------
    vocab: tokenizer_utils.Vocab
        bert vocab
    coarse_vocab: tokenizer_utils.CoarseVocab
        bert coarse vocab
    max_seq_len: int
        max sequence length
    seg_util: SegUtils
        qqseg instance
    basic_tokenizer: BasicTokenizer
        bert basic tokenizer
    wordpiece_tokenizer: WordpieceTokenizer
        bert wordpiece tokenizer

    """
    def __init__(self, cfg):
        super().__init__()
        self.vocab = tokenizer_utils.Vocab()
        self.vocab.load(self.global_config.MODEL.CONFIG.VOCAB_PATH)

        self.coarse_vocab = tokenizer_utils.CoarseVocab()
        self.coarse_vocab.load(self.global_config.MODEL.CONFIG.COARSE_VOCAB_PATH, index_offset=len(self.vocab))

        self.max_seq_len = self.global_config.DATASET.CONFIG.MAX_SEQ_LEN

        qq_seg_path = storage.get_storage_file(self.global_config.RUNTIME.CONFIG.QQSEG_PATH)
        self.seg_util = SegUtils(qq_seg_path)

        do_lower_case = True
        # if config.get_attr('do_lower_case'):
        #     do_lower_case = config.do_lower_case

        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

        # TODO
        self.wwm_mode = 'basic'
        self.debug = False
        self.test_load = 0

    def tokenize(self, text, text_pair=False):
        if text_pair:
            text_a, text_b = text
            text = text_a + '\t' + text_b
            span_bio_a, tokens_a = self.get_wwm_bio(text_a)
            src_a = [self.vocab.get(t) for t in tokens_a]
            src_a = [CLS_ID] + src_a
            span_bio_a = [(idx + 1, sp_len) for idx, sp_len in span_bio_a]
            span_bio_a.insert(0, (0, 1))

            span_bio_b, tokens_b = self.get_wwm_bio(text_b)
            src_b = [self.vocab.get(t) for t in tokens_b]
            src_b = [SEP_ID] + src_b + [SEP_ID]
            src = src_a + src_b

            span_bio_b = [(idx + len(src_a) + 1, sp_len) for idx, sp_len in span_bio_b]
            span_bio_b.insert(0, (len(src_a), 1))
            span_bio_b.append((len(src_a) + len(src_b) - 1, 1))

            span_bio = span_bio_a + span_bio_b

            seg = [1] * len(src_a) + [2] * len(src_b)
        else:
            span_bio, tokens = self.get_wwm_bio(text)
            src = [self.vocab.get(t) for t in tokens]
            src = [CLS_ID] + src
            span_bio = [(idx + 1, sp_len) for idx, sp_len in span_bio]
            span_bio.insert(0, (0, 1))
            seg = [1] * len(src)
            # print(span_bio, tokens, src)

        coarse_src = []
        for index_set in span_bio:
            i = index_set[0]

            span_len = index_set[1]
            tokens = []
            for j in range(span_len):
                token = src[i + j]
                tokens.append(str(token))
            coarse_key = ','.join(tokens)
            if coarse_key in self.coarse_vocab.w2i:
                for j in range(span_len):
                    coarse_src.append(self.coarse_vocab.w2i[coarse_key])
            else:
                for j in range(span_len):
                    coarse_src.append(src[i + j])

        if len(src) > self.max_seq_len:
            src = src[:self.max_seq_len]
            coarse_src = coarse_src[:self.max_seq_len]
            seg = seg[:self.max_seq_len]
        # while len(src) < self.max_seq_len:
        #    src.append(0)
        #    coarse_src.append(PAD_ID)
        #    seg.append(0)

        return src, coarse_src, seg

    # ------DEFINE YOUR SEGMENT LOGIC------#
    def proc_seg(self, clean_text):
        if self.wwm_mode == 'basic':
            w_words = [w for w in self.seg_util.basic_seg(clean_text) if w != ' ']
        elif self.wwm_mode == 'mix':
            w_words = [w for w in self.seg_util.mix_seg(clean_text) if w != ' ']
        elif self.wwm_mode == 'jieba':
            # print('proc jieba cut')
            import jieba
            w_words = [w for w in jieba.cut(clean_text) if w != ' ']
        else:
            return None
        return w_words

    def get_wwm_bio(self, text):
        words = self.basic_tokenizer.tokenize(text, cn_space_split=False)
        clean_text = " ".join(words)

        # DEFINE YOUR SEGMENT LOGIC
        w_words = self.proc_seg(clean_text)
        if self.debug:
            print(w_words)

        span_bio = []
        step = 0
        tokens = []
        for w_word in w_words:
            if w_word.strip() == '':
                continue
            span_len = 0
            if not self.is_all_chinese(w_word):
                for sub_word in self.wordpiece_tokenizer.tokenize(w_word):
                    tokens.append(sub_word)
                    span_len += 1
            else:
                for char in w_word:
                    tokens.append(char)
                span_len += len(w_word)

            span_bio.append((step, span_len))
            step += span_len
        if self.debug:
            print(span_bio)
        return span_bio, tokens

    def is_all_chinese(self, text):
        is_all_chinese = True
        for char in text:
            cp = ord(char)
            if not self.basic_tokenizer._is_chinese_char(cp):
                return False
        return True

    def get_vocab(self):
        return [self.vocab, self.coarse_vocab]

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab.w2i[token])
        if len(ids) > self.max_seq_len:
            logging.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids
