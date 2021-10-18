# -*- coding: utf-8 -*-
import collections
import unicodedata
import os
from lichee.utils import storage

import six

from lichee.dataset.bert_constants import *


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def convert_to_unicode(text):
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


class Vocab(object):
    """
    """

    def __init__(self):
        self.w2i = collections.OrderedDict()
        self.i2w = collections.OrderedDict()

    def load(self, vocab_path):
        vocab_path = storage.get_storage_file(vocab_path)
        with open(vocab_path, mode="r", encoding="utf-8") as reader:
            for index, line in enumerate(reader):
                w = line.strip()
                self.w2i[w] = index
                self.i2w[index] = w

    def get(self, w):
        return self.w2i.get(w, UNK_ID)

    def __getitem__(self, w):
        return self.w2i.get(w, UNK_ID)

    def __len__(self):
        return len(self.i2w)


class CoarseVocab(object):
    """
    """

    def __init__(self):
        self.w2i = collections.OrderedDict()
        self.i2w = collections.OrderedDict()

    def load(self, coarse_vocab_path, index_offset=0):
        coarse_vocab_path = storage.get_storage_file(coarse_vocab_path)
        with open(coarse_vocab_path, mode="r", encoding="utf-8") as reader:
            for index, line in enumerate(reader):
                char_index_key = line.strip().split('||')[0]
                self.w2i[char_index_key] = index_offset + index
                self.i2w[index_offset + index] = line.strip()

    def get(self, w):
        return self.w2i.get(w, UNK_ID)

    def __getitem__(self, w):
        return self.w2i.get(w, UNK_ID)

    def __len__(self):
        return len(self.i2w)


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab
