# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

from lichee import config


class BaseTokenizer(ABC):
    """define base class of tokenizer

    Attributes
    ----------
    global_config: yacs.config.CfgNode
        global config

    """

    def __init__(self):
        self.global_config = config.get_cfg()

    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def convert_tokens_to_ids(self, tokens):
        pass
