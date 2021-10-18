# -*- coding: utf-8 -*-
from lichee import config


class BaseFieldParser:
    def __init__(self):
        self.cfg = None
        self.alias = None
        self.key = None
        self.global_config = config.get_cfg()

    def init(self, cfg):
        '''
        :param cfg: field specified config
        :return:
        '''
        self.cfg = cfg

    def set_key(self, alias, key):
        '''
        :param alias: field alias
        :param key: field key
        :return:
        '''
        self.alias = alias
        self.key = key

    def parse(self, record, training=False):
        """
        将单行原始数据解析为训练所需数据
        :param record: 单行数据，dict类型
        :param training: 判断是否处于训练阶段，默认值 false
        :return:
        """
        raise NotImplementedError

    def collate(self, batch):
        """
        组batch逻辑
        :param batch: 需要组batch的数据，list类型，list内每个item为parse的输出
        :return: batch数据
        """
        raise NotImplementedError
