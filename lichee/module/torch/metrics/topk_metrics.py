# -*- coding: utf-8 -*-
import logging
import torch
import numpy as np

from lichee import plugin
from .metrics_base import BaseMetrics


@plugin.register_plugin(plugin.PluginType.MODULE_METRICS, "TOPK")
class TOPKMetrics(BaseMetrics):
    def __init__(self):
        super(TOPKMetrics, self).__init__()

    def calc(self, k=2):
        '''
        :param logits:
        logits = torch.tensor([[0.1, 0.3, 0.2, 0.4],
                       [0.5, 0.01, 0.9, 0.4],
                        [0.9, 0.02, 0.2, 0.6]])
        :param targets:
        targets = torch.tensor([3, 0, 1])
        目标结果的位置
        :param k:
        top k 的参数，一般取1， 2， 5，默认值为2
        :return:
        topk的准确率
        '''

        labels = np.concatenate(self.total_labels, axis=0)
        logits = np.concatenate(self.total_logits, axis=0)

        logits = torch.from_numpy(logits)
        targets = torch.from_numpy(labels)
        # 取出logits中topk的元素, indices为元素的位置
        # 结果为
        # logits=
        # ([[0.4000, 0.3000],
        # [0.9000, 0.5000],
        # [0.9000, 0.6000]])
        # indice=
        # ([[3, 1],
        # [2, 0],
        # [0, 3]])
        values, indices = torch.topk(logits, k=k, sorted=True)
        # 转为列模式，如([3], [0], [1])
        targets_y = torch.reshape(targets, [-1, 1])
        # 判断targets_y是否与indice相等
        # 结果为
        # ([[1., 0.],
        # [0., 1.],
        # [0., 0.]])
        # *1. 是因为mean只能处理float
        correct = (targets_y == indices) * 1.
        # 相当于 包含1的行数 / 总行数
        topk_accuracy = torch.mean(correct) * k
        logging.info(
            "top {} Acc, (Correct/Total): {:.4f} ({}/{})".format(k, topk_accuracy, torch.sum(correct), len(indices)))
        self.total_labels, self.total_logits = [], []
        return topk_accuracy

    def name(self):
        return "TOPK"
