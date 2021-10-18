# -*- coding: utf-8 -*-
import torch.nn as nn

from lichee import config
from lichee import plugin


class BaseTask(nn.Module):
    """BaseTask class
    provide get_output method, use in prediction,
    TASK_OUT will get from configuration.
    Base class of Task, others task class should derived from this.
    """
    def __init__(self):
        super(BaseTask, self).__init__()

    def forward(self, *args, label_inputs):
        """
        任务层模型实现
        :param args: 依赖的表示层的输出，按配置传入
        :param label_inputs: label数据，若模型调用时未传label数据，该数据为None
        :return: 训练时需返回 logits 和 loss；预测时仅返回 logits。
        """
        raise NotImplementedError("Not Implemented!")

    @classmethod
    def get_output(cls, logits):
        """

        :param logits:
        :return:
        """
        cfg = config.get_cfg()
        task_output_cls = plugin.get_plugin(plugin.PluginType.TASK_OUTPUT, cfg.MODEL.TASK.CONFIG.TASK_OUTPUT)
        return task_output_cls.get_output(logits)
