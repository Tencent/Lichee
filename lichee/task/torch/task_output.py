# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import softmax
import torch

from lichee import plugin


class BaseTaskOutput:
    @classmethod
    def get_outputs(cls, raw_outputs):
        """
        对任务层插件的输出进行后处理
        :param raw_outputs: 任务层输出 logits
        :return: 后处理结果
        """
        raise NotImplementedError


@plugin.register_plugin(plugin.PluginType.TASK_OUTPUT, "simple_cls_out")
class SimpleClsOut(BaseTaskOutput):
    """simple_cls_out give max_probs and preds as output
        max prob use softmax
    """
    @classmethod
    def get_output(cls, raw_outputs):
        if isinstance(raw_outputs[0], torch.Tensor):
            raw_outputs[0] = raw_outputs[0].cpu().numpy()
        probs = softmax(raw_outputs[0], axis=-1)
        max_probs = np.max(probs, axis=-1)
        preds = np.argmax(probs, axis=-1)
        return max_probs, preds


@plugin.register_plugin(plugin.PluginType.TASK_OUTPUT, "distill_cls_out")
class DistillClsOut(BaseTaskOutput):
    """distill_cls_out give logits and preds as output
        max prob use softmax
    """
    @classmethod
    def get_output(cls, raw_outputs):
        if isinstance(raw_outputs[0], torch.Tensor):
            raw_outputs[0] = raw_outputs[0].cpu().numpy()
        probs = softmax(raw_outputs[0], axis=-1)
        preds = np.argmax(probs, axis=-1)
        return raw_outputs[0], preds
