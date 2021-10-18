# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

from lichee import plugin
from lichee import config


@plugin.register_plugin(plugin.PluginType.MODULE_LOSS, 'mse_loss')
class MSELoss:

    @classmethod
    def build(cls, cfg):
        return nn.MSELoss()


@plugin.register_plugin(plugin.PluginType.MODULE_LOSS, 'cross_entropy')
class CrossEntropyLoss:

    @classmethod
    def build(cls, cfg):
        return nn.CrossEntropyLoss()


@plugin.register_plugin(plugin.PluginType.MODULE_LOSS, 'neg_log_likelihood')
class NLLLoss:

    @classmethod
    def build(cls, cfg):
        return nn.NLLLoss()


@plugin.register_plugin(plugin.PluginType.MODULE_LOSS, 'binary_cross_entropy')
class BinaryCrossEntropyLoss:

    @classmethod
    def build(cls, cfg):
        return nn.BCEWithLogitsLoss()


@plugin.register_plugin(plugin.PluginType.MODULE_LOSS, 'binary_focal_loss')
class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=[1.0, 1.0], gamma=2, ignore_index=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2))
            assert self.alpha.shape[0] == 2, \
                'the `alpha` shape is not match the number of class'
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

        self.one_hot_eye = None

    @classmethod
    def set_config_default(cls, cfg):
        d_c = {'loss_alpha': [1.0, 1.0],
               'loss_gamma': 2,
               'loss_ignore_index': None,
               'loss_reduction': 'mean'}
        for key, value in d_c.items():
            if key not in cfg.PARAM:
                cfg.PARAM[key] = value

    @classmethod
    def build(cls, cfg):
        cls.set_config_default(cfg)
        return cls(alpha=cfg.PARAM["loss_alpha"],
                   gamma=cfg.PARAM["loss_gamma"],
                   ignore_index=cfg.PARAM["loss_ignore_index"],
                   reduction=cfg.PARAM["loss_reduction"])

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        if self.one_hot_eye == None:
            self.one_hot_eye = torch.eye(2).cuda(target.device.index)
        target = self.one_hot_eye[target]

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * \
                   torch.log(torch.sub(1.0, prob)) * neg_mask

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
        return loss


@plugin.register_plugin(plugin.PluginType.MODULE_LOSS, 'focal_loss')
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=[0.25, 0.75], gamma=2, balance_index=-1, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float, int)):
            assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
            assert balance_index > -1
            alpha = torch.ones((self.num_class))
            alpha *= 1 - self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    @classmethod
    def set_config_default(cls, cfg):
        d_c = {'loss_alpha': [0.25, 0.75],
               'loss_gamma': 2,
               'loss_balance_index': -1,
               'loss_size_average': True}
        for key, value in d_c.items():
            if key not in cfg.PARAM:
                cfg.PARAM[key] = value

    @classmethod
    def build(cls, cfg):
        cls.set_config_default(cfg)
        return cls(num_class=config.get_cfg().DATASET.CONFIG.NUM_CLASS,
                   alpha=cfg.PARAM["loss_alpha"],
                   gamma=cfg.PARAM["loss_gamma"],
                   balance_index=cfg.PARAM["loss_balance_index"],
                   size_average=cfg.PARAM["loss_size_average"])

    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps  # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            alpha = self.alpha.to(logpt.device)
            alpha_class = alpha.gather(0, target.view(-1))
            logpt = alpha_class * logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


