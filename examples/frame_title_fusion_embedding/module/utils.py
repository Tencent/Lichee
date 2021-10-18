# -*- coding: utf-8 -*-
import logging

import six
from torch import nn
from torch import optim

from lichee import plugin


class ConfigOptim:

    @classmethod
    def config_from_dict(cls, dict_obj):
        optim = cls()
        for (key, value) in six.iteritems(dict_obj):
            if key in optim.__dict__:
                optim.__dict__[key] = value
        return optim


@plugin.register_plugin(plugin.PluginType.MODULE_OPTIMIZER, 'LayeredOptim')
class LayeredOptim(ConfigOptim):

    @classmethod
    def build(cls, model, cfg):
        '''
        :param model: the model
        :param cfg: the global config from your_config.yaml
        :return: optimizer
        '''
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        logging.info('num of parameters: {}'.format(len(parameters)))
        independent_lr_parameters = model.independent_lr_parameters()
        if independent_lr_parameters:  # deal with independent learning rate parameters
            ind_params = []
            for param_map in independent_lr_parameters:
                if 'base_lr' not in param_map and 'lr' in param_map:
                    param_map['base_lr'] = param_map['lr']
                ind_params += list(param_map['params'])
            normal_parameters = list(set(parameters).difference(set(ind_params)))
            param_list = independent_lr_parameters + [{"params": normal_parameters, 'lr': cfg.TRAINING.LEARNING_RATE}]
            logging.info('pretrain parameters: {}'.format(len(ind_params)))
        else:
            param_list = [
                {"params": parameters, 'lr': cfg.TRAINING.LEARNING_RATE, 'base_lr': cfg.TRAINING.LEARNING_RATE}]

        return optim.Adam(param_list)


@plugin.register_plugin(plugin.PluginType.MODULE_LOSS, 'BCELoss')
class BCELoss:

    @classmethod
    def build(cls, cfg):
        return nn.BCEWithLogitsLoss(reduction='sum')


@plugin.register_plugin(plugin.PluginType.MODULE_METRICS, 'PRScore')
class PRScore:
    def __init__(self):
        super().__init__()
        self.num_true = 0
        self.num_pred = 0
        self.num_target = 0

    def calc(self, threshold=0.5):
        # summarize the metrics
        res_info = {
            "precision": self.num_true / (self.num_pred + 1e-6),
            "recall": self.num_true / (self.num_target + 1e-6)
        }
        # the metric will reset after summary
        self.num_true = 0
        self.num_pred = 0
        self.num_target = 0
        return res_info

    def collect(self, labels, preds):
        '''
        :param labels: ground truth label
        :param preds: predictions
        :return:
        '''
        if isinstance(labels, list):
            assert len(labels) == 1
            labels = labels[0]
        logits, embeddings = preds
        labels = labels.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        assert len(labels) == len(logits)
        pred = logits > 0.5
        self.num_true += ((labels == pred) * labels).sum()
        self.num_pred += pred.sum()
        self.num_target += labels.sum()

    @staticmethod
    def name():
        return "Precison"
