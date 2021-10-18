# -*- coding: utf-8 -*-
import torch


class BaseRepresentation(torch.nn.Module):
    """
    base representation implementation

    """
    def __init__(self, representation_cfg):
        super(BaseRepresentation, self).__init__()
        self.representation_cfg = representation_cfg

    def forward(self, *args, **kwargs):
        raise NotImplementedError('not implemented!')

    @classmethod
    def load_pretrained_model(cls, cfg, pretrained_model_path):
        pass

    def independent_lr_parameters(self):
        if "LEARNING_RATE" in self.representation_cfg:
            return [{'params': [x for x in self.parameters() if x.requires_grad],
                     'lr': self.representation_cfg['LEARNING_RATE']}]
        return []
