# -*- coding: utf-8 -*-
import torch


class BaseModel(torch.nn.Module):
    """
    base model structure implementation

    """
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg

        self.representation_model_inputs_arr = []
        self.representation_model_outputs_arr = []
        self.representation_model_arr = torch.nn.ModuleList()

    def set_requires_grad(self, requires_grad):
        for representation_model in self.representation_model_arr:
            for param in representation_model.parameters():
                param.requires_grad = requires_grad

    def forward(self, input_ids):
        raise NotImplementedError('not implemented!')
