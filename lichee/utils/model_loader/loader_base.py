# -*- coding: utf-8 -*-
import torch


class BaseLoader(torch.nn.Module):
    """
    base model loader implementation

    """
    def __init__(self, model_path):
        super(BaseLoader, self).__init__()

        self.model_path = model_path

    def forward(self, inputs):
        pass
