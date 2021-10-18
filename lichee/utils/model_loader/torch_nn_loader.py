# -*- coding: utf-8 -*-
from lichee import plugin
from . import loader_base


@plugin.register_plugin(plugin.PluginType.UTILS_MODEL_LOADER, "torch_nn")
class TorchNNLoader(loader_base.BaseLoader):
    """provides ability to load model from torch nn format

    Attributes
    ----------
    model: torch.nn.Module
        loaded model instance

    """

    def __init__(self, model_path):
        super(TorchNNLoader, self).__init__(model_path)
        import torch
        self.model = torch.load(model_path)
        self.model.eval()

    def forward(self, inputs):
        # with torch.cuda.amp.autocast():
        return self.model(inputs)
