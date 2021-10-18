# -*- coding: utf-8 -*-
import math

import torch

from lichee import plugin


@plugin.register_plugin(plugin.PluginType.MODULE_LAYER, 'gelu')
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


@plugin.register_plugin(plugin.PluginType.MODULE_LAYER, 'relu')
def relu(x):
    return torch.nn.functional.relu
