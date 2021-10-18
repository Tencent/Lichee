# -*- coding: utf-8 -*-
import torch

from lichee import plugin
from lichee.representation import representation_base


@plugin.register_plugin(plugin.PluginType.REPRESENTATION, "context_gating")
class ContextGating(representation_base.BaseRepresentation):
    """
    context_gating representation
    
    """
    def __init__(self, representation_cfg) -> None:
        super().__init__(representation_cfg)
        feature_size = representation_cfg['FEATURE_SIZE']
        self.add_batch_norm = representation_cfg['ADD_BATCH_NORM']
        self.gating_weights = torch.nn.Linear(in_features=feature_size,
                                              out_features=feature_size,
                                              bias=False)
        self.bn = torch.nn.BatchNorm1d(num_features=feature_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        gates = self.gating_weights(x)

        if self.add_batch_norm:
            gates = self.bn(gates)

        gates = self.sigmoid(gates)

        activation = torch.mul(x, gates)

        return activation
