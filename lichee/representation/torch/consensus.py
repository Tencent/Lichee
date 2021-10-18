# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

from lichee import config
from lichee import plugin
from lichee.representation import representation_base


class NetVLADConsensus(torch.nn.Module):
    def __init__(self,
                 num_clusters,
                 alpha,
                 feature_size,
                 normalize_input=False) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.alpha = alpha
        self.feature_size = feature_size
        self.normalize_input = normalize_input

        self.conv = torch.nn.Conv1d(self.feature_size, self.num_clusters, kernel_size=1, bias=False)
        self.centroids = torch.nn.Parameter(torch.rand(self.num_clusters, self.feature_size), requires_grad=True)
        self._init_params()

    def _init_params(self):
        self.conv.weight = torch.nn.Parameter(
            (2.0 * 0.1 * self.centroids).unsqueeze(-1), requires_grad=True
        )
        self.conv.bias = torch.nn.Parameter(
            - 0.1 * self.centroids.norm(dim=1), requires_grad=True
        )

    def forward(self, x):
        # batch_size, num_segments, feature_size
        x = x.permute(0, 2, 1)

        # batch_size, feature_size, num_segments
        N, C = x.shape[:2]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # soft-assignment
        # batch_size, feature_size, num_segments
        conv_x = self.conv(x)

        # batch_size, num_clusters, num_segments
        soft_assign = conv_x.view(N, self.num_clusters, -1)

        # batch_size, num_clusters, num_segments
        soft_assign = F.softmax(soft_assign, dim=1)

        # batch_size, num_clusters, num_segments
        x_flatten = x.view(N, C, -1)

        # batch_size, feature_size, num_segments
        x_flatten_expand = x_flatten.expand(self.num_clusters, -1, -1, -1)

        # num_clusters, batch_size, feature_size, num_segments
        centroids_expand = self.centroids.expand(x_flatten.size(-1), -1, -1)

        # num_segments, num_clusters, feature_size
        x_flatten_expand_permute = x_flatten_expand.permute(1, 0, 2, 3)

        # batch_size, num_clusters, feature_size, num_segments
        centroids_expand_permute = centroids_expand.permute(1, 2, 0).unsqueeze(0)

        # 1,  num_clusters, feature_size, num_segments
        residual = x_flatten_expand_permute - centroids_expand_permute

        # batch_size, num_clusters, feature_size, num_segments
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        # batch_size, num_clusters, feature_size
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization

        # batch_size, num_clusters, feature_size
        vlad = vlad.view(N, -1)  # flatten

        # batch_size, num_clusters * feature_size
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        # batch_size, num_clusters * feature_size
        return vlad


@plugin.register_plugin(plugin.PluginType.REPRESENTATION, "nextvlad")
class NeXtVLAD(representation_base.BaseRepresentation):

    def __init__(self, representation_cfg) -> None:
        super().__init__(representation_cfg)
        self.feature_size = representation_cfg["FEATURE_SIZE"]
        self.output_size = representation_cfg["OUTPUT_SIZE"]
        self.expansion_size = representation_cfg["EXPANSION_SIZE"]
        self.cluster_size = representation_cfg["CLUSTER_SIZE"]
        self.groups = representation_cfg["NUM_GROUPS"]
        self.drop_rate = representation_cfg["DROPOUT_PROB"]

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)
        # self.apply(weights_init_kaiming)

    def forward(self, inputs):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad
