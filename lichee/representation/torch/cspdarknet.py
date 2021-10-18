# -*- coding: utf-8 -*-
import collections
import re
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from lichee.module.torch.layer import brick as vn_layer
from lichee.module.torch.layer.det_conv_module import ConvModule
from lichee.module.torch.layer.det_resnet_block import ResBlock
from lichee import plugin
from lichee.representation import representation_base
from lichee.representation.torch.common import load_pretrained_model_default, \
                                                state_dict_remove_pooler_default


@plugin.register_plugin(plugin.PluginType.REPRESENTATION, "darknet")
class DarknetRepresentation(representation_base.BaseRepresentation):
    """Darknet backbone.
    Args:
        depth (int): Depth of Darknet. Currently only support 53.
        out_indices (Sequence[int]): Output from which stages
        
    Example:
        >>> from mmdet.models import Darknet
        >>> import torch
        >>> self = Darknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """

    # Dict(depth: (layers, channels))
    arch_settings = {
        53: ((1, 2, 8, 8, 4), ((32, 64), (64, 128), (128, 256), (256, 512),
                               (512, 1024)))
    }

    def __init__(self, representation_cfg):
        super(DarknetRepresentation, self).__init__(representation_cfg)

        self.depth = representation_cfg["CONFIG"]["DEPTH"]
        if self.depth not in self.arch_settings:
            raise KeyError(f'invalid depth {self.depth} for darknet')
        self.layers, self.channels = self.arch_settings[self.depth]
        self.out_indices = representation_cfg["CONFIG"]["OUT_INDICES"]
        norm_type = representation_cfg["CONFIG"]["NORM_TYPE"]
        act_type = representation_cfg["CONFIG"]["ACT_TYPE"]

        cfg = dict(norm_type=norm_type, act_type=act_type)

        self.conv1 = ConvModule(3, 32, 3, padding=1, **cfg)

        self.cr_blocks = ['conv1']
        for i, n_layers in enumerate(self.layers):
            layer_name = f'conv_res_block{i + 1}'
            in_c, out_c = self.channels[i]
            self.add_module(
                layer_name,
                self.make_conv_res_block(in_c, out_c, n_layers, **cfg))
            self.cr_blocks.append(layer_name)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self, layer_name)
            x = cr_block(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    @staticmethod
    def make_conv_res_block(in_channels,
                            out_channels,
                            res_repeat,
                            norm_type='BN', 
                            act_type='LeakyReLU'):
        """In Darknet backbone, ConvLayer is usually followed by ResBlock. This
        function will make that. The Conv layers always have 3x3 filters with
        stride=2. The number of the filters in Conv layer is the same as the
        out channels of the ResBlock.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            res_repeat (int): The number of ResBlocks.
        """

        cfg = dict(norm_type=norm_type, act_type=act_type)

        model = nn.Sequential()
        model.add_module(
            'conv',
            ConvModule(
                in_channels, out_channels, 3, stride=2, padding=1, **cfg))
        for idx in range(res_repeat):
            model.add_module('res{}'.format(idx), ResBlock(out_channels, **cfg))
        return model

    @classmethod
    def load_pretrained_model(cls, representation_cfg, pretrained_model_path):
        model = load_pretrained_model_default(cls, representation_cfg, pretrained_model_path)
        return model

    @classmethod
    def state_dict_remove_pooler(cls, model_weight):
        new_state_dict = state_dict_remove_pooler_default(model_weight)
        return new_state_dict


@plugin.register_plugin(plugin.PluginType.REPRESENTATION, "cspdarknet")
class CSPDarknetRepresentation(representation_base.BaseRepresentation):
    custom_layers = (vn_layer.Resblock_body, vn_layer.Resblock_body.custom_layers)

    def __init__(self, representation_cfg):
        super(CSPDarknetRepresentation, self).__init__(representation_cfg)
        self.inplanes = representation_cfg["CONFIG"]["INPUT_CHANNELS"]
        self.layers = representation_cfg["CONFIG"]["LAYER_NUMS"]
        self.conv1 = vn_layer.Conv2dBatchMish(3, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            vn_layer.Resblock_body(self.inplanes, self.feature_channels[0], self.layers[0], first=True),
            vn_layer.Resblock_body(self.feature_channels[0], self.feature_channels[1], self.layers[1], first=False),
            vn_layer.Resblock_body(self.feature_channels[1], self.feature_channels[2], self.layers[2], first=False),
            vn_layer.Resblock_body(self.feature_channels[2], self.feature_channels[3], self.layers[3], first=False),
            vn_layer.Resblock_body(self.feature_channels[3], self.feature_channels[4], self.layers[4], first=False)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.
        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self
        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential, CSPDarknetRepresentation.custom_layers)):
                yield from self.__modules_recurse(module)
            else:
                yield module

    def forward(self, x):
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return [out3, out4, out5]
    
    @classmethod
    def load_pretrained_model(cls, representation_cfg, pretrained_model_path):
        model = cls(representation_cfg)

        state_dict = torch.load(pretrained_model_path,
                                map_location='cpu')

        state_dict = cls.state_dict_remove_pooler(state_dict)

        # Strict可以Debug参数
        model.load_state_dict(state_dict, strict=True)
        return model

    @classmethod
    def state_dict_remove_pooler(cls, model_weight):
        new_state_dict = collections.OrderedDict()
        for k, v in model_weight.items():
            # removing pooler layer weight
            if 'target.' in k:
                continue
            if 'pooler.dense' in k:
                continue

            k = re.sub('^module.', '', k)
            new_state_dict[k] = v
        return new_state_dict
