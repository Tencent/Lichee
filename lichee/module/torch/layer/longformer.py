# -*- coding: utf-8 -*-
import torch

from lichee.module.torch.layer.longformer_multi_headed_attn import LongformerSelfAttention
from lichee.module.torch.layer.transformer import TransformerSelfOutput, TransformerIntermediate, TransformerOutput


class Longformer(torch.nn.Module):
    """longformer model

    Attributes
    ----------
    attention: LongformerAttention
        longformer self-attention layer
    intermediate: LongformerIntermediate
        longformer intermediate layer
    output: LongformerOutput
        longformer output layer

    """

    def __init__(self, cfg, layer_id=0):
        super(Longformer, self).__init__()
        self.attention = LongformerAttention(cfg, layer_id)
        self.intermediate = LongformerIntermediate(cfg)
        self.output = LongformerOutput(cfg)

    def forward(self, hidden_states, attention_mask):
        """forward function of longformer model

        Parameters
        ----------
        hidden_states: torch.Tensor
            hidden states
        attention_mask: torch.Tensor
            attention masks

        Returns
        ------
        layer_output: torch.Tensor
            the output of longformer model
        """
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class LongformerAttention(torch.nn.Module):
    def __init__(self, cfg, layer_id=0):
        super(LongformerAttention, self).__init__()
        self.self = LongformerSelfAttention(cfg, layer_id)
        self.output = LongformerSelfOutput(cfg)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class LongformerSelfOutput(TransformerSelfOutput):
    def __init__(self, cfg):
        super(LongformerSelfOutput, self).__init__(cfg)


class LongformerIntermediate(TransformerIntermediate):
    def __init__(self, cfg):
        super(LongformerIntermediate, self).__init__(cfg)


class LongformerOutput(TransformerOutput):
    def __init__(self, cfg):
        super(LongformerOutput, self).__init__(cfg)
