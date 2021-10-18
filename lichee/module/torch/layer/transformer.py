# -*- coding: utf-8 -*-
import torch

from lichee import plugin
from lichee.module.torch.layer.multi_head_attention import MultiHeadedAttention
from lichee.module.torch.layer.normalization import LayerNorm


class Transformer(torch.nn.Module):
    """Transformer model

    """
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.attention = TransformerAttention(cfg)
        self.intermediate = TransformerIntermediate(cfg)
        self.output = TransformerOutput(cfg)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class TransformerAttention(torch.nn.Module):
    """Transformer attention

    """
    def __init__(self, cfg):
        super(TransformerAttention, self).__init__()
        self.self = MultiHeadedAttention(cfg)
        self.output = TransformerSelfOutput(cfg)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class TransformerSelfOutput(torch.nn.Module):
    """Transformer self-output

    """
    def __init__(self, cfg):
        super(TransformerSelfOutput, self).__init__()
        self.dense = torch.nn.Linear(cfg["CONFIG"]["HIDDEN_SIZE"], cfg["CONFIG"]["HIDDEN_SIZE"])
        self.LayerNorm = LayerNorm(cfg["CONFIG"]["HIDDEN_SIZE"])
        self.dropout = torch.nn.Dropout(cfg["CONFIG"]["HIDDEN_DROPOUT_PROB"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerIntermediate(torch.nn.Module):
    """Transformer intermediate

    """
    def __init__(self, cfg):
        super(TransformerIntermediate, self).__init__()
        self.dense = torch.nn.Linear(cfg["CONFIG"]["HIDDEN_SIZE"], cfg["CONFIG"]["INTERMEDIATE_SIZE"])
        self.intermediate_act_fn = plugin.get_plugin(plugin.PluginType.MODULE_LAYER, cfg["CONFIG"]["HIDDEN_ACT"])

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TransformerOutput(torch.nn.Module):
    """Transformer output

    """
    def __init__(self, cfg):
        super(TransformerOutput, self).__init__()
        self.dense = torch.nn.Linear(cfg["CONFIG"]["INTERMEDIATE_SIZE"], cfg["CONFIG"]["HIDDEN_SIZE"])
        self.LayerNorm = LayerNorm(cfg["CONFIG"]["HIDDEN_SIZE"])
        self.dropout = torch.nn.Dropout(cfg["CONFIG"]["HIDDEN_DROPOUT_PROB"])

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
