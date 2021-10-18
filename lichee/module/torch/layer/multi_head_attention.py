# -*- coding: utf-8 -*-
import math

import torch


class MultiHeadedAttention(torch.nn.Module):

    def __init__(self, cfg):
        super(MultiHeadedAttention, self).__init__()
        if cfg["CONFIG"]["HIDDEN_SIZE"] % cfg["CONFIG"]["NUM_ATTENTION_HEADS"] != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (cfg["CONFIG"]["HIDDEN_SIZE"], cfg["CONFIG"]["NUM_ATTENTION_HEADS"]))
        self.num_attention_heads = cfg["CONFIG"]["NUM_ATTENTION_HEADS"]
        self.attention_head_size = int(cfg["CONFIG"]["HIDDEN_SIZE"] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(cfg["CONFIG"]["HIDDEN_SIZE"], self.all_head_size)
        self.key = torch.nn.Linear(cfg["CONFIG"]["HIDDEN_SIZE"], self.all_head_size)
        self.value = torch.nn.Linear(cfg["CONFIG"]["HIDDEN_SIZE"], self.all_head_size)

        self.dropout = torch.nn.Dropout(cfg["CONFIG"]["ATTENTION_PROBS_DROPOUT_PROB"])

        self.softmax = torch.nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer
