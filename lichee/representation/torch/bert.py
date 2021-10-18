# -*- coding: utf-8 -*-
import collections
import re

import torch

from lichee import plugin
from lichee.module.torch.layer.embedding import BERTEmbedding, BertEmbeddingMixGrained
from lichee.module.torch.layer.transformer import Transformer
from lichee.representation import representation_base
from lichee.representation.torch.common import d_c as bert_default_config, load_pretrained_model_default, \
                                                        state_dict_remove_pooler_default


@plugin.register_plugin(plugin.PluginType.REPRESENTATION, "bert")
class BertRepresentation(representation_base.BaseRepresentation):
    """
    Bidirectional Encoder Representations from Transformers. Lichee Pretrained Models are
    available for both normal mode and mix grained mode and support different scenes, including
    comment, title, documents and others.

    Parameters
    ----------
    input_ids: torch.Tensor (dtype: torch.int64)
        Input_ids have different shapes in different mode.
        In normal mode:
            Input shape could be {Batch Size, 3, Max Length of Sequence}
            Token_ids, Segment_ids and attention masks should be placed in order.
        In mixed grained mode:
            Input shape could be {Batch Size, 4, Max Length of Sequence}
            Token_ids, Coarse_token_ids, Segment_ids and attention masks should be placed in order.
        Important Configurations:
            MIX_GRAINED:
                default is FALSE, set TRUE if mix_grained is needed
            NUM_HIDDEN_LAYERS:
                number of encoders, should match the pretrained model if PRETRAINED is TRUE.
            NUM_ATTENTION_HEADS:
                number of heads in one encoders, should match the pretrained model PRETRAINED is TRUE.

    Returns
    -------
    all_encoder_layers: torch.Tensor (dtype: torch.float32)
        Outputs of all encoder layers and their shapes depend on HIDDEN_SIZE in CONFIG and
        max length of input sequences.
    sequence_output: torch.Tensor (dtype: torch.float32)
        The last layer of all_encoder_layers, usually used for tasks.
    """
    def __init__(self, representation_cfg):
        super(BertRepresentation, self).__init__(representation_cfg)
        self.set_config_default()

        self.mix_grained = self.representation_cfg["CONFIG"]["MIX_GRAINED"]

        if self.mix_grained:
            self.embeddings = BertEmbeddingMixGrained(self.representation_cfg)
        else:
            self.embeddings = BERTEmbedding(self.representation_cfg)
        self.encoder = BERTEncoder(self.representation_cfg)

    def set_config_default(self):
        if "CONFIG" not in self.representation_cfg:
            self.representation_cfg["CONFIG"] = {}

        d_c = bert_default_config
        for key, value in d_c.items():
            if key not in self.representation_cfg["CONFIG"]:
                self.representation_cfg["CONFIG"][key] = value

    def forward(self, input_ids):
        input_ids = input_ids.permute([1, 0, 2])
        input_ids = input_ids.to(torch.int64)

        if self.mix_grained:
            token_ids = input_ids[0]
            coarse_token_ids = input_ids[1]
            token_ids = torch.stack([token_ids, coarse_token_ids])
            segment_ids = input_ids[2]
            attention_mask = input_ids[3]
        else:
            token_ids = input_ids[0]
            segment_ids = input_ids[1]
            attention_mask = input_ids[2]

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(token_ids, segment_ids)
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        return all_encoder_layers, sequence_output

    @classmethod
    def load_pretrained_model(cls, representation_cfg, pretrained_model_path):
        model = cls(representation_cfg)
        state_dict = torch.load(pretrained_model_path, map_location='cpu')
        state_dict = cls.state_dict_remove_pooler(state_dict)
        # Strict可以Debug参数
        model.load_state_dict(state_dict, strict=True)
        return model

    @classmethod
    def state_dict_remove_pooler(cls, model_weight):
        new_state_dict = state_dict_remove_pooler_default(model_weight)
        return new_state_dict


class BERTEncoder(torch.nn.Module):
    def __init__(self, representation_cfg):
        super(BERTEncoder, self).__init__()
        num_hidden_layers = representation_cfg["CONFIG"]["NUM_HIDDEN_LAYERS"]
        self.layer = torch.nn.ModuleList([Transformer(representation_cfg) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTPooler(torch.nn.Module):
    def __init__(self, representation_cfg):
        super(BERTPooler, self).__init__()
        self.dense = torch.nn.Linear(representation_cfg["CONFIG"]["HIDDEN_SIZE"],
                                     representation_cfg["CONFIG"]["HIDDEN_SIZE"])
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
