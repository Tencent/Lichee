# -*- coding: utf-8 -*-
import collections
import re

import torch

from lichee import plugin
from lichee.representation import representation_base
from lichee.module.torch.layer.embedding import BERTEmbedding, BertEmbeddingMixGrained
from lichee.module.torch.layer.longformer import Longformer
from lichee.representation.torch.common import d_c as bert_default_config, load_pretrained_model_default, \
                                                        state_dict_remove_pooler_default


@plugin.register_plugin(plugin.PluginType.REPRESENTATION, "docbert")
class DocBertRepresentation(representation_base.BaseRepresentation):
    """docbert is an implementation of LongFormer network.
    LongFormer provides attention mechanism that scales linearly with sequence length,
    making it easy to process documents of thousands of tokens or longer.
    It consistently outperforms RoBERTa on long document tasks and
    sets new state-of-the-art results on WikiHop and TriviaQA.
    Details of LongFormer can be referred `here <https://arxiv.org/abs/2004.05150>`_.

    Attributes
    ----------
    mix_grained: bool
        whether docbert uses mix-grained input
    embeddings: Optional[BertEmbeddingMixGrained, BERTEmbedding]
        docbert embedding layer
    encoder: DocBERTEncoder
        docbert encoder layer

    """

    def __init__(self, representation_cfg):
        super(DocBertRepresentation, self).__init__(representation_cfg)
        self.set_config_default()

        self.mix_grained = self.representation_cfg["CONFIG"]["MIX_GRAINED"]

        if self.mix_grained:
            self.embeddings = BertEmbeddingMixGrained(self.representation_cfg)
        else:
            self.embeddings = BERTEmbedding(self.representation_cfg)
        self.encoder = DocBERTEncoder(self.representation_cfg)

    def set_config_default(self):
        """set default config if config item does not exist

        """
        if "CONFIG" not in self.representation_cfg:
            self.representation_cfg["CONFIG"] = {}

        d_c = bert_default_config
        for key, value in d_c.items():
            if key not in self.representation_cfg["CONFIG"]:
                self.representation_cfg["CONFIG"][key] = value

    def forward(self, input_ids):
        """forward function of docbert model

        Parameters
        ----------
        input_ids: torch.Tensor
            input tokens after bert tokenization

        Returns
        ------
        all_encoder_layers: List[torch.Tensor]
            the output of each docbert self-attention block
        sequence_output: torch.Tensor
            the whole output of docbert encoder, which is the output of last self-attention block
        """
        input_ids = input_ids.permute([1, 0, 2])
        input_ids = input_ids.to(torch.int64)

        if self.mix_grained:
            token_ids, coarse_token_ids, segment_ids, attention_mask = input_ids
            token_ids = torch.stack([token_ids, coarse_token_ids])
        else:
            token_ids, segment_ids, attention_mask = input_ids

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

        state_dict = torch.load(pretrained_model_path,
                                map_location='cpu')

        state_dict = cls.remove_bert_words(state_dict)
        state_dict = cls.converter_gamma_to_weight(state_dict)
        state_dict = cls.state_dict_remove_pooler(state_dict)

        # Strict可以Debug参数
        model.load_state_dict(state_dict, strict=False)
        return model

    @classmethod
    def teg_state_dict_convert(cls, model_weight):
        new_state_dict = collections.OrderedDict()
        for k, v in model_weight.items():
            if 'target.' in k:
                continue
            k = re.sub('embedding.word_embedding', 'embeddings.word_embeddings', k)
            k = re.sub('embedding.position_embedding', 'embeddings.position_embeddings', k)
            k = re.sub('embedding.segment_embedding', 'embeddings.token_type_embeddings', k)
            k = re.sub('embedding.layer_norm', 'embeddings.LayerNorm', k)
            k = re.sub('encoder.transformer', 'encoder.layer', k)
            k = re.sub('self_attn.linear_layers.0', 'attention.self.query', k)
            k = re.sub('self_attn.linear_layers.1', 'attention.self.key', k)
            k = re.sub('self_attn.linear_layers.2', 'attention.self.value', k)
            k = re.sub('self_attn.final_linear', 'attention.output.dense', k)
            k = re.sub('layer_norm_1', 'attention.output.LayerNorm', k)
            k = re.sub('self_attn.final_linear', 'attention.output.dense', k)
            k = re.sub('feed_forward.linear_1', 'intermediate.dense', k)
            k = re.sub('feed_forward.linear_2', 'output.dense', k)
            k = re.sub('layer_norm_2', 'output.LayerNorm', k)
            new_state_dict[k] = v
        # self.model.load_state_dict(new_state_dict, strict = False)
        return new_state_dict

    @classmethod
    def state_dict_remove_pooler(cls, model_weight):
        new_state_dict = state_dict_remove_pooler_default(model_weight)
        return new_state_dict

    # 移除bert 关键字
    @classmethod
    def remove_bert_words(cls, model_weight):
        new_state_dict = collections.OrderedDict()
        for k, v in model_weight.items():
            k = re.sub('^bert\.', '', k)
            new_state_dict[k] = v
        return new_state_dict

    # 将 LayerNorm weight,bias 等词变为 LayerNorm gamma\beta
    @classmethod
    def converter_gamma_to_weight(cls, model_weight):
        old_keys = []
        new_keys = []
        for key in model_weight.keys():
            new_key = None
            if "LayerNorm.weight" in key:
                new_key = key.replace("LayerNorm.weight", "LayerNorm.gamma")
            if "LayerNorm.bias" in key:
                new_key = key.replace("LayerNorm.bias", "LayerNorm.beta")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            model_weight[new_key] = model_weight.pop(old_key)
        return model_weight


class DocBERTEncoder(torch.nn.Module):
    """docbert encoder layer

    Attributes
    ----------
    layer: torch.nn.ModuleList
        self-attention layers of docbert model

    """

    def __init__(self, representation_cfg):
        super(DocBERTEncoder, self).__init__()
        num_hidden_layers = representation_cfg["CONFIG"]["NUM_HIDDEN_LAYERS"]
        self.layer = torch.nn.ModuleList([Longformer(representation_cfg, i) for i in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        """forward function of docbert encoder layer

        Parameters
        ----------
        hidden_states: torch.Tensor
            hidden states
        attention_mask: torch.Tensor
            attention masks

        Returns
        -------
        all_encoder_layers: List[torch.Tensor]
            the output of each docbert self-attention block
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class DocBERTPooler(torch.nn.Module):
    """docbert pooler layer

    Attributes
    ----------
    dense: torch.nn.Linear
        dense layer of pooler
    activation: torch.nn.Tanh
        activation layer of pooler
    """

    def __init__(self, representation_cfg):
        super(DocBERTPooler, self).__init__()
        self.dense = torch.nn.Linear(representation_cfg["CONFIG"]["HIDDEN_SIZE"],
                                     representation_cfg["CONFIG"]["HIDDEN_SIZE"])
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        """forward function of docbert pooler layer

        Parameters
        ----------
        hidden_states: torch.Tensor
            hidden states

        Returns
        -------
        pooled_output: torch.Tensor
            the output of docbert pooler layer
        """
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
