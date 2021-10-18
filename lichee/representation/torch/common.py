# -*- coding: utf-8 -*-
import re
import torch
import collections

# bert/ docbert default configuration
d_c = {'MIX_GRAINED': False,
       'MIX_MODE': "basic",
       'ATTENTION_PROBS_DROPOUT_PROB': 0.1,
       'HIDDEN_ACT': 'gelu',
       'HIDDEN_DROPOUT_PROB': 0.1,
       'HIDDEN_SIZE': 768,
       'NUM_HIDDEN_LAYERS': 12,
       'INITIALIZER_RANGE': 0.02,
       'INTERMEDIATE_SIZE': 3072,
       'MAX_POSITION_EMBEDDINGS': 512,
       'NUM_ATTENTION_HEADS': 12}

# load pretrained model method
def load_pretrained_model_default(cls, representation_cfg, pretrained_model_path):
    """load pre-trained model from specified path

            Parameters
            ----------
            representation_cfg: Dict
                model config
            pretrained_model_path: str
                path of pre-trained model

            Returns
            -------
            model: DocBertRepresentation
                model loaded from pretrained_model_path
            """
    model = cls(representation_cfg)

    state_dict = torch.load(pretrained_model_path,
                            map_location='cpu')

    state_dict = cls.remove_bert_words(state_dict)
    state_dict = cls.converter_gamma_to_weight(state_dict)
    state_dict = cls.state_dict_remove_pooler(state_dict)

    # Strict可以Debug参数
    model.load_state_dict(state_dict, strict=False)
    return model


def state_dict_remove_pooler_default(model_weight):
    new_state_dict = collections.OrderedDict()
    for k, v in model_weight.items():
        if 'target.' in k:
            continue
        if 'pooler.dense' in k:
            continue
        k = re.sub('^module.', '', k)
        new_state_dict[k] = v
    return new_state_dict
