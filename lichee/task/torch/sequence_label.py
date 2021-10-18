# -*- coding: utf-8 -*-
import torch

from lichee import config
from lichee import plugin
from lichee.module.torch.layer.crf import CRF
import torch.nn.functional as F


@plugin.register_plugin(plugin.PluginType.TASK, "sequence_label")
class SequenceLabel(torch.nn.Module):
    """SequenceLabel

    Attributes
    ----------
    output_layer: torch.nn.Liner
        feature layers of sequence model.
    crf: torch.nn.Module
        conditional random field of sequence model. For more details, please check CRF.

    """
    def __init__(self):
        super(SequenceLabel, self).__init__()
        self.cfg = config.get_cfg()

        LABELS_NUM = self.cfg.DATASET.CONFIG["LABEL_DICT"]["LABELS_NUM"]
        self.dropout = torch.nn.Dropout(0.1)
        self.output_layer = torch.nn.Linear(self.cfg.MODEL.CONFIG.HIDDEN_SIZE, LABELS_NUM)
        self.crf = CRF(num_tags=LABELS_NUM, batch_first=True)

    def forward(self, bert_outputs, token_ids, labels_inputs):
        sequence_output = bert_outputs[1]
        token_ids = token_ids[0]
        # Target & Dropout
        sequence_output = self.dropout(sequence_output)
        logits = self.output_layer(sequence_output).contiguous()
        input_ids = token_ids.permute([1, 0, 2])
        mask = input_ids[2]
        pred_id = self.crf.decode(emissions=logits, mask=mask)

        tgt = labels_inputs
        if tgt is not None:
            # sequence
            tgt = tgt.contiguous()
            loss = self.crf(emissions=logits, tags=tgt, mask=mask)
            loss = -1*loss
            return pred_id, loss
        else:
            return pred_id

