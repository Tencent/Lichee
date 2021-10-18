# -*- coding: utf-8 -*-
import torch

from lichee import config
from lichee import plugin
from lichee.module.torch.layer.classifier import ClassifierBasic
import torch.nn.functional as F


@plugin.register_plugin(plugin.PluginType.TASK, "distill_cls")
class DistillClassification(torch.nn.Module):
    """Distill Classification is used for distill classification.

    Attributes
    ----------
    cfg: CfgNode
        Global Configuration.
    bert_pooler: torch.nn.Module
        pooler after encoders.

    hard_label_key: str
        label_key which used for hard loss computing

    soft_label_key: str
        label_key which used for soft loss computing

    hard_loss_fn: str
        hard loss function name

    ssoft_loss_fn: str
        soft loss function name
    """
    def __init__(self, target_cfg=None):
        super(DistillClassification, self).__init__()
        self.cfg = config.get_cfg()

        self.bert_pooler = BERTPooler(self.cfg.MODEL.CONFIG.HIDDEN_SIZE)
        self.classifier = ClassifierBasic(self.cfg.MODEL.CONFIG.HIDDEN_SIZE, self.cfg.DATASET.CONFIG.NUM_CLASS)

        self.hard_label_key = None
        self.hard_loss_fn = None
        self.soft_label_key = None
        self.soft_loss_fn = None

        self.soft_alpha = self.cfg.MODEL.TASK.CONFIG.SOFT_LOSS.SOFT_ALPHA
        self.init_loss()

        self.hard_loss_value_type = self.cfg.MODEL.TASK.CONFIG.LOSS.VALUE_TYPE

    def init_loss(self):
        soft_loss = plugin.get_plugin(plugin.PluginType.MODULE_LOSS, self.cfg.MODEL.TASK.CONFIG.SOFT_LOSS.NAME)
        self.soft_loss_fn = soft_loss.build(self.cfg.MODEL.TASK.CONFIG.SOFT_LOSS)

        hard_loss = plugin.get_plugin(plugin.PluginType.MODULE_LOSS, self.cfg.MODEL.TASK.CONFIG.LOSS.NAME)
        self.hard_loss_fn = hard_loss.build(self.cfg.MODEL.TASK.CONFIG.LOSS)

        self.hard_label_key = self.cfg.MODEL.TASK.CONFIG.LOSS.LABEL_NAME

    def forward(self, *target_inputs):
        representation_outputs, soft_tgt, label_dict = target_inputs

        all_encoder_layers, sequence_output = representation_outputs
        encoding = self.bert_pooler(sequence_output)
        logits = self.classifier(encoding)

        # Predict Mode
        if label_dict is None:
            return [logits]

        # Eval Mode
        hard_tgt = label_dict[self.hard_label_key]
        if self.hard_loss_value_type == "float":
            hard_tgt = hard_tgt.float()
        hard_loss = self.hard_loss_fn(logits, hard_tgt)

        if soft_tgt is None:
            return [logits], hard_loss

        # Train Mode
        if self.hard_loss_value_type == "float":
            hard_tgt = hard_tgt.float()
        hard_loss = self.hard_loss_fn(logits, hard_tgt)

        soft_loss = self.soft_loss_fn(logits, soft_tgt)
        loss = self.soft_alpha * soft_loss + (1 - self.soft_alpha) * hard_loss

        return [logits], loss

    @classmethod
    def get_output(cls, logits):
        logits = torch.tensor(logits[0])
        probs = F.softmax(logits, dim=-1)
        _, preds = torch.max(probs, dim=-1)
        preds = preds.cpu().numpy()
        logits = logits.cpu().numpy()
        return logits, preds


class BERTPooler(torch.nn.Module):
    def __init__(self, hidden_size):
        super(BERTPooler, self).__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
