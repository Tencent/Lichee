# -*- coding: utf-8 -*-
import torch

from lichee import config
from lichee import plugin
from lichee.module.torch.layer.classifier import ClassifierBasic
from lichee.task.torch.task_base import BaseTask


class BaseSimpleClassification(BaseTask):
    """Base class of simple classification.

    Attributes
    ----------
    cfg: CfgNode
        config content
    loss_fn: loss cls
        loss function
    loss_value_type: str
        data type of loss value
    label_key: str
        key of label
    classifier: torch.nn.Module
        classifier for different task

    """

    def __init__(self):
        super(BaseSimpleClassification, self).__init__()
        self.cfg = config.get_cfg()
        self.loss_fn = None
        self.init_loss()
        self.loss_value_type = self.cfg.MODEL.TASK.CONFIG.LOSS.VALUE_TYPE
        self.label_key = None
        self.classifier = None

    def init_loss(self):
        """ fill the loss_fn according with loss-related config

        """
        loss = plugin.get_plugin(plugin.PluginType.MODULE_LOSS, self.cfg.MODEL.TASK.CONFIG.LOSS.NAME)
        self.loss_fn = loss.build(self.cfg.MODEL.TASK.CONFIG.LOSS)

    def init_label(self, task_name):
        """fill the label key from graph config item

        Parameters
        ---------
        task_name: str
            name of task, used to find the label key from graph config
        """
        for graph_cfg in self.cfg.MODEL.GRAPH:
            name = graph_cfg["NAME"]
            if name != task_name:
                continue
            self.label_key = graph_cfg["LABELS"]
    
    def forward(self, *args, label_inputs):
        raise NotImplementedError("Not Implemented!")

    def forward_helper(self, labels_inputs, logits):
        """some common steps of forward function in classifications
        Parameters
        ---------
        labels_inputs: Dict
            inputs of labels
        logits: Any
            logits computed by classifier

        """
        if labels_inputs is not None:
            label = labels_inputs[self.label_key]
            if self.loss_value_type == "float":
                label = label.float()

            loss = self.loss_fn(logits, label)
            return [logits], loss
        else:
            return [logits]


@plugin.register_plugin(plugin.PluginType.TASK, "simple_cls")
class SimpleClassification(BaseSimpleClassification):
    """classification with Basic classifier

    """
    def __init__(self, target_cfg=None):
        super(SimpleClassification, self).__init__()
        self.bert_pooler = BERTPooler(self.cfg.MODEL.CONFIG.HIDDEN_SIZE)
        self.init_label("simple_cls")
        self.classifier = ClassifierBasic(self.cfg.MODEL.CONFIG.HIDDEN_SIZE, self.cfg.DATASET.CONFIG.NUM_CLASS)

    def forward(self, representation_outputs, labels_inputs):
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        all_encoder_layers, sequence_output = representation_outputs
        encoding = self.bert_pooler(sequence_output)
        logits = self.classifier(encoding)
        return self.forward_helper(labels_inputs, logits)


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


@plugin.register_plugin(plugin.PluginType.TASK, "simple_vgg_cls")
class SimpleVGGClassification(BaseTask):
    """classification with VGG

    """
    def __init__(self, target_cfg=None):
        super(SimpleVGGClassification, self).__init__()
        self.init_label("simple_vgg_cls")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, self.cfg.DATASET.CONFIG.NUM_CLASS),
        )

    def forward(self, representation_outputs, labels_inputs):
        x = torch.flatten(representation_outputs, 1)
        logits = self.classifier(x)
        return self.forward_helper(labels_inputs, logits)


@plugin.register_plugin(plugin.PluginType.TASK, "simple_resnet_cls")
class SimpleResNetClassification(BaseTask):
    """classification with ResNet classifier

    """
    def __init__(self, target_cfg=None):
        super(SimpleResNetClassification, self).__init__()
        self.init_label("simple_resnet_cls")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * self.cfg.MODEL.TASK.CONFIG.EXPANSION, self.cfg.DATASET.CONFIG.NUM_CLASS)
        )

    def forward(self, representation_outputs, labels_inputs):
        x = torch.flatten(representation_outputs, 1)
        logits = self.classifier(x)
        return self.forward_helper(labels_inputs, logits)


@plugin.register_plugin(plugin.PluginType.TASK, "simple_video_cls")
class SimpleVideoClassification(BaseTask):
    """classification with video classifier

    """
    def __init__(self, target_cfg=None):
        super(SimpleVideoClassification, self).__init__()
        self.init_label("simple_video_cls")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.cfg.MODEL.TASK.INPUT_SIZE, self.cfg.DATASET.CONFIG.NUM_CLASS)
        )

    def forward(self, representation_outputs, labels_inputs):
        x = torch.flatten(representation_outputs, 1)
        logits = self.classifier(x)

        return self.forward_helper(labels_inputs, logits)
