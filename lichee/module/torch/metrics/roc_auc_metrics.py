# -*- coding: utf-8 -*-
import logging

import numpy as np
from sklearn.metrics import roc_auc_score

from lichee import plugin
from .metrics_base import BaseMetrics


@plugin.register_plugin(plugin.PluginType.MODULE_METRICS, "ROC_AUC")
class ROCAUCMetrics(BaseMetrics):
    """define ROC AUC metric that measures single-label or multi-label classification problem

    Attributes
    ----------
    total_labels: List[np.ndarray]
        collected classification labels
    total_logits: List[np.ndarray]
        collected classification logits
    """

    def __init__(self):
        super(ROCAUCMetrics, self).__init__()

    def calc(self):
        """accumulate classification results and calculate accuracy metric value

        Returns
        ------
        score: float
            ROC AUC score
        """
        labels = np.concatenate(self.total_labels, axis=0)
        logits = np.concatenate(self.total_logits, axis=0)

        num_class = logits.shape[1]

        if len(labels.shape) == 2:
            # multilabel
            score = roc_auc_score(labels, logits)
        elif logits.shape[1] == 2:
            # binary
            score = roc_auc_score(labels, logits[:, 1])
        else:
            # multiclass
            score = roc_auc_score(labels, logits, multi_class='ovo', labels=np.arange(num_class))

        logging.info("ROC AUC score: %.4f", score)

        self.total_labels, self.total_logits = [], []

        return score

    def name(self):
        return "ROC_AUC"
