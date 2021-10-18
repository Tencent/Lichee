# -*- coding: utf-8 -*-
import logging
import numpy as np
from sklearn.metrics import accuracy_score

from lichee import plugin
from .metrics_base import BaseMetrics


@plugin.register_plugin(plugin.PluginType.MODULE_METRICS, "Accuracy")
class AccuracyMetrics(BaseMetrics):
    """define accuracy metric that measures single-label or multi-label classification problem

    Attributes
    ----------
    total_labels: List[np.ndarray]
        collected classification labels
    total_logits: List[np.ndarray]
        collected classification logits
    """
    def __init__(self):
        super(AccuracyMetrics, self).__init__()

    def calc(self, threshold=0.5):
        """accumulate classification results and calculate accuracy metric value

        Parameters
        ----------
        threshold: float
            classification threshold

        Returns
        ------
        res_info: Dict[str, Dict[str, float]]
            a dict containing classification accuracy,
            number of correct samples
            and number of total samples
        """
        labels = np.concatenate(self.total_labels, axis=0)
        logits = np.concatenate(self.total_logits, axis=0)

        if len(labels.shape) == 2:
            # multilabel
            pred = (logits >= threshold)
        elif len(labels.shape) == 1:
            pred = np.argmax(logits, axis=1)
        else:
            raise Exception("AccuracyMetrics: not supported labels")

        correct_sum = int(accuracy_score(labels, pred, normalize=False))
        num_sample = int(pred.shape[0])
        logging.info("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct_sum / num_sample, correct_sum, num_sample))

        res_info = {
            "Acc": {
                "value": correct_sum / num_sample,
                "correct": correct_sum,
                "total": num_sample
            }
        }

        self.total_labels, self.total_logits = [], []

        return res_info

    def name(self):
        return "Accuracy"
