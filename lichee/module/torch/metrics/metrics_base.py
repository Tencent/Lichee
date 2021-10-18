# -*- coding: utf-8 -*-
import torch


class BaseMetrics:
    def __init__(self):
        self.total_labels = []
        self.total_logits = []

    def calc(self):
        pass

    def collect(self, labels, logits):
        """collect classification result of each step

        Parameters
        ----------
        labels: List[torch.Tensor]
            classification labels
        logits: List[torch.Tensor]
            classification logits
        """
        assert len(labels) == len(logits)
        for label, logit in zip(labels, logits):
            if isinstance(label, torch.Tensor):
                label = label.data.cpu().numpy()
            if isinstance(logit, torch.Tensor):
                logit = logit.data.cpu().numpy()
            self.total_labels.append(label)
            self.total_logits.append(logit)

    def name(self):
        return ""
