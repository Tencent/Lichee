# -*- coding: utf-8 -*-
import torch


class ClassifierBasic(torch.nn.Module):
    def __init__(self, hidden_size, num_class, dropout_rate=0.1):
        super(ClassifierBasic, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc_logits = torch.nn.Linear(hidden_size, num_class)

    def forward(self, encoding):
        # encoding = self.dropout(encoding)
        logits = self.fc_logits(encoding)
        return logits
