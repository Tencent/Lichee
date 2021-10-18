# -*- coding: utf-8 -*-
import numpy as np

from lichee import plugin
from lichee.utils import logging
from .metrics_base import BaseMetrics


@plugin.register_plugin(plugin.PluginType.MODULE_METRICS, "PRF")
class PRFMetrics(BaseMetrics):
    """define PRF metric that measures single-label or multi-label classification problem

    Attributes
    ----------
    total_labels: List[np.ndarray]
        collected classification labels
    total_logits: List[np.ndarray]
        collected classification logits
    """

    def __init__(self):
        super(PRFMetrics, self).__init__()

    def calc(self, threshold=0.5):
        """accumulate classification results and calculate PRF metric value

        Parameters
        ----------
        threshold: float
            classification threshold

        Returns
        ------
        res_info: Dict[str, Dict[str, float]]
            a dict containing precision, recall, f1 score and support values
            calculated in micro and macro settings
        """

        labels = np.concatenate(self.total_labels, axis=0)
        logits = np.concatenate(self.total_logits, axis=0)
        num_class = logits.shape[1]

        if len(labels.shape) == 2:
            # multilabel
            preds = (logits >= threshold)
            res_info = multi_label_metric(labels, preds)
            self.total_labels, self.total_logits = [], []
            return res_info
        elif len(labels.shape) == 1:
            preds = np.argmax(logits, axis=1)
            res_info = confusion_metric(labels, preds, num_class)
            self.total_labels, self.total_logits = [], []
            return res_info
        else:
            raise Exception("PRFMetrics: not supported labels")

    def name(self):
        return "PRF"


def confusion_metric(labels, preds, num_class=None, sum_metric_drop_class=[]):
    if num_class is None:
        num_class = len(set(labels))

    confusion = np.zeros([num_class, num_class], dtype=np.int64)
    for i in range(len(labels)):
        confusion[preds[i], labels[i]] += 1

    res_info = {"PR": {}}

    logging.info("Confusion matrix:\n%s", confusion)
    logging.info("Report precision, recall, f1,  support:")

    tp_list = []
    p_list = []
    r_list = []
    p_ratio_list = []
    r_ratio_list = []
    f1_list = []
    sum_metric_drop_class = set(sum_metric_drop_class)
    for i in range(len(confusion)):
        tp = confusion[i, i].item()
        p = confusion[i, :].sum().item()
        r = confusion[:, i].sum().item()

        p_ratio = tp / (p + 1e-5)
        r_ratio = tp / (r + 1e-5)
        f1 = 2 * p_ratio * r_ratio / (p_ratio + r_ratio + 1e-5)

        if i not in sum_metric_drop_class:
            tp_list.append(confusion[i, i].item())
            p_list.append(p)
            r_list.append(r)
            p_ratio_list.append(p_ratio)
            r_ratio_list.append(r_ratio)
            f1_list.append(f1)

        res_info['PR']['Label_' + str(i)] = {'precise': f4(p_ratio), 'recall': f4(r_ratio), 'f1': f4(f1), 'support': r}
        logging.info("Label {}: {:.4f}, {:.4f}, {:.4f}, {:}".format(i, p_ratio, r_ratio, f1, r))

    support = int(sum(r_list))
    micro_p_ratio = sum(tp_list) / (sum(p_list) + 1e-5)
    micro_r_ratio = sum(tp_list) / (sum(r_list) + 1e-5)
    micro_f1 = 2 * micro_p_ratio * micro_r_ratio / (micro_p_ratio + micro_r_ratio + 1e-5)

    macro_p_ratio = sum(p_ratio_list) / len(p_ratio_list)
    macro_r_ratio = sum(r_ratio_list) / len(r_ratio_list)
    macro_f1 = sum(f1_list) / len(f1_list)

    logging.info("Macro precision, recall, f1,  support")
    logging.info('      %s, %s, %s, %s', f4(macro_p_ratio), f4(macro_r_ratio), f4(macro_f1), support)
    logging.info("Micro precision, recall, f1,  support")
    logging.info('      %s, %s, %s, %s', f4(micro_p_ratio), f4(micro_r_ratio), f4(micro_f1), support)

    res_info['Macro_PR'] = {'precise': f4(macro_p_ratio), 'recall': f4(macro_r_ratio), 'f1': f4(macro_f1),
                            'support': support}
    res_info['Micro_PR'] = {'precise': f4(micro_p_ratio), 'recall': f4(micro_r_ratio), 'f1': f4(micro_f1),
                            'support': support}
    res_info['CONFUSION_MATRIC'] = confusion.tolist()
    return res_info


def multi_label_metric(labels, preds):
    from sklearn.metrics import f1_score, average_precision_score
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)
    marco_f1 = f1_score(labels, preds, average="macro")
    m_ap = average_precision_score(labels, preds, average="macro")
    res_info = {
        'Marco F1': {'value': f4(marco_f1)},
        'mAP': {'value': f4(m_ap)}
    }
    logging.info("Marco F1: {:.4f}, mAP:{:.4f}".format(marco_f1, m_ap))
    return res_info


def f4(f):
    return float('{:.4f}'.format(f))
