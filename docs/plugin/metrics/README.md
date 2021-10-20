## 插件介绍
该插件负责模型训练的评估Metics

输入是List of numpy array, 分为labels和logits

输出是Dict，记录对应的指标和值

## 插件实现列表
- "Accuracy"： 正确率评估
- "PRF": Precision-Recall评估
- "ROC_AUC": ROC & AUC评估
- "topk": topk评估

## 插件配置
```
METRICS: "PRF,Accuracy"
```

## 自定义插件注册
```
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

```

