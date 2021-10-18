## 插件介绍
该插件负责计算模型loss，在任务层插件内使用

## 插件实现列表
- "cross_entropy"：交叉熵
- "binary_cross_entropy"：二分类交叉熵
- "binary_focal_loss"
- "focal_loss"
- "neg_log_likelihood"
- "mse_loss"

## 插件配置
```
TASK:
  ...
  CONFIG:
    ...
    LOSS:
      NAME: cross_entropy
      VALUE_TYPE: long
```

## 自定义插件注册
```
import torch.nn as nn
from lichee import plugin


@plugin.register_plugin(plugin.PluginType.MODULE_LOSS, 'cross_entropy')
class CrossEntropyLoss:
    @classmethod
    def build(cls, cfg):
        """
        :param cfg: MODEL.TASK.CONFIG.LOSS
        """
        return nn.CrossEntropyLoss()
```
