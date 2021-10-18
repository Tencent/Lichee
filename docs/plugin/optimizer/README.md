## 插件介绍
该插件负责模型训练的参数更新优化器

## 插件实现列表
- "SGD"：[SGD优化器](./sgd.md)
- "Adam"：[Adam优化器](./adam.md)
- "AdamW"：[AdamW优化器](./adamw.md)
- "BertAdamW"：[BertAdamW优化器](./bertadamw.md)

## 插件配置
```
OPTIMIZER:
  NAME: BertAdamW
  ...
```

## 自定义插件注册
```
import torch.nn as nn
from lichee import plugin


@plugin.register_plugin(plugin.PluginType.MODULE_OPTIMIZER, "SGD")
class SGD(ConfigOptim):

    @classmethod
    def build(cls, model, cfg):
        """
        :param cfg: global cfg
        """
        return torch.optim.SGD(model.parameters(),
                               lr=cfg.TRAINING.OPTIMIZER.LEARNING_RATE,
                               momentum=cfg.TRAINING.OPTIMIZER.OPTIM_MOMENTUM)
```
