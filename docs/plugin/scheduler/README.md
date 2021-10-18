## 插件介绍
该插件负责模型训练的学习率更新迭代器

## 插件实现列表
- "constant"：[constant迭代器](./constant.md)
- "warmup_constant"：[warmup_constant优化器](./warmup_constant.md)
- "warmup_linear"：[warmup_linear迭代器](./warmup_linear.md)

## 插件配置
```
SCHEDULER:
  NAME: warmup_linear
```

## 自定义插件注册
```
import torch.nn as nn
from lichee import plugin


@plugin.register_plugin(plugin.PluginType.MODULE_SCHEDULER, 'constant')
class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """

    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)

    @classmethod
    def build(cls, optimizer, cfg):
        """
        :param optimizer: 优化器实例
        :param cfg: global cfg
        """
        return cls(optimizer)
```
