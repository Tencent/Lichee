## 插件介绍
任务层插件定义了具体的训练任务。

## 插件实现列表
- "simple_cls"：[bert分类任务](./simple_cls.md)

## 插件配置
```
TASK:
  NAME: simple_cls
  ...
```

## 自定义插件注册
```
import torch.nn as nn
from lichee import plugin
from lichee.task.torch.task_base import BaseTask


@plugin.register_plugin(plugin.PluginType.TASK, "user_defined")
class UserDefinedClassification(BaseTask):
    def __init__(self):
        super(BaseTask, self).__init__()

    def forward(self, *args, label_inputs):
        """
        任务层模型实现
        :param args: 依赖的表示层的输出，按配置传入
        :param label_inputs: label数据，若模型调用时未传label数据，该数据为None
        :return: 训练时需返回 logits 和 loss；预测时仅返回 logits。
        """
        raise NotImplementedError
```

## 插件依赖
#### loss插件
任务插件若收到label_inputs数据，则表示处于模型训练阶段，需要返回loss结果。
框架里提供了一些默认的loss插件实现，用户可以自由选择是否使用，具体可以[参考文档](../loss)

```
from lichee import config
from lichee import plugin


cfg = config.get_cfg()
loss = plugin.get_plugin(plugin.PluginType.MODULE_LOSS, cfg.MODEL.TASK.CONFIG.LOSS.NAME)
loss_fn = loss.build(cfg.MODEL.TASK.CONFIG.LOSS)
```

#### 任务输出插件（仅在预测器内使用）
框架允许对任务层的输出添加后处理插件，作为最终模型的输出，插件实现可[参考文档](./task_output)
