## 插件介绍
任务输出插件负责对任务层插件的输出进行后处理。（仅在预测器内使用）

## 插件实现列表
- "simple_cls_out"：bert分类任务输出（结合框架 bert 分类任务插件使用）

## 插件配置
```
TASK:
  CONFIG:
    TASK_OUTPUT: simple_cls_out
  ...
```

## 自定义插件注册
```
from lichee import plugin


@plugin.register_plugin(plugin.PluginType.TASK_OUTPUT, "user_defined")
class UserDefinedTaskOutput:
    @classmethod
    def get_outputs(cls, raw_outputs):
        """
        对任务层插件的输出进行后处理
        :param raw_outputs: 任务层输出 logits
        :return: 后处理结果
        """
        raise NotImplementedError
```
