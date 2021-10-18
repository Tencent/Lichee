## 插件介绍
该插件实现了单列数据的读取方式。

输入：原始文本输入
输出：训练所需输入

## 插件实现列表
- "bert_text"：[单分类标签](./field_parser/bert_text.md)
- "bert_text_pair"：[多分类标签](./field_parser/bert_text_pair.md)
- "single_cls"：[单分类标签](./field_parser/single_cls.md)
- "multi_cls"：[多分类标签](./field_parser/multi_cls.md)

## 插件配置
FIELD 配置项是一个 list，对应多个 field parser。

每个 field parser 对应训练过程中所需的一个或者多个训练输入。
```NAME```配置了 field parser 插件名。
框架会将此 field parser 的配置设置到插件实例的```cfg```变量内，用户可以在自定义插件内自行处理。

```
FIELD:
- NAME: single_cls
  KEY: label
  ALIAS: label
```

## 自定义插件注册
```
from lichee import plugin
from lichee.dataset.field_parser.field_parser_base import BaseFieldParser


@plugin.register_plugin(plugin.PluginType.FIELD_PARSER, "user_defined")
class UserDefinedParser(BaseFieldParser):
    def __init__(self):
        super().__init__()

        print(self.cfg)  # 该插件对应的配置（该变量由框架设置）

    def parse(self, row, training=False):
        """
        将单行原始数据解析为训练所需数据
        :param record: 单行数据，dict类型
        :param training: 判断是否处于训练阶段，默认值 false
        :return: dict数据，该数据会update到全局dict内。若key冲突，则会覆盖之前parser的结果。
        """
        raise NotImplementedError

    def collate(self, batch):
        """
        组batch逻辑
        :param batch: 需要组batch的数据，list类型，list内每个item为parse的输出
        :return: batch数据
        """
        raise NotImplementedError
```

