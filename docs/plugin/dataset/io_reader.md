## 插件介绍
该插件实现了文本读取方式

## 插件实现列表
- "tfrecord"：tfrecord 数据文件
- "tsv"：tsv 数据文件，数据"\t"间隔，第一行为head，其他行为数据

## 插件配置
```
DATASET:
  FORMAT: tsv
```

## 自定义插件注册
```
from lichee import plugin
from lichee.utils.tfrecord import tfrecord_loader
from lichee.utils.tfrecord.tools import create_index
from lichee.dataset.io_reader.io_reader_base import BaseIOReader


@plugin.register_plugin(plugin.PluginType.DATA_IO_READER, "user_defined")
class UserDefinedReader(BaseIOReader):
    @classmethod
    def convert_to_tfrecord(cls, data_file, description_file):
        """
        转换 用户数据文件 为 tfrecord 文件

        :param data_file: 用户数据文件
        :param description_file: 描述文件
        :return: tfrecord 文件
        """
        raise NotImplementedError
```
