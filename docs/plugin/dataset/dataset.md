## 插件介绍
该插件实现了 torch dataset 接口

## 插件实现列表
- "dataset_mem"

## 插件配置
```
DATASET:
  NAME: dataset_mem
```

## 自定义插件注册
```
from abc import ABC
from lichee import plugin
from lichee.dataloader.dataset_base import BaseDataset


@plugin.register_plugin(plugin.PluginType.DATA_LOADER, "user_defined")
class UserDefinedDataset(torch.utils.data.Dataset, BaseDataset, ABC):
    def __init__(self, cfg, data_file, desc_file, training=True):
        super().__init__(cfg, data_file, desc_file, training)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def collate(self, batch):
        raise NotImplementedError
```
