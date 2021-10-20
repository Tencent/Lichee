## 插件介绍
表示层插件包括：representation.py

输入：表示层所需的输入TensorList

输出：表示层所需的输出TensorList

## 插件实现列表
- [bert](./bert.md)

## 插件配置
REPRESENTATION配置项是一个list，对应多个表示层
```
REPRESENTATION:
  - NAME: repr_name
    TYPE: repr_type
```

## 自定义插件注册
```
import torch


class BaseRepresentation(torch.nn.Module):
    """
    base representation implementation

    """
    def __init__(self, representation_cfg):
        super(BaseRepresentation, self).__init__()
        self.representation_cfg = representation_cfg

    def forward(self, *args, **kwargs):
        raise NotImplementedError('not implemented!')

    @classmethod
    def load_pretrained_model(cls, cfg, pretrained_model_path):
        pass

    def independent_lr_parameters(self):
        if "LEARNING_RATE" in self.representation_cfg:
            return [{'params': [x for x in self.parameters() if x.requires_grad],
                     'lr': self.representation_cfg['LEARNING_RATE']}]
        return []
```
