## 路径配置
配置文件内的路径设置需要满足此规则：```{storage插件名}://{storage路径}```

```
# 本地路径（支持相对路径和绝对路径）
local://{local_path}

# http url
http://...
https://...
```

## 示例展示
框架提供了可执行的[demo](../../examples/base_bert_cls_local)和[示例配置](../../examples/base_bert_cls_local/test.yaml)

## 配置介绍
配置文件格式如下：
```
MODEL:
    # 模型配置
    ...

DATASET:
    # 数据集配置
    ...

RUNTIME:
    # 运行时配置
    ...

TRAINING:
    # 训练配置
    ...
```

配置文件由以下四块组成：
- [MODEL](./model.md)
- [DATASET](./dataset.md)
- [RUNTIME](./runtime.md)
- [TRAINING](./training.md)
