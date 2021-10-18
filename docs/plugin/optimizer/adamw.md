## 插件介绍
AdamW优化器

## 插件配置
```
OPTIMIZER:
  NAME: AdamW
  LEARNING_RATE: 2e-5
  OPTIM_WEIGHT_DECAY: 1e-4
  OPTIM_EPS: 1e-8
```

LEARNING_RATE: float >= 0. 学习率.
OPTIM_WEIGHT_DECAY: float >= 0. 每次参数更新后学习率衰减值.
OPTIM_EPS: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
