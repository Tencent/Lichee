## 插件介绍
Adam优化器

## 插件配置
```
OPTIMIZER:
  NAME: Adam
  LEARNING_RATE: 2e-5
  OPTIM_AMSGRAD: false
  OPTIM_WEIGHT_DECAY: 1e-4
  OPTIM_BETA1: 0.9
  OPTIM_BETA2: 0.999
  OPTIM_EPS: 1e-8
```

LEARNING_RATE: float >= 0. 学习率.
OPTIM_AMSGRAD: boolean. 是否应用此算法的 AMSGrad 变种，来自论文 "On the Convergence of Adam and Beyond"。
OPTIM_WEIGHT_DECAY: float >= 0. 每次参数更新后学习率衰减值.
OPTIM_BETA1: float, 0 < beta < 1. 通常接近于 1。
OPTIM_BETA2: float, 0 < beta < 1. 通常接近于 1。
OPTIM_EPS: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
