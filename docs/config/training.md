## TRAINING 配置
```
TRAINING:
  EPOCHS: {训练迭代次数}
  OPTIMIZER: {optimizer配置}
  SCHEDULER: {scheduler配置}
```

- **EPOCHS**
EPOCHS定义了模型训练epoch次数。

```
EPOCHS: 5
```

- **OPTIMIZER**
OPTIMIZER定义了模型训练的optimizer。

```
OPTIMIZER:
  NAME: BertAdamW
  LEARNING_RATE: 2e-5
  OPTIM_EPS: 1e-6
  OPTIM_WEIGHT_DECAY: 0.0
  CORRECT_BLAS: false
```

- **SCHEDULER**
SCHEDULER定义了模型训练的scheduler。

```
SCHEDULER:
  NAME: warmup_linear
```
