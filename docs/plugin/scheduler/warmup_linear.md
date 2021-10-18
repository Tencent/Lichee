## 插件介绍
warmup_linear迭代器，将学习率先递增到100%，再减小
## 插件配置
```
SCHEDULER:
    NAME: warmup_linear
    LEARNING_RATE_SCHEDULE_WARMUP_STEP_RATIO: 0.0
```

LEARNING_RATE_SCHEDULE_WARMUP_STEP_RATIO: warmup step ratio，默认值为0.1
