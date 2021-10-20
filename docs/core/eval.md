## 目的
指定配置文件，选择对应的评估器，评估所有的epoch模型。

## 评估命令
```
python3.6 main.py --evaluator=evaluator_base --model_config_file=test.yaml
```

## 评估产出
```
bert_test
├── ... 
└── model.pth        # 效果最优的模型文件
```

## 评估配置
依赖以下配置，具体可[参考文档](../config)
```
MODEL:
  GRAPH: {训练图配置}  # 定义了模型输入

# 获取评估数据集
DATASET:
  NAME: {dataset插件名}
  FORMAT: {数据集插件名}
  FIELD: {数据列插件配置列表}
  DESC_PATH: {数据集描述文件路径}
  EVAL_DATA: {评估数据集配置}
    DATA_PATH: {数据集路径列表}
    BATCH_SIZE: {batch size}
  CONFIG: {数据集其他配置}

# 定义了 评估方式 和 模型导出方式
RUNTIME:
  SAVE_MODEL_DIR: {产出文件保存路径}
  METRICS: {模型评估方式}
  EXPORT: {模型导出格式}

# 待评估的epoch模型数量
TRAINING:
  EPOCHS: {训练迭代次数}
```
