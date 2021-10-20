## 目的
指定配置文件，选择对应的预测器，预测结果。

## 预测命令
```
python3.6 main.py --predictor=predictor_base --model_config_file=test.yaml
```

## 预测产出
```
bert_test
├── ... 
└── predict_res.txt  # 预测结果
```

## 预测配置
依赖以下配置，具体可[参考文档](../config)
```
MODEL: 
  TASK: {任务层配置}  # 定义了模型输出
  GRAPH: {训练图配置}  # 定义了模型输入

# 获取预测数据集
DATASET:
  NAME: {dataset插件名}
  FORMAT: {数据集插件名}
  FIELD: {数据列插件配置列表}
  DESC_PATH: {数据集描述文件路径}
  EVAL_DATA: {评估数据集配置}
    DATA_PATH: {数据集路径列表}
    BATCH_SIZE: {batch size}
  CONFIG: {数据集其他配置}

# 定义了模型格式和存储路径
RUNTIME:
  SAVE_MODEL_DIR: {产出文件保存路径}
  EXPORT: {模型导出格式}
```
