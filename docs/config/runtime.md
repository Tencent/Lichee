### RUNTIME 配置
```
RUNTIME:
  IMPLEMENT: {训练方式}
  SEED: {随机种子}
  AUTOCAST: {混合粒度训练}
  DEBUG: {是否开启debug}
  SAVE_MODEL_DIR: {产出文件保存路径}
  EXPORT: {模型导出格式}
  PREDICT: {数据预测配置}
  REPORT_STEPS: {日志打印周期}
  METRICS: {模型评估方式}
  CONFIG: {运行时其他配置}
```

- **IMPLEMENT**
IMPLEMENT定义了训练方式

```
# 支持 DataParellel 和 DistributedDataParallel
IMPLEMENT: DataParellel
```

- **SEED**
SEED定义了随机种子

```
SEED: 7
```

- **AUTOCAST**
AUTOCAST定义了是否开启混合精度训练

```
AUTOCAST: false
```

- **DEBUG**
DEBUG定义了是否开启debug模式，减少训练迭代次数，加快训练调试

```
DEBUG: false
```

- **SAVE_MODEL_DIR**
SAVE_MODEL_DIR定义了产出文件的保存路径

```
SAVE_MODEL_DIR: local://bert_test
```

- **EXPORT**
EXPORT定义了模型的导出格式；
目前支持torch_nn和onnx，默认为torch_nn；
若模型存在动态输入，需开启DYNAMIC配置。

```
EXPORT: 
  TYPE: onnx
  NAME: model.onnx
  DYNAMIC: true
```

- **PREDICT**
PREDICT定义了数据预测相关配置；
该配置仅当完成模型训练后，需要单独运行模型预测功能时使用；
SHOULD_EXPORT_MODEL定义了是否需要导出上线文件；EXPORT_RESULT_PATH定义了预测结果文件路径。

```
PREDICT:
  SHOULD_EXPORT_MODEL: true
  EXPORT_RESULT_PATH: local://predict_result.txt
```

- **REPORT_STEPS**
REPORT_STEPS定义了训练中的日志打印周期

```
REPORT_STEPS: 20
```

- **METRICS**
METRICS定义了模型评估插件；
若需要多个评估维度，逗号间隔；
默认值：PRF。

```
METRICS: PRF,Accuracy
```

- **CONFIG**
CONFIG用于运行时的扩展配置，业务可以将扩展信息填写到此处。
