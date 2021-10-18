## MODEL 配置
```
MODEL:
  NAME: {模型插件名}
  REPRESENTATION: {表示层配置}
  TASK: {任务层配置}
  GRAPH: {训练图配置}
  CONFIG: {模型其他配置}
```

- **NAME**
NAME定义了模型插件名，默认的模型插件实现"model_standard"通过 REPRESENTATION、TASK、GRAPH 配置来构建模型

```
NAME: model_standard
```

- **REPRESENTATION**
REPRESENTATION定义了表示层的所有组件配置；
表示层用于将用户数据抽象为高维向量；
每个表示层组件需要定义 NAME（组件名，作为唯一表示）、TYPE（插件名）、FINE_TUNING（fine tuning开关）、PRETATINED（预训练模型开关）、MODEL_PATH（预训练模型路径）。

```
REPRESENTATION:
  - NAME: bert_repr
    TYPE: bert
    FINE_TUNING: true
    PRETRAINED: true
    MODEL_PATH: local://bert_google/bert_google.bin
    CONFIG:
```

- **TASK**
TASK定义了具体的训练任务，比如单分类任务、目标检测任务等；
TASK需要定义 NAME（任务层插件名）和 CONFIG内的LOSS（可选）；
TASK仅存在一个。

```
Task:
  NAME: simple_cls
  CONFIG: 
    LOSS:
      NAME: cross_entropy
      VALUE_TYPE: long
```

- **GRAPH**
GRAPH定义了图结构内的数据流向，由表示层和任务层组成；
用户可以定义多个表示层，但只能存在一个任务层；
表示层需要定义 INPUTS（输入）和 OUTPUTS（输出）；
任务层需要定义 LABEL（目标列）和 INPUTS（输入）；
INPUTS仅来源于field_parser输出和组件输出。

```
GRAPH:
  # 表示层数据流配置
  - NAME: bert_repr
    INPUTS:
      - text_a
    OUTPUTS:
      - bert_output

  # 任务层数据流配置
  - NAME: simple_cls
    LABEL: label
    INPUTS:
      - bert_output
```

- **CONFIG**
CONFIG用于模型的扩展配置，业务可以将扩展信息填写到此处。

```
CONFIG:
  USER_DEFINED: xxx
```