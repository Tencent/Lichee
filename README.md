[TOC]
## 框架简介
方便使用，支持多模态，多任务的统一训练框架

能力列表：
- bert + 分类任务
- 自定义任务训练（插件注册）

## 框架设计

框架采用分层的思想组织模型训练流程。
- DATA 层负责读取用户数据，根据 field 管理数据。
- Parser 层负责转换原始数据为模型的输入。
- MODEL 层为模型层，具体由表示层（REPRESENTATION）和任务层（TASK）组成。
    - 表示层用于抽取数据的高维特征，框架里内置了一些成熟实现（包括bert、NeXtVLAD等）。
    - 任务层用于拟合具体的训练任务，框架里提供一些默认实现（包括分类任务等），用户也可以根据训练任务，自定义任务模型。
    - 任务层可用于实现多任务训练。
- 框架通过配置文件组合 DATA、Parser、MODEL、Optimizer、Scheduler，构建具体的训练流程。
- 框架还内置了成熟的组件模块（Module），包括 Metrics、Loss、Layer 等，供用户选择使用。

<img src="./resources/images/lichee-design.png">

详细可[参考文档](./docs/desc.md)

## 框架安装
参考[文档](./docs/install.md)

## 使用说明
```
cd examples/base_bert_cls_mac
sh train.sh
sh eval.sh
sh predict.sh
```

模型训练任务被拆分为三步，每个步骤可以独立运行：
- [训练](./docs/core/train.md)
- [评估](./docs/core/eval.md)
- [预测](./docs/core/predict.md)

任务执行均依赖配置文件，详细介绍可[参考文档](./docs/config)

若框架默认实现无法满足需求，也可以实现自定义插件，详细介绍可[参考文档](./docs/plugin)
