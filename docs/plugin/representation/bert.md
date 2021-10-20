## 插件介绍
BERT表示层，输出对应的token_ids, seg_ids, mask_ids, 输出每个encoder层的embedding和最终层的embedding， 支持标准Bert和混合粒度Bert两种模式，其中，混合粒度Bert需要分词方法。

## 插件配置
```
REPRESENTATION:
  - NAME: bert
    TYPE: bert
    CONFIG:
      MIX_GRAINED: False,
      MIX_MODE: "basic",
      ATTENTION_PROBS_DROPOUT_PROB: 0.1,
      HIDDEN_ACT: 'gelu',
      HIDDEN_DROPOUT_PROB: 0.1,
      HIDDEN_SIZE: 768,
      NUM_HIDDEN_LAYERS: 12,
      INITIALIZER_RANGE: 0.02,
      INTERMEDIATE_SIZE: 3072,
      MAX_POSITION_EMBEDDINGS: 512,
      NUM_ATTENTION_HEADS: 12
```

MIX_GRAINED：混合粒度模式

MIX_MODE: 混合模式， cat OR basic

HIDDEN_SIZE：embedding长度

NUM_HIDDEN_LAYERS：encoder数量

MAX_POSITION_EMBEDDINGS：最长embedding数量

NUM_ATTENTION_HEADS：HEADs 数量