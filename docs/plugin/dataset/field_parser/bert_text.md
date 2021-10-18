## 插件介绍
对单个文本数据进行 bert tokenizer 处理

## 插件配置
```
FIELD:
- NAME: bert_text
  KEY: text_a
  ALIAS: text_a
  VOCAB_PATH: https://pcg-kandian-alg-race-1251316161.cos.ap-guangzhou.myqcloud.com/lichee_opensource_model/bert_vocab.txt
  MAX_SEQ_LEN: 128
```

KEY：输入head
ALIAS：输出键值
VOCAB_PATH：词表路径
MAX_SEQ_LEN：最大文本长度
