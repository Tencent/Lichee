## 插件介绍
对两个文本数据进行 bert tokenizer 处理

## 插件配置
```
FIELD:
- NAME: bert_text_pair
  KEY: text_a,text_b
  VOCAB_PATH: https://pcg-kandian-ai-1251316161.cos.ap-guangzhou.myqcloud.com/lichee-dev/bert_vocab.txt
  MAX_SEQ_LEN: 128
  TYPE_VOCAB_SIZE: 3
```

KEY：输入head，逗号间隔
ALIAS：输出键值
VOCAB_PATH：词表路径
MAX_SEQ_LEN：最大文本长度
TYPE_VOCAB_SIZE: token_type_ids的词典大小，设置为2/3
