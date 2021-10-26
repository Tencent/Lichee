#### data目录说明

- bert_dir 存放bert pytorch模型，具体包括预训练模型 bert_google.bin 和模型训练使用词表 vocab.txt 
    - baseline使用bert base预训练模型，参赛者可以根据需求替换或变更。例如使用[模型和bert词表](https://huggingface.co/bert-base-chinese/tree/main)
    - 使用其它版本bert模型需要在../embedding_example.yaml中匹配配置。具体涉及REPRESENTATION 和 DATASET config
     
- tag_list.txt 是用于多标签分类的tag子集，用于构建示例训练目标。参赛者可以根据预训练数据，自由构建预训练模型目标。具体构造可以使用tag_id,category_id等，比赛对此不做限定。
- train.tfrecord 是训练数据集数据
- test.tfrecord 验证集数据

#### 具体目录结构
├── desc.json \
├── test.tfrecord \  
└── train.tfrecord
