### DATASET 配置
```
DATASET:
  NAME: {dataset插件名}
  FORMAT: {数据集插件名}
  FIELD: {数据列插件配置列表}
  DESC_PATH: {数据集描述文件路径}
  TRAIN_DATA: {训练数据集配置}
    DATA_PATH: {数据集路径列表}
    BATCH_SIZE: {batch size}
  EVAL_DATA: {评估数据集配置}
    DATA_PATH: {数据集路径列表}
    BATCH_SIZE: {batch size}
  CONFIG: {数据集其他配置}
```

- **NAME**
NAME定义了DATASET插件名；
目前框架仅提供了”dataset_mem“。

```
NAME: dataset_mem
```

- **FORMAT**
FORMAT定义了数据集格式插件名；
目前框架提供了 ”tfrecord“ 和 ”tsv“（第一行为列表名，后面为数据集，\t间隔）。
业务如有自定义数据集，可以编写io_reader插件，可[参考文档](https://iwiki.woa.com/pages/viewpage.action?pageId=605707260)。

```
FORMAT: tsv
```

- **FIELD**
FIELD定义了训练依赖的不同输入，每个输入对应一个field_parser插件。
field_parser输入定义在 KEY 内，若单个field_parser需要多个输入，可以逗号间隔。
field_parser输出默认标识为 KEY，用户也可以用 ALIAS 重命名。
</br><font color="#dd0000">field_parser若依赖多个输入，应该用ALIAS对输出重命名。【确保key里不包含逗号】</font>
<font color="#dd0000">field_parser的输出被用于模型的数据流。</font></br>

```
FIELD:
  - NAME: single_cls
    KEY: label
  - NAME: bert_text
    KEY: text_a
  - NAME: bert_text_pair
    KEY: text_a,text_b
    ALIAS: text_merge
```

- **DESC_PATH**
DESC_PATH定义了数据集的描述文件路径。

```
DESC_PATH: local://desc.json
```

数据集表述文件需要定义所有列数据的解析格式，且遵守json格式：
```
{
  "text": "byte",  # 解析为bytes
  "images": "bytes"  # 解析为bytes_list
}
```

- **TRAIN_PATH**
TRAIN_PATH定义了训练数据集的路径集合和采样batch_size。

```
TRAIN_DATA:
  DATA_PATH:
    - local://train.tsv
  BATCH_SIZE: 6
```

- **EVAL_PATH**
EVAL_PATH定义了评估数据集的路径集合和采样batch_size。

```
EVAL_DATA:
  DATA_PATH:
    - local://train.tsv
  BATCH_SIZE: 6
```

- **CONFIG**
CONFIG用于数据集的扩展配置，业务可以将扩展信息填写到此处。

```
CONFIG:
  USER_DEFINED: xxx
```
