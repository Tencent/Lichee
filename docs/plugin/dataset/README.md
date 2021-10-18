## 插件介绍
```
                    **dataset_base**
                             |
                             |
                      **dataset_mem**
                             |
                             |
            |-----------------------------|
            |                             |
 **io_reader_base**        **field_parser_base**
            |
            |
     **tfrecord**     
```

数据集插件包括三种类型：
- dataset：torch dataset 接口实现，默认实现"dataset_mem"依赖了以下两种插件
- io_reader：文件格式插件，决定了文件的读取方式（目前支持tfrecord、csv格式）
- field_parser：文件内单列数据的解析插件（目前支持bert_text、single_cls等）

## 插件类型
- [dataset](./dataset.md)
- [io_reader](./io_reader.md)
- [field parser](./field_parser.md)
