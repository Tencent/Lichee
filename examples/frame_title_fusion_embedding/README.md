#### 视频多模态相似性训练
 - 这是视频多模态相似性训练代码
 - 该用例中使用了自定义注册插件，注册的插件包括
    - FIELD_PARSER:
        - [frame_feature](https://git.woa.com/fcc-ai-lab/lichee-dev/blob/master/examples/frame_title_fusion_embedding/module/feature_parser.py#L11)
        - [tag_cls](https://git.woa.com/fcc-ai-lab/lichee-dev/blob/master/examples/frame_title_fusion_embedding/module/feature_parser.py#L61)
        - [id](https://git.woa.com/fcc-ai-lab/lichee-dev/blob/master/examples/frame_title_fusion_embedding/module/feature_parser.py#L112)
    - TASK: [concat_cls](https://git.woa.com/fcc-ai-lab/lichee-dev/blob/master/examples/frame_title_fusion_embedding/module/models.py#L22)
    - TRAINER: [embedding_trainer](https://git.woa.com/fcc-ai-lab/lichee-dev/blob/master/examples/frame_title_fusion_embedding/module/models.py#L61)
    - MODULE_OPTIMIZER: [LayeredOptim](https://git.woa.com/fcc-ai-lab/lichee-dev/blob/master/examples/frame_title_fusion_embedding/module/utils.py#L22)
    - MODULE_LOSS: [BCELoss](https://git.woa.com/fcc-ai-lab/lichee-dev/blob/master/examples/frame_title_fusion_embedding/module/utils.py#L51)
    - MODULE_METRICS: [PRScore](https://git.woa.com/fcc-ai-lab/lichee-dev/blob/master/examples/frame_title_fusion_embedding/module/utils.py#L59)

#### 代码目录结构
  - 配置yaml: embedding_example.yaml
  - 训练入口： main.py
  - 训练示例脚本: train.sh
  - 测试示例脚本： eval.sh
  - 示例代码依赖： ./module
  - 数据文件夹：./data
  
#### 训练&测试命令
  - 如示例训练脚本所示：
```bash
python3 main.py --trainer=embedding_trainer --model_config_file=embedding_example.yaml
```
  - 测试需要指定checkpoint和指定配置文件，示例如下：
```bash
#python3 main.py --trainer=embedding_trainer --model_config_file=your_config.yaml --mode test --checkpoint your_check_point.bin --dataset SPEARMAN_DATA
python3 main.py --trainer=embedding_trainer --model_config_file=embedding_example.yaml --mode test --checkpoint Epoch_1_0.0000_0.0000.bin --dataset SPEARMAN_DATA
```
