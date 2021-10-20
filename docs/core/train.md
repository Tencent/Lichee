## 目的
指定配置文件，选择对应的训练器，进行模型训练。

## 训练命令
```
# DataParallel 训练
python3.6 main.py --trainer=trainer_base --model_config_file=test.yaml

# DistributedDataParallel 单机多卡训练
python3.6 -m torch.distributed.run --nproc_per_node=2 main.py --trainer=trainer_base --model_config_file=test.yaml

# DistributedDataParallel 多机多卡训练
python3.6 -m torch.distributed.run --nproc_per_node=2 --nnodes=2 -node_rank=0 --master_addr=192.168.1.1 --master_port=12345 main.py --trainer=trainer_base --model_config_file=test.yaml
python3.6 -m torch.distributed.run --nproc_per_node=2 --nnodes=2 -node_rank=1 --master_addr=192.168.1.1 --master_port=12345 main.py --trainer=trainer_base --model_config_file=test.yaml
```

## 训练产出
```
bert_test
├── checkpoint       # epoch模型文件
    ├── Epoch_0.bin
    ├── ... 
├── res_info.json    # 模型评估效果
└── task.yaml        # 任务配置文件
```

## 训练配置
依赖完整配置，具体可[参考文档](../config)
