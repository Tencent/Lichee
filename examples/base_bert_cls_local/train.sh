python3.6 main.py --trainer=trainer_base --model_config_file=test.yaml
# python3.6 -m torch.distributed.run --nproc_per_node=2 main.py --trainer=trainer_base --model_config_file=test.yaml
# python3.6 -m torch.distributed.run --nproc_per_node=2 --nnodes=2 -node_rank=0 --master_addr=192.168.1.1 --master_port=12345 main.py --trainer=trainer_base --model_config_file=test.yaml
# python3.6 -m torch.distributed.run --nproc_per_node=2 --nnodes=2 -node_rank=1 --master_addr=192.168.1.1 --master_port=12345 main.py --trainer=trainer_base --model_config_file=test.yaml
