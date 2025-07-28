#!/bin/bash
# FSDP多机训练启动脚本

# FSDP特定环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_ASYNC_ERROR_HANDLING=1

# NODE_RANK=0 MASTER_ADDR="100.64.34.144" torchrun \
#     --nproc_per_node=8 \
#     --nnodes=1 \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --master_port=29888 \
#     -m llamafactory.cli train \
#     --config_file fsdp_config.yaml

FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_ADDR="100.64.34.144" MASTER_PORT=29888 python -m llamafactory.cli train qwen3-14b-fsdp.yaml
