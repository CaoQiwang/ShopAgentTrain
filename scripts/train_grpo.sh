#!/usr/bin/env bash

# bash scripts/train_grpo.sh

set -euo pipefail

# 大型、持久化缓存放 NFS
export HF_HOME=/9950backfile/zccao/cache/huggingface
export HF_DATASETS_CACHE=/9950backfile/zccao/cache/huggingface/datasets
export MODELSCOPE_CACHE=/9950backfile/zccao/cache/modelscope
export TORCH_HOME=/9950backfile/zccao/cache/torch
export PIP_CACHE_DIR=/9950backfile/zccao/cache/pip

# Triton、CUDA、Torch 编译缓存放 /dev/shm
CACHE_ROOT=/dev/shm/zccao/cache
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/torch-extensions
export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/torchinductor
export CUDA_CACHE_PATH=$CACHE_ROOT/cuda-cache
export XDG_CACHE_HOME=$CACHE_ROOT/xdg-cache

# 临时文件放 /dev/shm
export TMPDIR=/dev/shm/zccao/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MODEL_PATH=/9950backfile/zccao/output/shopagent-sft/v0-20260928-173215/checkpoint-300 bash scripts/train_grpo.sh
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/ms-swift:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SHOP_ENV_URL="${SHOP_ENV_URL:-http://127.0.0.1:5000}"

MODEL_PATH="${MODEL_PATH:-/9950backfile/zccao/output/shopagent-sft/v0-20260928-173215/checkpoint-300}"
DATASET="$PROJECT_ROOT/data/rl_tasks_2000.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/shopagent-grpo}"
TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
mkdir -p "$OUTPUT_DIR"

# Four GPUs share training and vLLM rollout in colocate mode.
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6" \
NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29502 \
python3 -m swift.cli.main rlhf \
    --rlhf_type grpo \
    --model "$MODEL_PATH" \
    --tuner_type full \
    --dataset "$DATASET" \
    --external_plugins "$PROJECT_ROOT/src/train/swift_plugin.py" \
    --agent_template qwen3_5 \
    --enable_thinking false \
    --torch_dtype bfloat16 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_tensor_parallel_size 4 \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 20000 \
    --vllm_enforce_eager true \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --move_model_batches 16 \
    --multi_turn_scheduler shop_tool_scheduler \
    --gym_env shop_simulator \
    --use_gym_env true \
    --max_turns 40 \
    --max_length 16384 \
    --truncation_strategy left \
    --max_completion_length 256 \
    --num_generations 8 \
    --dynamic_sample true \
    --max_resample_times 2 \
    --generation_batch_size 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --use_liger_kernel true \
    --deepspeed zero3 \
    --beta 0.01 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.05 \
    --num_train_epochs 1 \
    --temperature 1.0 \
    --split_dataset_ratio 0.05 \
    --eval_strategy steps \
    --eval_steps 50 \
    --per_device_eval_batch_size 1 \
    --num_generations_eval 4 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --report_to tensorboard \
    --output_dir "$OUTPUT_DIR"
