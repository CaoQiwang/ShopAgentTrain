#!/usr/bin/env bash
set -euo pipefail
# bash /9950backfile/zccao/workspace/shopagent/scripts/train_sft.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/ms-swift:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 持久化缓存放 NFS
export HF_HOME=/9950backfile/zccao/cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export MODELSCOPE_CACHE=/9950backfile/zccao/cache/modelscope
export TORCH_HOME=/9950backfile/zccao/cache/torch
export PIP_CACHE_DIR=/9950backfile/zccao/cache/pip

# 编译缓存和临时文件放 /dev/shm
CACHE_ROOT=/dev/shm/zccao/cache
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/torch-extensions
export TORCHINDUCTOR_CACHE_DIR=$CACHE_ROOT/torchinductor
export CUDA_CACHE_PATH=$CACHE_ROOT/cuda-cache
export XDG_CACHE_HOME=$CACHE_ROOT/xdg-cache
export TMPDIR=/dev/shm/zccao/tmp

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$MODELSCOPE_CACHE" \
    "$TORCH_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$TMPDIR"

# 转换 tool_calls，避免被当前 ms-swift 的数据预处理丢弃；原文件不变
DATA=/9950backfile/zccao/workspace/shopagent/data/qwen3.5-397b-a17b_reward1_sft.jsonl
SWIFT_DATA="${DATA%.jsonl}_swift.jsonl"
python3 - "$DATA" "$SWIFT_DATA" <<'PY'
import json, os, sys
from pathlib import Path
from src.env.tools import SHOP_TOOL_SCHEMAS

source, target = map(Path, sys.argv[1:])
if source.resolve() == target.resolve():
    raise SystemExit("不能覆盖原始数据")
temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
count = line_no = 0
try:
    with source.open(encoding="utf-8-sig") as src, temp.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = []
            for msg in row["messages"]:
                calls = msg.get("tool_calls")
                if calls:
                    if msg["role"] != "assistant" or not isinstance(calls, list):
                        raise ValueError("tool_calls 必须是 assistant 消息中的数组")
                    if msg.get("content"):
                        messages.append({"role": "assistant", "content": msg["content"]})
                    for call in calls:
                        fn = dict(call["function"])
                        if isinstance(fn["arguments"], str):
                            fn["arguments"] = json.loads(fn["arguments"])
                        if not isinstance(fn["arguments"], dict) or not isinstance(fn.get("name"), str):
                            raise ValueError("工具 name 必须为字符串，arguments 必须为 JSON object")
                        messages.append({"role": "tool_call", "content": json.dumps(fn, ensure_ascii=False)})
                else:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            if not any(m["role"] in {"assistant", "tool_call"} and m["content"] for m in messages):
                raise ValueError("没有可训练的 assistant/tool_call 内容")
            tools = row.get("tools") or SHOP_TOOL_SCHEMAS
            if isinstance(tools, str):
                tools = json.loads(tools)
            if not isinstance(tools, list):
                raise ValueError("tools 必须是工具定义数组")
            row.update(messages=messages, tools=json.dumps(tools, ensure_ascii=False))
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    if not count:
        raise ValueError("没有有效数据")
    os.replace(temp, target)
except (ValueError, TypeError, KeyError, AttributeError) as exc:
    raise SystemExit(f"第 {line_no} 行数据转换失败：{exc}") from exc
finally:
    temp.unlink(missing_ok=True)
print(f"已准备 {count} 条轨迹：{target}")
PY

CUDA_VISIBLE_DEVICES=0,1,6,7 \
NPROC_PER_NODE=4 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 \
python3 -m swift.cli.main sft \
    --model /9950backfile/zccao/weights/Qwen3.5-4B \
    --tuner_type full \
    --dataset "$SWIFT_DATA" \
    --agent_template qwen3_5 \
    --freeze_vit true \
    --freeze_aligner true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --learning_rate 5e-6 \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --max_length 16384 \
    --truncation_strategy delete \
    --split_dataset_ratio 0.01 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 5 \
    --save_only_model true \
    --add_version true \
    --logging_steps 5 \
    --dataset_num_proc 1 \
    --dataloader_num_workers 0 \
    --report_to tensorboard \
    --output_dir /9950backfile/zccao/output/shopagent-sft
