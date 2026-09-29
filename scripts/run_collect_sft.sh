#!/usr/bin/env bash
set -Eeuo pipefail

# 默认从 output/qwen3.5-397b-a17b 中收集 reward=1 的 rollout，并转换为
# 每行 {"messages": [...]} 的 chat-SFT JSONL，保存到项目 data 目录。
# MAX_TASK_ID 是 SFT/RL 的排他性分界：task_id >= MAX_TASK_ID 的数据保留给 RL。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-397b-a17b}"
INPUT_ROOT="${INPUT_ROOT:-output}"
MIN_TASK_ID="${MIN_TASK_ID:-1495}"
MAX_TASK_ID="${MAX_TASK_ID:-}"
OUTPUT_PATH="${OUTPUT_PATH:-data/${MODEL_NAME}_reward1_sft.jsonl}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "错误：找不到 Python：${PYTHON_BIN}" >&2
  exit 127
fi
if [[ -z "${MAX_TASK_ID}" ]]; then
  echo "错误：必须设置 MAX_TASK_ID，task_id >= MAX_TASK_ID 的任务将保留给 RL" >&2
  echo "示例：MAX_TASK_ID=10000 bash scripts/run_collect_sft.sh" >&2
  exit 2
fi
if [[ ! "${MIN_TASK_ID}" =~ ^[0-9]+$ || ! "${MAX_TASK_ID}" =~ ^[0-9]+$ ]]; then
  echo "错误：MIN_TASK_ID 和 MAX_TASK_ID 必须是非负整数" >&2
  exit 2
fi
if (( MAX_TASK_ID <= MIN_TASK_ID )); then
  echo "错误：MAX_TASK_ID 必须大于 MIN_TASK_ID" >&2
  exit 2
fi

command=(
  "${PYTHON_BIN}" scripts/collect_sft.py
  --model "${MODEL_NAME}"
  --input-root "${INPUT_ROOT}"
  --min-task-id "${MIN_TASK_ID}"
  --max-task-id "${MAX_TASK_ID}"
)

command+=(--output-path "${OUTPUT_PATH}")

# 允许在调用末尾附加 collect_sft.py 的其他参数。
command+=("$@")

echo "项目目录: ${PROJECT_ROOT}"
echo "模型: ${MODEL_NAME}"
echo "轨迹目录: ${INPUT_ROOT}/${MODEL_NAME}"
echo "SFT task_id 范围: [${MIN_TASK_ID}, ${MAX_TASK_ID})"
echo "RL task_id 范围: [${MAX_TASK_ID}, +∞)"
echo "输出格式: 每行一个 {\"messages\": [...]} 样本"
echo "输出文件: ${OUTPUT_PATH}"

exec "${command[@]}"
