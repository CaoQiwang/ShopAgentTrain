#!/usr/bin/env bash
set -Eeuo pipefail

# ShopSimulator rollout 结果评估脚本。
# 默认读取 output/<模型名称>/*.json，并将结果写入
# output/<模型名称>/eval/evaluation.json。
# 所有配置均可通过同名环境变量覆盖，脚本参数会继续传给 eval.py。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-qwen35-9b-tau2}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

# 设置 RESULT_DIR 后直接评估该目录，不再根据 MODEL_NAME 拼接目录。
RESULT_DIR="${RESULT_DIR:-}"
# 留空时由 eval.py 读取 summary.json 的 selected；没有 summary 时使用有效文件数。
EXPECTED_TASKS="${EXPECTED_TASKS:-}"
# 留空时保存到结果目录下的 eval/evaluation.json。
SAVE_PATH="${SAVE_PATH:-}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "错误：找不到 Python：${PYTHON_BIN}" >&2
  exit 127
fi

command=("${PYTHON_BIN}" -m src.evaluation.eval)

if [[ -n "${RESULT_DIR}" ]]; then
  command+=(--result-dir "${RESULT_DIR}")
  resolved_result_dir="${RESULT_DIR}"
else
  command+=(--model "${MODEL_NAME}" --output-dir "${OUTPUT_DIR}")
  resolved_result_dir="${OUTPUT_DIR}/${MODEL_NAME}"
fi

if [[ -n "${EXPECTED_TASKS}" ]]; then
  command+=(--expected-tasks "${EXPECTED_TASKS}")
fi

if [[ -n "${SAVE_PATH}" ]]; then
  command+=(--save-path "${SAVE_PATH}")
  resolved_save_path="${SAVE_PATH}"
else
  resolved_save_path="${resolved_result_dir}/eval/evaluation.json"
  command+=(--save-path "${resolved_save_path}")
fi

# 允许在脚本末尾继续追加 eval.py 支持的参数。
command+=("$@")

echo "项目目录: ${PROJECT_ROOT}"
echo "结果目录: ${resolved_result_dir}"
if [[ -n "${EXPECTED_TASKS}" ]]; then
  echo "评测任务总数: ${EXPECTED_TASKS}"
else
  echo "评测任务总数: 自动读取 summary.json 或按有效结果数推断"
fi
echo "评估结果: ${resolved_save_path}"

exec "${command[@]}"
