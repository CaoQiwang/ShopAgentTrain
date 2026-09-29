#!/usr/bin/env bash
set -Eeuo pipefail

# ShopSimulator 批量评测启动脚本。
# 所有配置均可通过同名环境变量覆盖，脚本参数会继续传给 rollout.py。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-qwen35-9b-tau2}"
# 显式设置 MODEL_BASE_URL="" 时使用 OpenAI SDK 的默认服务地址。
MODEL_BASE_URL="${MODEL_BASE_URL-http://127.0.0.1:8041/v1}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
ENV_BASE_URL="${ENV_BASE_URL:-http://127.0.0.1:5000}"

WORKERS="${WORKERS:-4}"
TASK_START="${TASK_START:-0}"
TASK_COUNT="${TASK_COUNT:-1495}"
TASK_IDS="${TASK_IDS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

MAX_STEPS="${MAX_STEPS:-40}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-120}"
ENV_TIMEOUT="${ENV_TIMEOUT:-60}"
OVERWRITE="${OVERWRITE:-0}"
PERSONA="${PERSONA:-0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "错误：找不到 Python：${PYTHON_BIN}" >&2
  exit 127
fi

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

command=(
  "${PYTHON_BIN}" -m src.evaluation.rollout
  --model "${MODEL_NAME}"
  --api-key "${API_KEY}"
  --env-base-url "${ENV_BASE_URL}"
  --workers "${WORKERS}"
  --output-dir "${OUTPUT_DIR}"
  --max-steps "${MAX_STEPS}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --model-timeout "${MODEL_TIMEOUT}"
  --env-timeout "${ENV_TIMEOUT}"
)

if [[ -n "${MODEL_BASE_URL}" ]]; then
  command+=(--model-base-url "${MODEL_BASE_URL}")
fi

if [[ -n "${TASK_IDS}" ]]; then
  command+=(--task-ids "${TASK_IDS}")
else
  command+=(--task-start "${TASK_START}" --task-count "${TASK_COUNT}")
fi

if is_true "${OVERWRITE}"; then
  command+=(--overwrite)
fi

if is_true "${PERSONA}"; then
  command+=(--persona)
fi

# 允许在脚本调用末尾追加 rollout.py 的其他参数，并覆盖前面的默认值。
command+=("$@")

echo "项目目录: ${PROJECT_ROOT}"
echo "模型: ${MODEL_NAME}"
echo "模型服务: ${MODEL_BASE_URL:-OpenAI 默认地址}"
echo "环境服务: ${ENV_BASE_URL}"
echo "并发环境数: ${WORKERS}"
echo "单轮最大输出 tokens: ${MAX_TOKENS}"
if [[ -n "${TASK_IDS}" ]]; then
  echo "任务 ID: ${TASK_IDS}"
else
  echo "任务范围: ${TASK_START}..$((TASK_START + TASK_COUNT - 1))"
fi
echo "输出目录: ${OUTPUT_DIR}/${MODEL_NAME}"

exec "${command[@]}"
