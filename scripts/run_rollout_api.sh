#!/usr/bin/env bash
set -Eeuo pipefail

# 使用阿里云百炼 OpenAI 兼容 API 执行 ShopSimulator rollout。
# 密钥和 URL 从项目根目录的 .env 读取，已有 shell 环境变量优先于 .env。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"
cd "${PROJECT_ROOT}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "错误：找不到 .env 文件：${ENV_FILE}" >&2
  exit 2
fi

# 仅解析 KEY=VALUE，不执行 .env 中的命令。这样 API key 中的特殊字符也不会
# 被 shell 二次解释。调用脚本前显式设置的环境变量不会被 .env 覆盖。
while IFS= read -r line || [[ -n "${line}" ]]; do
  line="${line%$'\r'}"
  [[ "${line}" =~ ^[[:space:]]*$ || "${line}" =~ ^[[:space:]]*# ]] && continue
  line="${line#export }"
  [[ "${line}" == *=* ]] || continue

  key="${line%%=*}"
  value="${line#*=}"
  key="${key#"${key%%[![:space:]]*}"}"
  key="${key%"${key##*[![:space:]]}"}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"

  [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
  if [[ "${value}" == \"*\" && "${value}" == *\" ]] || \
     [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
    value="${value:1:${#value}-2}"
  fi
  if [[ ! -v "${key}" ]]; then
    printf -v "${key}" '%s' "${value}"
    export "${key}"
  fi
done < "${ENV_FILE}"

: "${DASHSCOPE_API_KEY:?错误：.env 中缺少 DASHSCOPE_API_KEY}"
: "${DASHSCOPE_URL:?错误：.env 中缺少 DASHSCOPE_URL}"

# 兼容误把 Markdown 链接粘贴进 .env 的情况；仍建议保存为纯 URL。
markdown_url_pattern='^\[([^]]+)\]\(([^)]+)\)$'
if [[ "${DASHSCOPE_URL}" =~ ${markdown_url_pattern} ]]; then
  DASHSCOPE_URL="${BASH_REMATCH[1]}"
fi
if [[ ! "${DASHSCOPE_URL}" =~ ^https?:// ]]; then
  echo "错误：DASHSCOPE_URL 必须是 http(s) URL" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-qwen35-9b-tau2}"
ENV_BASE_URL="${ENV_BASE_URL:-http://127.0.0.1:5000}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

WORKERS="${WORKERS:-4}"
TASK_START="${TASK_START:-0}"
TASK_COUNT="${TASK_COUNT:-1495}"
TASK_IDS="${TASK_IDS:-}"
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
if ! "${PYTHON_BIN}" -c 'import openai, requests' >/dev/null 2>&1; then
  echo "错误：缺少 Python 依赖，请执行：${PYTHON_BIN} -m pip install openai requests" >&2
  exit 3
fi

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

# 不把密钥放进命令行参数，避免它出现在进程列表中。
export OPENAI_API_KEY="${DASHSCOPE_API_KEY}"

command=(
  "${PYTHON_BIN}" -m src.evaluation.rollout
  --model "${MODEL_NAME}"
  --model-base-url "${DASHSCOPE_URL}"
  --env-base-url "${ENV_BASE_URL}"
  --model-extra-body '{"enable_thinking":false}'
  --workers "${WORKERS}"
  --output-dir "${OUTPUT_DIR}"
  --max-steps "${MAX_STEPS}"
  --max-tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --model-timeout "${MODEL_TIMEOUT}"
  --env-timeout "${ENV_TIMEOUT}"
)

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

# 允许在调用末尾附加 rollout.py 的其他参数。
command+=("$@")

echo "项目目录: ${PROJECT_ROOT}"
echo "模型: ${MODEL_NAME}"
echo "模型 API: ${DASHSCOPE_URL}"
echo "环境服务: ${ENV_BASE_URL}"
echo "并发数: ${WORKERS}"
if [[ -n "${TASK_IDS}" ]]; then
  echo "任务 ID: ${TASK_IDS}"
else
  echo "任务范围: ${TASK_START}..$((TASK_START + TASK_COUNT - 1))"
fi
echo "输出目录: ${OUTPUT_DIR}/${MODEL_NAME}"

exec "${command[@]}"
