#!/usr/bin/env bash
set -euo pipefail

MODEL="qwen35-9b-tau2"
JUDGE_MODEL="deepseek-v4-pro"

cd "$(dirname "$0")/.."
python3 -m src.evaluation.llm_judge \
  --model "$MODEL" \
  --judge-model "$JUDGE_MODEL"
