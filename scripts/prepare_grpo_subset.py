#!/usr/bin/env python3
"""从已有 GRPO JSONL 中提取前 N 条任务。"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/shop_grpo_tasks_10000.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/shop_grpo_tasks_4000.jsonl")
    parser.add_argument("--count", type=int, default=4000)
    args = parser.parse_args()

    if args.source.resolve() == args.output.resolve():
        parser.error("输出文件不能覆盖输入文件")
    lines = args.source.read_text(encoding="utf-8").splitlines()
    if not 0 < args.count <= len(lines):
        parser.error(f"count 必须在 1..{len(lines)} 之间")

    selected = lines[:args.count]
    task_ids = [json.loads(line)["env_config"]["task_id"] for line in selected]
    if len(set(task_ids)) != args.count:
        raise ValueError("抽取结果中有重复的 task_id")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(f"已写入前 {args.count} 条任务：{args.output}")


if __name__ == "__main__":
    main()
