#!/usr/bin/env python3
"""将 ShopSimulator 预留任务转换为 ms-swift 多轮 GRPO 的输入 JSONL。

每行只包含公开的用户指令和环境任务 ID。在线 rollout 适配器需使用
``env_config.task_id`` 调用 ShopEnv.reset；购买目标和答案由环境持有，
不能放入模型输入。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "environments/ShopSimulator/shop_env/data/items_eval_train.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/shop_grpo_tasks_10000.jsonl"
DEFAULT_START_TASK_ID = 10000


def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            records = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取源数据 {path}: {exc}") from exc
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError("源数据必须是由 JSON object 组成的数组")
    return records


def validate_task_mapping(records: Sequence[dict[str, Any]]) -> None:
    """确保源数组下标仍与模拟器的 goal ID 一一对应。"""

    seen_asins: set[str] = set()
    for task_id, record in enumerate(records):
        asin = record.get("asin")
        instructions = record.get("instructions")
        if not isinstance(asin, str) or asin == "nan" or len(asin) > 20:
            raise ValueError(f"task_id={task_id}: 无效 asin，环境会跳过该商品")
        if asin in seen_asins:
            raise ValueError(f"task_id={task_id}: 重复 asin，环境会跳过该商品")
        seen_asins.add(asin)
        if not isinstance(instructions, list) or len(instructions) != 1:
            raise ValueError(
                f"task_id={task_id}: instructions 必须恰有一条，否则任务 ID 不匹配"
            )
        instruction = instructions[0]
        if not isinstance(instruction, dict) or not instruction.get("attributes"):
            raise ValueError(f"task_id={task_id}: 缺少目标属性，环境不会创建该任务")
        text = instruction.get("instruction")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"task_id={task_id}: 缺少公开的用户指令")


def make_row(task_id: int, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": record["instructions"][0]["instruction"]}
        ],
        "env_config": {"task_id": task_id},
    }


def write_jsonl_atomic(
    path: Path, records: Sequence[dict[str, Any]], start_task_id: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for task_id in range(start_task_id, len(records)):
                row = make_row(task_id, records[task_id])
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="生成 ShopSimulator 多轮 GRPO 任务输入 JSONL"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--start-task-id", type=int, default=DEFAULT_START_TASK_ID,
        help="首个纳入的环境任务 ID，包含此 ID（默认：10000）",
    )
    args = parser.parse_args(argv)

    if args.source.resolve() == args.output.resolve():
        parser.error("输出路径不能覆盖源数据")
    records = load_records(args.source)
    if not 0 <= args.start_task_id < len(records):
        parser.error(f"start-task-id 必须在 0..{len(records) - 1} 范围内")
    validate_task_mapping(records)
    for task_id in range(args.start_task_id, len(records)):
        if records[task_id].get("tag") != "train":
            raise ValueError(f"task_id={task_id}: 非 train 任务，拒绝写入 GRPO 数据")

    write_jsonl_atomic(args.output, records, args.start_task_id)
    print(f"输出文件：{args.output}")
    print(f"任务范围：{args.start_task_id}..{len(records) - 1}")
    print(f"任务数量：{len(records) - args.start_task_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
