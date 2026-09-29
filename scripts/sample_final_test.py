#!/usr/bin/env python3
"""从 ShopSimulator 测试集随机抽样并输出 JSONL。

默认从 ``items_eval_train.json`` 的前 1495 条测试任务中，以固定随机种子 42
无放回抽取 100 条，写入 ``data/final_test_100.jsonl``。每条记录保留原始字段，
并增加 ``task_id``（即记录在源数据列表中的下标）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


DEFAULT_SOURCE = Path(
    "environments/ShopSimulator/shop_env/data/items_eval_train.json"
)
DEFAULT_OUTPUT = Path("data/final_test_100.jsonl")
DEFAULT_TEST_SIZE = 1495
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SEED = 42


def load_records(path: Path) -> list[dict[str, Any]]:
    """读取由 JSON object 构成的顶层数组。"""

    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as exc:
        raise ValueError(f"源数据文件不存在：{path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取源数据文件 {path}：{exc}") from exc

    if not isinstance(value, list):
        raise ValueError("源数据顶层必须是 JSON array")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError("源数据中的每条记录必须是 JSON object")
    return value


def sample_records(
    records: Sequence[Mapping[str, Any]],
    *,
    test_size: int,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    """从 ``[0, test_size)`` 中无放回抽样并附加 task_id。"""

    if test_size <= 0:
        raise ValueError("test_size 必须大于 0")
    if test_size > len(records):
        raise ValueError(
            f"test_size={test_size} 超过源数据记录数 {len(records)}"
        )
    if sample_size <= 0:
        raise ValueError("sample_size 必须大于 0")
    if sample_size > test_size:
        raise ValueError("sample_size 不能大于 test_size")

    task_ids = random.Random(seed).sample(range(test_size), sample_size)
    sampled = [
        {"task_id": task_id, **dict(records[task_id])}
        for task_id in task_ids
    ]
    return sampled, task_ids


def write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """原子写入 JSONL，避免中断时留下不完整文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 ShopSimulator 测试集随机抽样并生成 JSONL",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"源 JSON 数组（默认：{DEFAULT_SOURCE}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出 JSONL（默认：{DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=DEFAULT_TEST_SIZE,
        help="源数据开头属于测试集的记录数（默认：1495）",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="抽样数量（默认：100）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="随机种子（默认：42）",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args.source)
    sampled, task_ids = sample_records(
        records,
        test_size=args.test_size,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    write_jsonl_atomic(args.output, sampled)

    print(f"源数据：{args.source}")
    print(f"测试集范围：0..{args.test_size - 1}")
    print(f"随机种子：{args.seed}")
    print(f"抽样数量：{len(sampled)}")
    print(f"输出文件：{args.output}")
    print("task_ids：" + ",".join(str(task_id) for task_id in task_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

