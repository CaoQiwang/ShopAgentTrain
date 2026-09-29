#!/usr/bin/env python3
"""从单个模型的 rollout 结果中收集满分轨迹并生成 chat-SFT JSONL。

默认读取 ``output/<model>/*.json``，只保留
``1495 <= task_id < max_task_id`` 且 ``reward == 1`` 的结果，并写入
``output/data/<model>_reward1_sft.jsonl``。``max_task_id`` 及之后的数据留给
RL 使用。每一行采用通用的 chat-SFT 格式：``{"messages": [...]}``。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


DEFAULT_OUTPUT_ROOT = Path("output")
DEFAULT_MIN_TASK_ID = 1495
REWARD_TOLERANCE = 1e-9


def safe_model_name(model: str) -> str:
    """生成与 rollout.py 一致的跨平台模型目录名。"""

    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "__", model).strip(" .")
    return name or "unnamed-model"


def _task_id(result: Mapping[str, Any], path: Path) -> Optional[int]:
    """读取 task_id；兼容旧结果中仅以数字文件名记录任务 ID 的情况。"""

    value = result.get("task_id", path.stem)
    if isinstance(value, bool):
        return None
    try:
        task_id = int(value)
    except (TypeError, ValueError):
        return None

    # 拒绝 1495.5 这类会被 int() 截断的异常值。
    if isinstance(value, float) and not value.is_integer():
        return None
    if isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            return None
    return task_id


def _is_full_reward(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        reward = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(reward) and math.isclose(
        reward,
        1.0,
        rel_tol=0.0,
        abs_tol=REWARD_TOLERANCE,
    )


def _messages(result: Mapping[str, Any]) -> Optional[list[dict[str, Any]]]:
    """提取新格式 messages，并兼容旧格式 conversation。"""

    value = result.get("messages")
    if value is None:
        value = result.get("conversation")
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(message, dict) for message in value):
        return None
    return value


def _result_files(model_dir: Path) -> Iterable[Path]:
    """稳定地枚举结果文件；数字文件名优先按数值排序。"""

    def sort_key(path: Path) -> tuple[int, int | str]:
        try:
            return (0, int(path.stem))
        except ValueError:
            return (1, path.name)

    return sorted(model_dir.glob("*.json"), key=sort_key)


def collect_sft_examples(
    model_dir: Path,
    *,
    max_task_id: int,
    min_task_id: int = DEFAULT_MIN_TASK_ID,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """收集 SFT 样本，并返回样本及各类扫描计数。"""

    if min_task_id < 0:
        raise ValueError("min_task_id 不能为负数")
    if max_task_id <= min_task_id:
        raise ValueError("max_task_id 必须大于 min_task_id")

    examples: list[tuple[int, dict[str, Any]]] = []
    counts = {
        "scanned": 0,
        "selected": 0,
        "excluded_test_set": 0,
        "reserved_for_rl": 0,
        "not_full_reward": 0,
        "invalid_json": 0,
        "invalid_task_id": 0,
        "missing_messages": 0,
        "duplicate_task_id": 0,
    }
    selected_task_ids: set[int] = set()

    for path in _result_files(model_dir):
        counts["scanned"] += 1
        try:
            with path.open("r", encoding="utf-8") as stream:
                result = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            counts["invalid_json"] += 1
            print(f"警告：无法读取 {path}: {exc}", file=sys.stderr)
            continue

        if not isinstance(result, Mapping):
            counts["invalid_json"] += 1
            print(f"警告：根节点不是 JSON object，跳过 {path}", file=sys.stderr)
            continue

        task_id = _task_id(result, path)
        if task_id is None:
            counts["invalid_task_id"] += 1
            continue
        if task_id < min_task_id:
            counts["excluded_test_set"] += 1
            continue
        if task_id >= max_task_id:
            counts["reserved_for_rl"] += 1
            continue
        if not _is_full_reward(result.get("reward")):
            counts["not_full_reward"] += 1
            continue

        messages = _messages(result)
        if messages is None:
            counts["missing_messages"] += 1
            print(f"警告：满分轨迹缺少有效 messages/conversation，跳过 {path}", file=sys.stderr)
            continue
        if task_id in selected_task_ids:
            counts["duplicate_task_id"] += 1
            print(f"警告：task_id={task_id} 重复，跳过 {path}", file=sys.stderr)
            continue

        # 不加入 task_id/reward 等元数据，确保结果可直接用于标准 chat-SFT。
        examples.append((task_id, {"messages": messages}))
        selected_task_ids.add(task_id)

    examples.sort(key=lambda item: item[0])
    counts["selected"] = len(examples)
    return [example for _, example in examples], counts


def write_jsonl_atomic(path: Path, examples: Iterable[Mapping[str, Any]]) -> None:
    """原子写入 JSONL，避免中断时留下不完整的数据集。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for example in examples:
                stream.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")))
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
        description="收集指定模型 reward=1 的非测试集轨迹并转换为 SFT JSONL",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="模型名称，对应 input-root 下的模型目录",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="rollout 输出根目录（默认：output）",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="JSONL 输出路径（默认：output/data/<model>_reward1_sft.jsonl）",
    )
    parser.add_argument(
        "--min-task-id",
        type=int,
        default=DEFAULT_MIN_TASK_ID,
        help="纳入数据的最小 task_id（默认：1495，即排除 0-1494）",
    )
    parser.add_argument(
        "--max-task-id",
        type=int,
        required=True,
        help="SFT 数据不包含此 task_id 及之后的任务，它们将保留给 RL",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_task_id < 0:
        raise ValueError("min-task-id 不能为负数")
    if args.max_task_id <= args.min_task_id:
        raise ValueError("max-task-id 必须大于 min-task-id")

    model_name = safe_model_name(args.model)
    model_dir = args.input_root / model_name
    if not model_dir.is_dir():
        print(f"错误：模型结果目录不存在：{model_dir}", file=sys.stderr)
        return 2

    output_path = args.output_path or (
        args.input_root / "data" / f"{model_name}_reward1_sft.jsonl"
    )
    examples, counts = collect_sft_examples(
        model_dir,
        max_task_id=args.max_task_id,
        min_task_id=args.min_task_id,
    )
    write_jsonl_atomic(output_path, examples)

    print(f"模型结果目录：{model_dir}")
    print(f"扫描 JSON：{counts['scanned']}")
    print(f"写入满分 SFT 轨迹：{counts['selected']}")
    print(f"排除前 {args.min_task_id} 个测试任务：{counts['excluded_test_set']}")
    print(
        f"为 RL 保留 task_id >= {args.max_task_id} 的任务："
        f"{counts['reserved_for_rl']}"
    )
    print(f"输出文件：{output_path}")
    skipped_details = {
        key: value
        for key, value in counts.items()
        if key
        not in {"scanned", "selected", "excluded_test_set", "reserved_for_rl"}
        and value
    }
    if skipped_details:
        print("其他跳过统计：" + json.dumps(skipped_details, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
