#!/usr/bin/env python3
"""按 rollout 奖励筛选 RL 任务，只排除奖励全部为 1 的任务。

奖励文件每行格式：``{"task_id": 10000, "rewards": [1, 0, 1, 1]}``。
任务文件每行需包含 ``env_config.task_id``。输出完整保留原任务对象，不写入奖励。
全 0、全 0.8、其他相同奖励或混合奖励都会保留；仅全 1 任务会被排除。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REWARDS = (
    PROJECT_ROOT
    / "data/qwen35-4b-sft-ckpt300_rollout_rewards_8.jsonl"
)
DEFAULT_TASKS = PROJECT_ROOT / "data/rl_tasks.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/rl_tasks_filtered.jsonl"


def load_reward_groups(path: Path) -> tuple[dict[int, list[float]], int]:
    """读取奖励组，并要求所有任务具有相同的 rollout 数量。"""

    groups: dict[int, list[float]] = {}
    group_size: Optional[int] = None
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"无法读取奖励文件 {path}: {exc}") from exc

    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: 不是合法 JSON") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number}: 必须是 JSON object")
            task_id = row.get("task_id")
            rewards = row.get("rewards")
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
                raise ValueError(f"{path}:{line_number}: task_id 必须是非负整数")
            if task_id in groups:
                raise ValueError(f"{path}:{line_number}: task_id={task_id} 重复")
            if not isinstance(rewards, list) or not rewards:
                raise ValueError(f"{path}:{line_number}: rewards 必须是非空数组")
            if not all(
                type(reward) in (int, float) and math.isfinite(float(reward))
                for reward in rewards
            ):
                raise ValueError(f"{path}:{line_number}: rewards 必须全部是有限数值")
            if group_size is None:
                group_size = len(rewards)
            elif len(rewards) != group_size:
                raise ValueError(
                    f"{path}:{line_number}: rewards 数量为 {len(rewards)}，"
                    f"预期 {group_size}"
                )
            groups[task_id] = [float(reward) for reward in rewards]

    if not groups or group_size is None:
        raise ValueError(f"奖励文件为空：{path}")
    return groups, group_size


def is_all_one(rewards: Sequence[float], tolerance: float) -> bool:
    """判断一组奖励是否全部等于 1，允许极小浮点误差。"""

    return all(
        math.isclose(reward, 1.0, rel_tol=0.0, abs_tol=tolerance)
        for reward in rewards
    )


def filter_task_rows(
    task_path: Path,
    reward_groups: Mapping[int, Sequence[float]],
    *,
    tolerance: float = 1e-9,
    allow_partial: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """保留有奖励且奖励不全为 1 的原始任务。"""

    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    missing: list[int] = []
    excluded_all_one = 0
    scored = 0

    try:
        stream = task_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"无法读取任务文件 {task_path}: {exc}") from exc

    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task_id = row["env_config"]["task_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"{task_path}:{line_number}: 缺少合法的 env_config.task_id"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{task_path}:{line_number}: 必须是 JSON object")
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
                raise ValueError(
                    f"{task_path}:{line_number}: task_id 必须是非负整数"
                )
            if task_id in seen:
                raise ValueError(f"{task_path}:{line_number}: task_id={task_id} 重复")
            seen.add(task_id)

            rewards = reward_groups.get(task_id)
            if rewards is None:
                missing.append(task_id)
                continue
            scored += 1
            if is_all_one(rewards, tolerance):
                excluded_all_one += 1
            else:
                selected.append(row)

    unknown_reward_ids = sorted(set(reward_groups) - seen)
    if unknown_reward_ids:
        raise ValueError(
            "奖励文件包含任务文件中不存在的 task_id："
            + ",".join(map(str, unknown_reward_ids[:10]))
        )
    if missing and not allow_partial:
        raise ValueError(
            f"有 {len(missing)} 个任务缺少奖励，例如："
            + ",".join(map(str, missing[:10]))
            + "；奖励尚未收集完整时可使用 --allow-partial"
        )

    counts = {
        "task_rows": len(seen),
        "scored": scored,
        "selected": len(selected),
        "excluded_all_one": excluded_all_one,
        "missing_rewards": len(missing),
    }
    return selected, counts


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rewards",
        type=Path,
        default=DEFAULT_REWARDS,
        help=f"rollout 奖励 JSONL（默认：{DEFAULT_REWARDS}）",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=DEFAULT_TASKS,
        help=f"原始 RL 任务 JSONL（默认：{DEFAULT_TASKS}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"筛选后的 RL 数据集（默认：{DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--one-tolerance",
        type=float,
        default=1e-9,
        help="判断奖励等于 1 时允许的绝对误差（默认：1e-9）",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="允许奖励文件尚未覆盖全部任务；缺少奖励的任务不会输出",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.one_tolerance < 0 or not math.isfinite(args.one_tolerance):
        parser.error("one-tolerance 必须是有限非负数")
    if args.output.resolve() in {args.rewards.resolve(), args.tasks.resolve()}:
        parser.error("输出文件不能覆盖奖励文件或原始任务文件")

    reward_groups, group_size = load_reward_groups(args.rewards)
    selected, counts = filter_task_rows(
        args.tasks,
        reward_groups,
        tolerance=args.one_tolerance,
        allow_partial=args.allow_partial,
    )
    write_jsonl_atomic(args.output, selected)

    print(f"奖励文件：{args.rewards}")
    print(f"原始任务：{args.tasks}")
    print(f"每任务 rollout 数：{group_size}")
    print(f"原始任务数：{counts['task_rows']}")
    print(f"已有奖励任务数：{counts['scored']}")
    print(f"排除全 1 任务数：{counts['excluded_all_one']}")
    print(f"保留任务数：{counts['selected']}")
    print(f"缺少奖励任务数：{counts['missing_rewards']}")
    print(f"输出文件：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "filter_task_rows",
    "is_all_one",
    "load_reward_groups",
    "write_jsonl_atomic",
]
