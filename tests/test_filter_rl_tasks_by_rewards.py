"""RL task filtering by rollout reward groups."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.filter_rl_tasks_by_rewards import (
    filter_task_rows,
    load_reward_groups,
)


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class FilterRlTasksTests(unittest.TestCase):
    def test_keeps_every_group_except_all_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rewards_path = root / "rewards.jsonl"
            tasks_path = root / "tasks.jsonl"
            write_jsonl(
                rewards_path,
                [
                    {"task_id": 10, "rewards": [1, 1, 1, 1]},
                    {"task_id": 11, "rewards": [0, 0, 0, 0]},
                    {"task_id": 12, "rewards": [0.8, 0.8, 0.8, 0.8]},
                    {"task_id": 13, "rewards": [1, 0, 1, 0]},
                ],
            )
            write_jsonl(
                tasks_path,
                [
                    {"messages": [{"role": "user", "content": str(task_id)}],
                     "env_config": {"task_id": task_id}}
                    for task_id in range(10, 14)
                ],
            )

            groups, size = load_reward_groups(rewards_path)
            selected, counts = filter_task_rows(tasks_path, groups)

            self.assertEqual(size, 4)
            self.assertEqual(
                [row["env_config"]["task_id"] for row in selected],
                [11, 12, 13],
            )
            self.assertEqual(counts["excluded_all_one"], 1)
            self.assertEqual(counts["selected"], 3)

    def test_missing_rewards_are_strict_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks_path = root / "tasks.jsonl"
            write_jsonl(
                tasks_path,
                [
                    {"env_config": {"task_id": 10}},
                    {"env_config": {"task_id": 11}},
                ],
            )
            groups = {10: [0.0, 0.0]}
            with self.assertRaisesRegex(ValueError, "缺少奖励"):
                filter_task_rows(tasks_path, groups)

            selected, counts = filter_task_rows(
                tasks_path, groups, allow_partial=True
            )
            self.assertEqual(
                [row["env_config"]["task_id"] for row in selected], [10]
            )
            self.assertEqual(counts["missing_rewards"], 1)

    def test_rejects_inconsistent_reward_group_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rewards.jsonl"
            write_jsonl(
                path,
                [
                    {"task_id": 10, "rewards": [1, 0]},
                    {"task_id": 11, "rewards": [1, 0, 1]},
                ],
            )
            with self.assertRaisesRegex(ValueError, "预期 2"):
                load_reward_groups(path)

if __name__ == "__main__":
    unittest.main()
