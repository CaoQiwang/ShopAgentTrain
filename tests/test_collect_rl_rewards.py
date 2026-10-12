"""Reward-only RL rollout collection checks."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_rl_rewards import (
    CUSTOM_REWARD,
    load_completed,
    load_task_ids,
    rewrite_results,
    run_reward_rollout,
)


class FakeEnv:
    def __init__(self) -> None:
        self.closed = False

    def close(self):
        self.closed = True


class FakeLoop:
    def __init__(self) -> None:
        self.env = FakeEnv()
        self.task_ids = []

    def run(self, task_id):
        self.task_ids.append(task_id)
        return {
            "status": "purchased",
            "steps": 3,
            "reward": 0.75,
            "reward_detail": {"r_att": 0.75},
            "trajectory": ["not persisted"],
        }


class RewardRolloutTests(unittest.TestCase):
    def test_rollout_returns_run_rollout_simulator_reward(self):
        loop = FakeLoop()
        reward = asyncio.run(
            run_reward_rollout(
                10000,
                client=None,
                model="test-model",
                env_base_url="http://unused",
                env_timeout=1,
                max_turns=4,
                temperature=1.0,
                max_tokens=512,
                loop_factory=lambda: loop,
            )
        )

        self.assertEqual(reward, 0.75)
        self.assertEqual(loop.task_ids, [10000])
        self.assertTrue(loop.env.closed)

    def test_custom_rollout_converts_purchased_status_to_done(self):
        partial_loop = FakeLoop()
        partial_reward = asyncio.run(
            run_reward_rollout(
                10000,
                client=None,
                model="test-model",
                env_base_url="http://unused",
                env_timeout=1,
                max_turns=4,
                temperature=1.0,
                max_tokens=512,
                reward_mode=CUSTOM_REWARD,
                loop_factory=lambda: partial_loop,
            )
        )
        self.assertEqual(partial_reward, 0.0)

        full_loop = FakeLoop()
        full_loop.run = lambda task_id: {
            "status": "purchased",
            "steps": 3,
            "reward": 1.0,
            "reward_detail": {
                "r_type": 1.0,
                "r_att": 1.0,
                "r_option": 1.0,
                "r_price": True,
            },
        }
        full_reward = asyncio.run(
            run_reward_rollout(
                10000,
                client=None,
                model="test-model",
                env_base_url="http://unused",
                env_timeout=1,
                max_turns=4,
                temperature=1.0,
                max_tokens=512,
                reward_mode=CUSTOM_REWARD,
                loop_factory=lambda: full_loop,
            )
        )
        self.assertEqual(full_reward, 1.0)
        self.assertTrue(full_loop.env.closed)

    def test_load_task_ids_filters_inclusively_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tasks.jsonl"
            rows = [
                {"env_config": {"task_id": 9999}},
                {"env_config": {"task_id": 10000}},
                {"env_config": {"task_id": 10001}},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.assertEqual(load_task_ids(path, 10000), [10000, 10001])

            path.write_text(
                json.dumps(rows[1]) + "\n" + json.dumps(rows[1]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "重复"):
                load_task_ids(path, 10000)

    def test_resume_ignores_only_a_truncated_last_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rewards.jsonl"
            path.write_text(
                '{"task_id":10000,"rewards":[1,1,0,1]}\n{"task_id":',
                encoding="utf-8",
            )
            completed = load_completed(path, 4)
            self.assertEqual(completed, {10000: [1.0, 1.0, 0.0, 1.0]})
            rewrite_results(path, completed)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{"task_id":10000,"rewards":[1.0,1.0,0.0,1.0]}\n',
            )

    def test_resume_rejects_extra_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rewards.jsonl"
            path.write_text(
                '{"task_id":10000,"rewards":[1,1,1,1],"trajectory":[]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "只能包含"):
                load_completed(path, 4)


if __name__ == "__main__":
    unittest.main()
