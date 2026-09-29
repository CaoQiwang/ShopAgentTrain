import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_sft import collect_sft_examples, write_jsonl_atomic


def _write_result(directory: Path, task_id: int, **updates: object) -> None:
    result = {
        "task_id": task_id,
        "reward": 1.0,
        "messages": [
            {"role": "user", "content": f"task {task_id}"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call-{task_id}",
                        "type": "function",
                        "function": {"name": "buy_now", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": f"call-{task_id}", "content": "{}"},
        ],
    }
    result.update(updates)
    (directory / f"{task_id}.json").write_text(
        json.dumps(result, ensure_ascii=False),
        encoding="utf-8",
    )


class CollectReward1SftTests(unittest.TestCase):
    def test_collect_excludes_first_1495_and_keeps_tool_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_result(directory, 1494)
            _write_result(directory, 1495, reward=0.5)
            _write_result(directory, 1496)
            _write_result(
                directory,
                1497,
                messages=None,
                conversation=[{"role": "assistant", "content": "legacy"}],
            )
            _write_result(directory, 1498)

            examples, counts = collect_sft_examples(directory, max_task_id=1498)

        self.assertEqual(len(examples), 2)
        self.assertEqual(
            examples[0]["messages"][1]["tool_calls"][0]["id"],
            "call-1496",
        )
        self.assertEqual(
            examples[1],
            {"messages": [{"role": "assistant", "content": "legacy"}]},
        )
        self.assertEqual(counts["excluded_test_set"], 1)
        self.assertEqual(counts["reserved_for_rl"], 1)
        self.assertEqual(counts["not_full_reward"], 1)

    def test_task_range_must_not_be_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                collect_sft_examples(Path(temporary), max_task_id=1495)

    def test_write_jsonl_writes_one_compact_json_object_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "data" / "train.jsonl"
            examples = [
                {"messages": [{"role": "user", "content": "你好"}]},
                {"messages": [{"role": "assistant", "content": "世界"}]},
            ]

            write_jsonl_atomic(destination, examples)
            lines = destination.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual([json.loads(line) for line in lines], examples)
        self.assertIn("你好", lines[0])


if __name__ == "__main__":
    unittest.main()
