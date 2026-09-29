import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

# 精简测试环境没有 requests 时，提供仅用于导入类型的替身。
if "requests" not in sys.modules and importlib.util.find_spec("requests") is None:
    requests_stub = types.ModuleType("requests")
    requests_stub.Session = object  # type: ignore[attr-defined]
    requests_stub.RequestException = Exception  # type: ignore[attr-defined]
    sys.modules["requests"] = requests_stub

from src.evaluation.eval_100 import (
    build_jsonl_record,
    build_summary,
    load_final_test,
)


class FinalTestEvaluationTests(unittest.TestCase):
    def _write_dataset(self, path: Path, count: int) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for task_id in range(count):
                stream.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "instructions": [
                                {"instruction": f"用户问题 {task_id}"}
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def test_load_final_test_preserves_file_order_and_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final.jsonl"
            self._write_dataset(path, 3)
            tasks = load_final_test(path, expected_count=3)

        self.assertEqual(
            tasks,
            [(0, "用户问题 0"), (1, "用户问题 1"), (2, "用户问题 2")],
        )

    def test_load_final_test_rejects_wrong_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final.jsonl"
            self._write_dataset(path, 2)
            with self.assertRaises(ValueError):
                load_final_test(path, expected_count=3)

    def test_output_record_contains_query_and_complete_result(self) -> None:
        result = {
            "status": "purchased",
            "reward": 1.0,
            "trajectory": [{"tool": "buy_now"}],
            "messages": [{"role": "assistant", "content": ""}],
        }
        record = build_jsonl_record(7, "测试 query", result)

        self.assertEqual(record["task_id"], 7)
        self.assertEqual(record["query"], "测试 query")
        self.assertEqual(record["result"], result)

    def test_summary_counts_rewards_and_statuses(self) -> None:
        summary = build_summary(
            model="test-model",
            dataset=Path("data/final.jsonl"),
            results=[
                {"status": "purchased", "reward": 1.0},
                {"status": "max_steps", "reward": 0.25},
            ],
            trajectories_path=Path("output/trajectories.jsonl"),
        )

        self.assertEqual(summary["task_count"], 2)
        self.assertEqual(summary["full_reward_count"], 1)
        self.assertEqual(summary["average_reward"], 0.625)
        self.assertEqual(
            summary["status_counts"],
            {"max_steps": 1, "purchased": 1},
        )


if __name__ == "__main__":
    unittest.main()
