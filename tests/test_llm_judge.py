import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.evaluation.llm_judge import (
    RolloutRecord,
    build_judge_payload,
    build_summary,
    is_strict_success,
    judge_record,
    load_first_rollouts,
    read_env_file,
    validate_judgment,
)


def valid_failure_judgment(task_id: int) -> dict:
    return {
        "task_id": task_id,
        "success": False,
        "failure_reason": {
            "category": "premature_termination",
            "explanation": "仍有可用搜索动作时提前结束。",
            "evidence_turns": [1],
        },
        "decision_evidence_sufficiency": {
            "score": 2,
            "explanation": "已有部分依据，但没有继续核验。",
            "evidence_turns": [1],
        },
        "process_efficiency": {
            "score": 3,
            "explanation": "没有重复，但结束过早。",
            "evidence_turns": [1],
        },
    }


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class LlmJudgeTests(unittest.TestCase):
    def test_read_env_file_supports_quotes_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text(
                "# comment\nexport DASHSCOPE_API_KEY='secret'\n"
                'DASHSCOPE_URL="https://example.test/v1"\n',
                encoding="utf-8",
            )
            values = read_env_file(path)

        self.assertEqual(values["DASHSCOPE_API_KEY"], "secret")
        self.assertEqual(values["DASHSCOPE_URL"], "https://example.test/v1")

    def test_load_first_rollouts_stops_at_limit_and_preserves_jsonl_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectories.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for task_id in (9, 2, 7, 1):
                    stream.write(
                        json.dumps(
                            {
                                "task_id": task_id,
                                "query": f"query {task_id}",
                                "result": {"status": "max_steps", "messages": []},
                            }
                        )
                        + "\n"
                    )
            records = load_first_rollouts(path, limit=3)

        self.assertEqual([record.task_id for record in records], [9, 2, 7])
        self.assertEqual(records[0].query, "query 9")

    def test_load_numeric_results_uses_numeric_task_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for task_id in (10, 2, 1):
                (directory / f"{task_id}.json").write_text(
                    json.dumps({"task": str(task_id), "status": "max_steps"}),
                    encoding="utf-8",
                )
            records = load_first_rollouts(directory, limit=2)

        self.assertEqual([record.task_id for record in records], [1, 2])

    def test_strict_success_uses_reward_or_all_four_components(self) -> None:
        self.assertTrue(is_strict_success({"reward": 1.0}))
        self.assertTrue(
            is_strict_success(
                {
                    "reward": 0,
                    "reward_detail": {
                        "r_type": 1,
                        "r_att": 1,
                        "r_option": 1,
                        "r_price": 1,
                    },
                }
            )
        )
        self.assertFalse(
            is_strict_success(
                {
                    "reward_detail": {
                        "r_type": 1,
                        "r_att": 1,
                        "r_option": 0,
                        "r_price": 1,
                    }
                }
            )
        )

    def test_payload_excludes_system_message_and_parses_tool_json(self) -> None:
        record = RolloutRecord(
            3,
            "买商品",
            {
                "status": "max_steps",
                "messages": [
                    {"role": "system", "content": "hidden"},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"task": "买商品", "observation": {"actions": []}}
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_products",
                                    "arguments": '{"query":"商品"}',
                                }
                            }
                        ],
                    },
                ],
            },
        )
        payload = build_judge_payload(record)

        self.assertEqual(len(payload["visible_trace"]), 2)
        self.assertNotIn("hidden", json.dumps(payload, ensure_ascii=False))
        self.assertEqual(
            payload["visible_trace"][1]["actions"][0]["arguments"],
            {"query": "商品"},
        )

    def test_validate_judgment_rejects_success_failure_reason(self) -> None:
        value = valid_failure_judgment(4)
        value["success"] = True
        with self.assertRaisesRegex(ValueError, "failure_reason"):
            validate_judgment(value, task_id=4, success=True, max_turn=1)

    def test_judge_record_requests_json_mode_and_validates_response(self) -> None:
        expected = valid_failure_judgment(8)
        completions = FakeCompletions(json.dumps(expected, ensure_ascii=False))
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        record = RolloutRecord(
            8,
            "测试任务",
            {"status": "finished_without_purchase", "messages": []},
        )
        # 给 evidence_turns 提供一个可引用的可见 turn。
        record.result["messages"] = [{"role": "user", "content": "初始页面"}]

        judgment = judge_record(client, record, retries=1)

        self.assertEqual(judgment, expected)
        self.assertEqual(
            completions.request["response_format"], {"type": "json_object"}
        )
        self.assertEqual(completions.request["temperature"], 0)

    def test_summary_aggregates_scores_and_failure_categories(self) -> None:
        judgment = valid_failure_judgment(1)
        records = [
            {
                "task_id": 1,
                "judge_status": "ok",
                "judgment": judgment,
            },
            {"task_id": 2, "judge_status": "error", "error": "failed"},
        ]
        summary = build_summary(
            records,
            target_model="target",
            judge_model="judge",
            source=Path("input"),
            details_path=Path("details.jsonl"),
        )

        self.assertEqual(summary["successful_judgment_count"], 1)
        self.assertEqual(summary["failed_judgment_count"], 1)
        self.assertEqual(summary["average_scores"]["process_efficiency"], 3)
        self.assertEqual(
            summary["failure_reason_counts"], {"premature_termination": 1}
        )


if __name__ == "__main__":
    unittest.main()
