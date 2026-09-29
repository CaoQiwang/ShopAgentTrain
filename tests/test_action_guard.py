"""action_guard 的离线单元测试。"""

import importlib.util
import sys
import types
import unittest

# 这组测试完全离线；精简测试环境没有 requests 时，提供仅用于导入类型的替身。
if importlib.util.find_spec("requests") is None:
    requests_stub = types.ModuleType("requests")
    requests_stub.Session = object  # type: ignore[attr-defined]
    requests_stub.RequestException = Exception  # type: ignore[attr-defined]
    sys.modules["requests"] = requests_stub

from src.core.agent_loop import AgentLoop
from src.env.action_guard import (
    ActionFormatError,
    ActionUnavailableError,
    validate_action,
)


class _FakeCompletions:
    def __init__(self, tool_name: str, arguments: str) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.last_request = None

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.last_request = kwargs
        return {
            "id": "completion_test",
            "model": "test-model",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "reasoning": "diagnostic reasoning",
                        "tool_calls": [
                            {
                                "id": "call_test",
                                "function": {
                                    "name": self.tool_name,
                                    "arguments": self.arguments,
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 12,
                "total_tokens": 112,
            },
        }


class _FakeClient:
    def __init__(self, tool_name: str, arguments: str = "{}") -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(tool_name, arguments)


class _FakeEnv:
    def __init__(self) -> None:
        self.active = False
        self.step_calls = 0

    def reset(self, task_id):  # type: ignore[no-untyped-def]
        return {"instruction": "测试任务"}

    def step(self, action):  # type: ignore[no-untyped-def]
        self.step_calls += 1
        raise AssertionError("场景非法动作不应发送给环境")


class _FakeConverter:
    def reset(self) -> None:
        pass

    def convert(self, observation, action=None):  # type: ignore[no-untyped-def]
        return {
            "observation_version": "my-shopping-observation-v1",
            "page_type": "product_subpage",
            "search_available": False,
            "actions": ["back to search", "< prev"],
        }


class _StructuredEnv:
    def __init__(self) -> None:
        self.active = False
        self.step_calls = 0

    def reset(self, task_id):  # type: ignore[no-untyped-def]
        return {
            "instruction": "测试任务",
            "observation_state": {
                "observation_version": "my-shopping-observation-v1",
                "page_type": "search_results",
                "query": "测试商品",
                "products": [],
                "search_available": False,
                "actions": ["12345678"],
            },
        }

    def step(self, action):  # type: ignore[no-untyped-def]
        self.step_calls += 1
        return {
            "instruction": "这是不应被优先解析的旧版观测",
            "observation_state": {
                "observation_version": "my-shopping-observation-v1",
                "page_type": "terminal",
                "search_available": False,
                "actions": [],
            },
            "done": True,
            "over": True,
            "reward": 1.0,
            "reward_detail": {},
            "purchase": {},
            "goal": {},
        }


class _RecordingConverter:
    def __init__(self) -> None:
        self.inputs = []

    def reset(self) -> None:
        self.inputs = []

    def set_instruction(self, instruction):  # type: ignore[no-untyped-def]
        pass

    def convert(self, observation, action=None):  # type: ignore[no-untyped-def]
        self.inputs.append((observation, action))
        return dict(observation)


class ActionGuardTests(unittest.TestCase):
    def test_agent_loop_prefers_structured_observation_on_reset_and_step(self) -> None:
        env = _StructuredEnv()
        converter = _RecordingConverter()
        result = AgentLoop(
            _FakeClient("open_product", '{"asin":"12345678"}'),
            "test-model",
            env=env,
            converter=converter,
            max_steps=1,
        ).run(0)

        self.assertEqual(env.step_calls, 1)
        self.assertEqual(converter.inputs[0][0]["page_type"], "search_results")
        self.assertEqual(converter.inputs[1][0]["page_type"], "terminal")
        self.assertNotIn("instruction", converter.inputs[1][0])
        self.assertEqual(result["final_observation"]["page_type"], "terminal")

    def test_search_is_allowed_when_search_is_available(self) -> None:
        validate_action(
            "search[儿童座椅]",
            {"search_available": True, "actions": []},
        )

    def test_search_is_rejected_on_product_page(self) -> None:
        with self.assertRaises(ActionUnavailableError) as caught:
            validate_action(
                "search[儿童座椅]",
                {
                    "page_type": "product_detail",
                    "search_available": False,
                    "actions": ["buy now"],
                },
            )

        feedback = caught.exception.feedback()["error"]
        self.assertEqual(feedback["type"], "action_unavailable")
        self.assertEqual(feedback["page_type"], "product_detail")

    def test_click_must_be_in_latest_actions(self) -> None:
        with self.assertRaises(ActionUnavailableError) as caught:
            validate_action(
                "click[Features]",
                {
                    "page_type": "product_subpage",
                    "search_available": False,
                    "actions": ["back to search", "< prev"],
                },
            )

        feedback = caught.exception.feedback()["error"]
        self.assertEqual(feedback["type"], "action_unavailable")
        self.assertEqual(
            feedback["available_actions"], ["back to search", "< prev"]
        )

    def test_click_comparison_matches_environment_case_handling(self) -> None:
        validate_action(
            "click[Buy Now]",
            {
                "search_available": False,
                "actions": ["buy now"],
            },
        )

    def test_dynamic_target_must_come_from_latest_actions(self) -> None:
        with self.assertRaises(ActionUnavailableError):
            validate_action(
                "click[old-asin]",
                {
                    "page_type": "search_results",
                    "search_available": False,
                    "actions": ["current-asin"],
                },
            )

    def test_bad_action_format_has_distinct_error_type(self) -> None:
        with self.assertRaises(ActionFormatError) as caught:
            validate_action("click[]", {"actions": []})

        feedback = caught.exception.feedback()["error"]
        self.assertEqual(feedback["type"], "action_format_invalid")
        self.assertIn("expected_format", feedback)

    def test_agent_loop_returns_unavailable_error_without_calling_env(self) -> None:
        env = _FakeEnv()
        client = _FakeClient("view_features")
        result = AgentLoop(
            client,
            "test-model",
            env=env,
            converter=_FakeConverter(),
            max_steps=1,
        ).run(0)

        self.assertEqual(env.step_calls, 0)
        self.assertEqual(result["status"], "max_steps")
        self.assertNotIn("model_responses", result)
        self.assertEqual(client.chat.completions.last_request["max_tokens"], 512)
        self.assertFalse(
            client.chat.completions.last_request["extra_body"]
            ["chat_template_kwargs"]["enable_thinking"]
        )
        tool_message = result["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertIn('"type": "action_unavailable"', tool_message["content"])

    def test_agent_loop_labels_tool_call_format_error(self) -> None:
        env = _FakeEnv()
        result = AgentLoop(
            _FakeClient("not_a_registered_tool"),
            "test-model",
            env=env,
            converter=_FakeConverter(),
            max_steps=1,
        ).run(0)

        self.assertEqual(env.step_calls, 0)
        self.assertIn(
            '"type": "tool_call_format_invalid"',
            result["messages"][-1]["content"],
        )

    def test_agent_loop_uses_explicit_dashscope_thinking_parameter(self) -> None:
        client = _FakeClient("view_features")
        AgentLoop(
            client,
            "test-model",
            env=_FakeEnv(),
            converter=_FakeConverter(),
            max_steps=1,
            model_options={"extra_body": {"enable_thinking": False}},
        ).run(0)

        extra_body = client.chat.completions.last_request["extra_body"]
        self.assertEqual(extra_body, {"enable_thinking": False})


if __name__ == "__main__":
    unittest.main()
