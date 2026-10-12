"""Offline checks for the ShopSimulator GRPO rollout lifecycle."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.env.observation import OBSERVATION_VERSION
from src.env.result_reward import exact_match_result_reward
from src.train.shop_rollout_env import ShopRolloutEnv


def state(page_type="search", actions=None, search_available=True):
    return {
        "observation_version": OBSERVATION_VERSION,
        "page_type": page_type,
        "search_available": search_available,
        "actions": actions or [],
    }


class FakeShopEnv:
    instances = []
    fail_reset = False
    fail_step = False
    purchase_detail = {"r_type": 1.0, "r_att": 1.0, "r_option": 1.0, "r_price": True}
    purchase_reward = 1.0

    def __init__(self, base_url=None):
        self.base_url = base_url
        self.active = False
        self.actions = []
        self.close_count = 0
        self.__class__.instances.append(self)

    def reset(self, task_id):
        self.active = True
        if self.fail_reset:
            raise RuntimeError("reset failed after allocation")
        return {"instruction": "购买商品", "observation_state": state()}

    def step(self, action):
        if self.fail_step:
            raise RuntimeError("step failed")
        self.actions.append(action)
        if action.startswith("search["):
            return {
                "observation_state": state("search_results", ["A123456789"], False),
                "reward": 0,
            }
        if action == "click[A123456789]":
            return {
                "observation_state": state("product_detail", ["Buy Now"], False),
                "reward": 0,
            }
        if action == "click[Buy Now]":
            self.active = False
            return {
                "observation_state": state("done", [], False),
                "reward": self.purchase_reward,
                "reward_detail": self.purchase_detail,
                "done": True,
                "over": True,
            }
        raise AssertionError(action)

    def close(self):
        self.close_count += 1
        self.active = False


def call(name, arguments=None):
    return {
        "id": "call-1",
        "function": {"name": name, "arguments": json.dumps(arguments or {})},
    }


class ExactResultRewardTests(unittest.TestCase):
    def test_full_match_receives_one_and_only_option_mismatch_receives_point_three(self):
        full = {"r_type": 1.0, "r_att": 1.0, "r_option": 1.0, "r_price": True}
        self.assertEqual(exact_match_result_reward({"done": True, "reward_detail": full, "reward": 0.4}), 1.0)
        for key in full:
            with self.subTest(metric=key):
                partial = dict(full, **{key: 0.999})
                expected = 0.3 if key == "r_option" else 0.0
                self.assertEqual(exact_match_result_reward({"done": True, "reward_detail": partial}), expected)
                missing = {name: value for name, value in full.items() if name != key}
                self.assertEqual(exact_match_result_reward({"done": True, "reward_detail": missing}), 0.0)

        self.assertEqual(
            exact_match_result_reward({"done": True, "reward_detail": {**full, "r_option": 0}}),
            0.3,
        )
        self.assertEqual(
            exact_match_result_reward({"done": True, "reward_detail": {**full, "r_option": 0, "r_price": 0}}),
            0.0,
        )
        for invalid_option in (-1, 1.1, float("nan")):
            with self.subTest(invalid_option=invalid_option):
                self.assertEqual(
                    exact_match_result_reward({"done": True, "reward_detail": {**full, "r_option": invalid_option}}),
                    0.0,
                )

    def test_no_purchase_or_bad_detail_receives_zero(self):
        full = {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1}
        self.assertEqual(exact_match_result_reward({"done": False, "reward_detail": full}), 0.0)
        self.assertEqual(exact_match_result_reward({"over": True, "reward_detail": full}), 0.0)
        self.assertEqual(exact_match_result_reward({"done": True, "reward": 1.0}), 0.0)
        self.assertEqual(exact_match_result_reward({"done": True, "reward_detail": {**full, "r_att": "1"}}), 0.0)


class CoreRolloutTests(unittest.TestCase):
    def setUp(self):
        FakeShopEnv.purchase_detail = {"r_type": 1.0, "r_att": 1.0, "r_option": 1.0, "r_price": True}
        FakeShopEnv.purchase_reward = 1.0
        self.fake = FakeShopEnv()
        self.episode = ShopRolloutEnv(env=self.fake)
        asyncio.run(self.episode.reset(10000))

    def tearDown(self):
        asyncio.run(self.episode.close())

    def test_reset_accepts_older_observation_converter(self):
        class OlderConverter:
            def reset(self):
                pass

            def convert(self, observation, **kwargs):
                return state()

        episode = ShopRolloutEnv(env=FakeShopEnv(), converter=OlderConverter())
        try:
            initial_message, _ = asyncio.run(episode.reset(10000))
            self.assertIn("购买商品", initial_message)
        finally:
            asyncio.run(episode.close())

    def test_purchase_and_release(self):
        async def run():
            self.assertEqual((await self.episode.step([])).status, "no_tool_call")
            self.assertEqual((await self.episode.step([call("buy_now")])).status, "action_rejected")
            self.assertEqual((await self.episode.step([call("search_products", {"query": "商品"})])).status, "running")
            self.assertEqual((await self.episode.step([call("open_product", {"asin": "A123456789"})])).status, "running")
            bought = await self.episode.step([call("buy_now")])
            self.assertTrue(bought.done)
            self.assertEqual(bought.reward, 1.0)
            await self.episode.close()
            await self.episode.close()

        asyncio.run(run())
        self.assertEqual(self.fake.actions, ["search[商品]", "click[A123456789]", "click[Buy Now]"])
        self.assertEqual(self.fake.close_count, 1)

    def test_finish_without_purchase_does_not_hit_simulator(self):
        result = asyncio.run(
            self.episode.step([call("finish_without_purchase", {"reason": "no_suitable_product"})])
        )
        self.assertTrue(result.done)
        self.assertEqual(result.reward, 0)
        self.assertEqual(self.fake.actions, [])

    def test_invalid_and_multiple_calls(self):
        result = asyncio.run(self.episode.step([call("search_products")]))
        self.assertEqual(result.status, "invalid_tool_call")
        self.assertEqual(result.reward, -0.1)
        result = asyncio.run(self.episode.step([call("search_products"), call("buy_now")]))
        self.assertEqual(result.status, "multiple_tool_calls")
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(self.fake.actions, [])

    def test_partial_purchase_uses_simulator_reward(self):
        self.fake.purchase_detail = {"r_type": 1.0, "r_att": 0.5, "r_option": 1.0, "r_price": True}
        self.fake.purchase_reward = 0.8

        async def run():
            await self.episode.step([call("search_products", {"query": "商品"})])
            await self.episode.step([call("open_product", {"asin": "A123456789"})])
            return await self.episode.step([call("buy_now")])

        result = asyncio.run(run())
        self.assertTrue(result.done)
        self.assertEqual(result.reward, 0.8)

    def test_correct_product_with_wrong_option_gives_point_three(self):
        self.fake.purchase_detail = {"r_type": 1.0, "r_att": 1.0, "r_option": 0.5, "r_price": True}
        self.fake.purchase_reward = 0.3

        async def run():
            await self.episode.step([call("search_products", {"query": "商品"})])
            await self.episode.step([call("open_product", {"asin": "A123456789"})])
            return await self.episode.step([call("buy_now")])

        self.assertEqual(asyncio.run(run()).reward, 0.3)


def load_plugin_with_swift_stubs():
    registries = {"envs": {}, "multi_turns": {}}

    class Env:
        def __init__(self, env_config):
            self.env_config = env_config

    class MultiTurnScheduler:
        def __init__(self, infer_engine=None, max_turns=None, **kwargs):
            self.infer_engine = infer_engine
            self.max_turns = max_turns

        @staticmethod
        def _extract_logprobs_from_choice(choice):
            return [] if choice.logprobs is None else [x["logprob"] for x in choice.logprobs["content"]]

    class Qwen3_5AgentTemplate:
        def _format_tools(self, tools, system):
            return "# Tools\n" + ",".join(t["function"]["name"] for t in tools) + "\n" + system

    def module(name, **attributes):
        result = types.ModuleType(name)
        result.__dict__.update(attributes)
        return result

    fake_modules = {
        "swift": module("swift"),
        "swift.agent_template": module("swift.agent_template"),
        "swift.agent_template.qwen3_coder": module(
            "swift.agent_template.qwen3_coder", Qwen3_5AgentTemplate=Qwen3_5AgentTemplate
        ),
        "swift.infer_engine": module("swift.infer_engine"),
        "swift.infer_engine.protocol": module(
            "swift.infer_engine.protocol", RolloutOutput=lambda **kw: types.SimpleNamespace(**kw)
        ),
        "swift.rollout": module("swift.rollout"),
        "swift.rollout.gym_env": module("swift.rollout.gym_env", Env=Env, envs=registries["envs"]),
        "swift.rollout.multi_turn": module(
            "swift.rollout.multi_turn", MultiTurnScheduler=MultiTurnScheduler,
            multi_turns=registries["multi_turns"],
        ),
        "swift.utils": module("swift.utils", remove_response=lambda messages: messages.pop()),
    }
    path = Path(__file__).resolve().parents[1] / "src/train/swift_plugin.py"
    with patch.dict(sys.modules, fake_modules):
        spec = importlib.util.spec_from_file_location("test_shop_swift_plugin", path)
        plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin)
    return plugin, registries


def choice(name=None, arguments=None, *, token_ids=None, finish_reason="stop"):
    tool_calls = [call(name, arguments)] if name else None
    return types.SimpleNamespace(
        message=types.SimpleNamespace(content=f"<tool_call>{name}</tool_call>" if name else "text", tool_calls=tool_calls),
        token_ids=token_ids if token_ids is not None else [1, 2],
        finish_reason=finish_reason,
        logprobs=None,
    )


class FakeEngine:
    def __init__(self, choices):
        self.choices = list(choices)

    async def infer_async(self, request, config, **kwargs):
        next_choice = self.choices.pop(0)
        if isinstance(next_choice, Exception):
            raise next_choice
        return types.SimpleNamespace(choices=[next_choice])


class SwiftAdapterTests(unittest.TestCase):
    def setUp(self):
        self.plugin, self.registries = load_plugin_with_swift_stubs()
        FakeShopEnv.instances.clear()
        FakeShopEnv.fail_reset = False
        FakeShopEnv.fail_step = False
        FakeShopEnv.purchase_detail = {"r_type": 1.0, "r_att": 1.0, "r_option": 1.0, "r_price": True}
        FakeShopEnv.purchase_reward = 1.0
        self.shop_patch = patch.object(self.plugin, "ShopEnv", FakeShopEnv)
        self.shop_patch.start()

    def tearDown(self):
        self.shop_patch.stop()

    @staticmethod
    def request(uuid="sample-1"):
        return types.SimpleNamespace(
            uuid=uuid,
            data_dict={"env_config": {"task_id": 10000}},
            messages=[{"role": "user", "content": "placeholder"}],
            tools=None,
            chat_template_kwargs=None,
        )

    def scheduler(self, choices, max_turns=40):
        return self.plugin.ShopToolScheduler(
            infer_engine=FakeEngine(choices), gym_env="shop_simulator", max_turns=max_turns
        )

    def test_server_rollout_keeps_every_turn_and_releases_slot(self):
        self.assertEqual(self.plugin.ShopMultiTurnScheduler.__bases__[0].__name__, "MultiTurnScheduler")
        self.assertIs(self.registries["multi_turns"]["shop_tool_scheduler"], self.plugin.ShopMultiTurnScheduler)
        scheduler = self.scheduler([
            choice("search_products", {"query": "商品"}, token_ids=[11]),
            choice("open_product", {"asin": "A123456789"}, token_ids=[22]),
            choice("buy_now", token_ids=[33]),
        ])
        request = self.request()
        result = asyncio.run(scheduler.run(request, object()))
        self.assertEqual(result.response_token_ids, [[11], [22], [33]])
        self.assertEqual(result.response_loss_mask, [[1], [1], [1]])
        self.assertEqual(result.rollout_infos["total_reward"], 1.0)
        self.assertEqual(result.rollout_infos["num_turns"], 3)
        self.assertEqual(result.messages[-1]["role"], "assistant")
        self.assertIn("# Tools", result.messages[0]["content"])
        self.assertIsNone(request.tools)
        self.assertEqual(FakeShopEnv.instances[-1].close_count, 1)
        self.assertFalse(scheduler._envs)

    def test_server_uses_simulator_score(self):
        FakeShopEnv.purchase_detail = {"r_type": 1.0, "r_att": 0.5, "r_option": 1.0, "r_price": True}
        FakeShopEnv.purchase_reward = 0.8
        scheduler = self.scheduler([
            choice("search_products", {"query": "商品"}),
            choice("open_product", {"asin": "A123456789"}),
            choice("buy_now"),
        ])
        result = asyncio.run(scheduler.run(self.request(), object()))
        self.assertEqual(result.rollout_infos["total_reward"], 0.8)
        self.assertEqual(result.rollout_infos["step_rewards"], [0.0, 0.0, 0.8])

    def test_invalid_action_penalizes_trajectory_once(self):
        scheduler = self.scheduler([
            choice(),
            choice("buy_now"),
            choice("search_products", {"query": "商品"}),
            choice("open_product", {"asin": "A123456789"}),
            choice("buy_now"),
        ])
        result = asyncio.run(scheduler.run(self.request(), object()))
        self.assertEqual(result.rollout_infos["step_rewards"], [-0.1, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(result.rollout_infos["total_reward"], 0.9)

    def test_colocate_hooks_advance_and_release(self):
        scheduler = self.scheduler([])
        request = self.request()

        async def run():
            await scheduler.on_trajectory_start([request])
            first = choice("search_products", {"query": "商品"}, token_ids=[1])
            request.messages.append({"role": "assistant", "content": first.message.content})
            self.assertFalse((await scheduler.on_turn_end(request, first, 1))["done"])
            transition = scheduler.step(request, first, 1)
            self.assertEqual(transition["response_token_ids"], [1])
            self.assertEqual(request.messages[-1]["role"], "tool")
            final = choice("finish_without_purchase", {"reason": "no_suitable_product"})
            request.messages.append({"role": "assistant", "content": final.message.content})
            self.assertTrue((await scheduler.on_turn_end(request, final, 2))["done"])

        asyncio.run(run())
        self.assertEqual(FakeShopEnv.instances[-1].close_count, 1)

    def test_limits_and_failures_release_slot(self):
        scenarios = [
            ([choice()], 1, "max_turns"),
            ([choice(finish_reason="length")], 40, "generation_truncated"),
        ]
        for outputs, turns, status in scenarios:
            with self.subTest(status=status):
                result = asyncio.run(self.scheduler(outputs, turns).run(self.request(), object()))
                self.assertEqual(result.rollout_infos["status"], status)
                self.assertEqual(result.messages[-1]["role"], "assistant")
                self.assertEqual(FakeShopEnv.instances[-1].close_count, 1)

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            asyncio.run(self.scheduler([RuntimeError("generation failed")]).run(self.request(), object()))
        self.assertEqual(FakeShopEnv.instances[-1].close_count, 1)

        FakeShopEnv.fail_step = True
        with self.assertRaisesRegex(RuntimeError, "step failed"):
            asyncio.run(self.scheduler([choice("search_products", {"query": "商品"})]).run(self.request(), object()))
        self.assertEqual(FakeShopEnv.instances[-1].close_count, 1)

        FakeShopEnv.fail_step = False
        FakeShopEnv.fail_reset = True
        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            asyncio.run(self.scheduler([]).run(self.request(), object()))
        self.assertEqual(FakeShopEnv.instances[-1].close_count, 1)

        FakeShopEnv.fail_reset = False
        missing_ids = choice("search_products", {"query": "商品"})
        missing_ids.token_ids = None
        with self.assertRaisesRegex(RuntimeError, "token IDs"):
            asyncio.run(self.scheduler([missing_ids]).run(self.request(), object()))
        self.assertEqual(FakeShopEnv.instances[-1].actions, [])
        self.assertEqual(FakeShopEnv.instances[-1].close_count, 1)


if __name__ == "__main__":
    unittest.main()
