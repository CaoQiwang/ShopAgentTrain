"""ms-swift 4.4.2 multi-turn GRPO adapter for ShopSimulator.

Load this module with ``--external_plugins src/train/swift_plugin.py`` and use
``--multi_turn_scheduler shop_tool_scheduler --gym_env shop_simulator
--use_gym_env true``. The rollout data supplies ``env_config.task_id``.
"""

from __future__ import annotations

import os
from typing import Any

from swift.agent_template.qwen3_coder import Qwen3_5AgentTemplate
from swift.infer_engine.protocol import RolloutOutput
from swift.rollout.gym_env import Env, envs
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns
from swift.utils import remove_response

from src.env.client import DEFAULT_BASE_URL, ShopEnv
from src.env.tools import SHOP_TOOL_SCHEMAS
from src.train.shop_rollout_env import ShopRolloutEnv


class ShopSwiftEnv(Env):
    """Bridge one ms-swift trajectory to one ShopSimulator HTTP slot."""

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)
        base_url = env_config.get("base_url") or os.environ.get(
            "SHOP_ENV_URL", DEFAULT_BASE_URL
        )
        self.episode = ShopRolloutEnv(env=ShopEnv(base_url=base_url))

    async def reset(self, config: Any) -> tuple[str, dict[str, Any], str]:
        task_id = self.env_config.get("task_id")
        observation, system_message = await self.episode.reset(task_id)
        return observation, {"task_id": task_id}, system_message

    async def step(self, action: list[dict[str, Any]]) -> tuple[str, float, bool, dict[str, Any]]:
        calls = action[-1].get("tool_calls") if action else None
        result = await self.episode.step(calls or [])
        info = {
            "status": result.status,
            "feedback_role": result.feedback_role,
            "tool_call_id": result.tool_call_id,
        }
        info.update(result.info)
        return result.feedback, result.reward, result.done, info

    async def terminate(self, reason: str) -> tuple[str, float, bool, dict[str, Any]]:
        result = self.episode.terminate(reason)
        return "", result.reward, True, {"status": result.status}

    async def close(self) -> None:
        await self.episode.close()


class ShopMultiTurnScheduler(MultiTurnScheduler):
    """Shop-specific multi-turn hooks and per-trajectory environment lifecycle."""

    def __init__(self, infer_engine: Any = None, max_turns: int | None = None, **kwargs: Any) -> None:
        max_turns = 40 if max_turns is None else max_turns
        if max_turns <= 0:
            raise ValueError("max_turns 必须大于 0")
        super().__init__(infer_engine=infer_engine, max_turns=max_turns, **kwargs)
        self.gym_env_name = kwargs.get("gym_env")
        self._envs: dict[str, ShopSwiftEnv] = {}
        self._total_rewards: dict[str, float] = {}
        self._step_rewards: dict[str, list[float]] = {}
        self._feedback: dict[str, tuple[str, str, str | None]] = {}
        self._tool_formatter = Qwen3_5AgentTemplate()

    def _create_env(self, env_config: dict[str, Any]) -> ShopSwiftEnv:
        env_name = env_config.get("name", self.gym_env_name)
        if env_name != "shop_simulator":
            raise ValueError(f"ShopMultiTurnScheduler 只能使用 shop_simulator，收到 {env_name!r}")
        if env_name not in envs:
            raise ValueError(f"未知环境 {env_name!r}；可用环境：{list(envs)}")
        return envs[env_name](env_config)

    async def _close_and_remove(self, uuid: str) -> None:
        env = self._envs.pop(uuid, None)
        try:
            if env is not None:
                try:
                    await env.close()
                except Exception:
                    # Preserve the rollout result or the original failure.
                    pass
        finally:
            self._total_rewards.pop(uuid, None)
            self._step_rewards.pop(uuid, None)
            self._feedback.pop(uuid, None)

    async def on_trajectory_start(self, requests: list[Any]) -> None:
        initialized: list[str] = []
        try:
            for req in requests:
                uuid = req.uuid
                if not uuid:
                    raise ValueError("ms-swift rollout request 缺少 uuid")
                if uuid in self._envs:
                    await self._close_and_remove(uuid)
                if req.tools:
                    raise ValueError("GRPO 数据不要设置 tools；适配器会把工具定义写入 system")

                env_config = (req.data_dict or {}).get("env_config", {})
                env = self._create_env(env_config)
                # Register before reset so a failed reset can still release its slot.
                self._envs[uuid] = env
                initialized.append(uuid)
                self._total_rewards[uuid] = 0.0
                self._step_rewards[uuid] = []
                observation, _info, system_message = await env.reset(req)

                # The dataset does not carry a tools column. Put exactly the
                # Qwen3.5 tool prompt in messages so both generation and the
                # trainer's later re-encoding see the same system text.
                tool_system = self._tool_formatter._format_tools(
                    SHOP_TOOL_SCHEMAS, system_message
                )
                req.messages = [
                    {"role": "system", "content": tool_system},
                    {"role": "user", "content": observation},
                ]
                req.tools = None
                req.chat_template_kwargs = dict(req.chat_template_kwargs or {})
                req.chat_template_kwargs.setdefault("enable_thinking", False)
        except BaseException:
            for uuid in initialized:
                await self._close_and_remove(uuid)
            raise

    async def on_turn_end(
        self, infer_request: Any, response_choice: Any, current_turn: int
    ) -> dict[str, Any]:
        uuid = infer_request.uuid
        env = self._envs.get(uuid)
        if env is None:
            raise RuntimeError(f"rollout 环境不存在：{uuid}")

        try:
            if response_choice.token_ids is None:
                raise RuntimeError("ms-swift 未返回生成 token IDs，无法对齐多轮 GRPO loss")
            if response_choice.finish_reason == "length":
                feedback, reward, done, info = await env.terminate(
                    "generation_truncated"
                )
            else:
                # The base scheduler already appended the raw generated text to
                # messages. Keep it intact for loss/token alignment, and pass the
                # parsed tool calls only to the environment executor.
                action = [{
                    "role": "assistant",
                    "content": response_choice.message.content,
                    "tool_calls": response_choice.message.tool_calls or [],
                }]
                feedback, reward, done, info = await env.step(action)
                if not done and current_turn >= self.max_turns:
                    _, final_reward, done, final_info = await env.terminate("max_turns")
                    reward += final_reward
                    info = {**info, **final_info}

            self._total_rewards[uuid] += float(reward)
            self._step_rewards[uuid].append(float(reward))
            if not done:
                self._feedback[uuid] = (
                    feedback,
                    info.get("feedback_role", "user"),
                    info.get("tool_call_id"),
                )

            rollout_infos = {
                "total_reward": self._total_rewards[uuid],
                "step_rewards": list(self._step_rewards[uuid]),
                "gym_done": done,
                "status": info.get("status", "running"),
                "task_id": env.episode.task_id,
            }
            if done:
                await self._close_and_remove(uuid)
            return {"done": done, "rollout_infos": rollout_infos}
        except BaseException:
            await self._close_and_remove(uuid)
            raise

    def step(self, infer_request: Any, response_choice: Any, current_turn: int) -> dict[str, Any]:
        uuid = infer_request.uuid
        if uuid not in self._feedback:
            raise RuntimeError(f"任务 {uuid} 缺少下一轮工具反馈")
        feedback, role, call_id = self._feedback.pop(uuid)
        message: dict[str, Any] = {"role": role, "content": feedback}
        if role == "tool" and call_id is not None:
            message["tool_call_id"] = call_id
        infer_request.messages.append(message)

        token_ids = list(response_choice.token_ids)
        return {
            "infer_request": infer_request,
            "response_token_ids": token_ids,
            "response_loss_mask": [1] * len(token_ids),
        }

    async def run(self, infer_request: Any, request_config: Any, **kwargs: Any) -> Any:
        """Server mode: retain every generated turn and always release its slot.

        ms-swift 4.4.2's base ``run`` omits the final turn's token IDs after a
        non-continuation turn. Tool responses always separate our turns, so
        collect IDs here before returning the rollout to GRPO.
        """

        try:
            await self.on_trajectory_start([infer_request])
            token_ids: list[list[int]] = []
            loss_masks: list[list[int]] = []
            rollout_logprobs: list[list[float]] = []
            rollout_infos: dict[str, Any] = {}
            current_turn = 1

            while True:
                messages = infer_request.messages
                if messages and messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
                    remove_response(messages)
                response = await self.infer_engine.infer_async(
                    infer_request, request_config, **kwargs
                )
                choice = response.choices[0]
                messages.append({"role": "assistant", "content": choice.message.content or ""})
                turn_result = await self.on_turn_end(infer_request, choice, current_turn)
                rollout_infos.update(turn_result.get("rollout_infos") or {})

                turn_ids = list(choice.token_ids)
                token_ids.append(turn_ids)
                loss_masks.append([1] * len(turn_ids))
                turn_logprobs = self._extract_logprobs_from_choice(choice)
                if turn_logprobs:
                    rollout_logprobs.append(turn_logprobs)

                if turn_result["done"]:
                    if sum(map(len, rollout_logprobs)) != sum(map(len, token_ids)):
                        rollout_logprobs = []
                        # OnPolicySample otherwise falls back to the last
                        # choice's logprobs as if this were a single turn.
                        choice.logprobs = None
                    return RolloutOutput(
                        response=response,
                        messages=messages,
                        response_token_ids=token_ids,
                        response_loss_mask=loss_masks,
                        rollout_infos={**rollout_infos, "num_turns": current_turn},
                        rollout_logprobs=rollout_logprobs,
                    )

                self.step(infer_request, choice, current_turn)
                current_turn += 1
        finally:
            if infer_request.uuid:
                await self._close_and_remove(infer_request.uuid)


ShopToolScheduler = ShopMultiTurnScheduler

envs["shop_simulator"] = ShopSwiftEnv
multi_turns["shop_tool_scheduler"] = ShopMultiTurnScheduler


__all__ = ["ShopSwiftEnv", "ShopMultiTurnScheduler", "ShopToolScheduler"]
