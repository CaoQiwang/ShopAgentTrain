"""ShopSimulator episode logic shared by the ms-swift rollout adapter.

This module has no ms-swift dependency so the environment lifecycle can be
checked without loading a training model.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from src.core.agent_loop import DEFAULT_SYSTEM_PROMPT
from src.env.action_guard import ActionGuardError, validate_action
from src.env.client import ShopEnv
from src.env.observation import ObservationConverter
from src.env.tools import ToolCallError, tool_call_to_action


NO_TOOL_FEEDBACK = "请只调用一个已注册工具，不要输出普通文本。"
MULTIPLE_TOOLS_FEEDBACK = "每轮只能调用一个工具，请重新选择一个动作。"


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


@dataclass
class TurnResult:
    feedback: str = ""
    feedback_role: str = "user"
    tool_call_id: Optional[str] = None
    reward: float = 0.0
    done: bool = False
    status: str = "running"
    info: dict[str, Any] = field(default_factory=dict)


class ShopRolloutEnv:
    """One independent ShopEnv slot and observation converter per trajectory."""

    def __init__(
        self,
        *,
        env: Optional[ShopEnv] = None,
        converter: Optional[ObservationConverter] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.env = env or ShopEnv()
        self.converter = converter or ObservationConverter()
        self.system_prompt = system_prompt
        self.task_id: Optional[int] = None
        self.task = ""
        self.latest_observation: dict[str, Any] = {}
        self.closed = False
        self.done = False
        self.last_env_result: dict[str, Any] = {}
        self._invalid_action_penalized = False

    def _invalid_action_reward(self) -> float:
        reward = 0.0 if self._invalid_action_penalized else -0.1
        self._invalid_action_penalized = True
        return reward

    async def reset(self, task_id: int) -> tuple[str, str]:
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
            raise ValueError("env_config.task_id 必须是非负整数")
        if self.closed:
            raise RuntimeError("已关闭的环境不能重新使用")

        self.converter.reset()
        self._invalid_action_penalized = False
        reset_result = await asyncio.to_thread(self.env.reset, task_id)
        self.task_id = task_id
        self.task = " ".join(
            str(reset_result.get("instruction", "")).replace("[SEP]", " ").split()
        )
        if not self.task:
            raise ValueError(f"task_id={task_id} 的 reset 响应缺少 instruction")
        set_instruction = getattr(self.converter, "set_instruction", None)
        if callable(set_instruction):
            set_instruction(self.task)
        initial_raw = (
            self.task + "\n\n搜索功能是否可用: True\n\n可点击的按钮: []"
        )
        observation_source = reset_result.get("observation_state")
        self.latest_observation = self.converter.convert(
            initial_raw if observation_source is None else observation_source
        )
        initial_message = _json(
            {"task": self.task, "observation": self.latest_observation}
        )
        return initial_message, self.system_prompt

    async def step(self, tool_calls: Sequence[Any]) -> TurnResult:
        if self.closed or self.done or self.task_id is None:
            raise RuntimeError("环境未启动或已结束")
        if not tool_calls:
            return TurnResult(
                feedback=NO_TOOL_FEEDBACK,
                reward=self._invalid_action_reward(),
                status="no_tool_call",
            )
        if len(tool_calls) != 1:
            return TurnResult(
                feedback=MULTIPLE_TOOLS_FEEDBACK,
                reward=self._invalid_action_reward(),
                status="multiple_tool_calls",
            )

        call = tool_calls[0]
        call_id = _get(call, "id")
        call_id = str(call_id) if call_id else None
        function = _get(call, "function", {})
        name = _get(function, "name", "")
        arguments = _get(function, "arguments", "{}")

        try:
            action = tool_call_to_action(name, arguments)
        except ToolCallError as exc:
            return TurnResult(
                feedback=_json({"error": {"type": "tool_call_format_invalid", "message": str(exc)}}),
                feedback_role="tool",
                tool_call_id=call_id,
                reward=self._invalid_action_reward(),
                status="invalid_tool_call",
            )

        if name == "finish_without_purchase":
            self.done = True
            return TurnResult(
                feedback=_json({"status": "finished_without_purchase"}),
                feedback_role="tool",
                tool_call_id=call_id,
                done=True,
                status="finished_without_purchase",
            )

        try:
            validate_action(action, self.latest_observation)
        except ActionGuardError as exc:
            return TurnResult(
                feedback=_json(exc.feedback()),
                feedback_role="tool",
                tool_call_id=call_id,
                reward=self._invalid_action_reward(),
                status="action_rejected",
            )

        env_result = await asyncio.to_thread(self.env.step, action)
        self.last_env_result = dict(env_result)
        source = env_result.get("observation_state")
        self.latest_observation = self.converter.convert(
            env_result if source is None else source, action=action
        )
        done = bool(env_result.get("done"))
        over = bool(env_result.get("over"))
        self.done = done or over
        status = "purchased" if done else "environment_over" if over else "running"
        return TurnResult(
            feedback=_json(self.latest_observation),
            feedback_role="tool",
            tool_call_id=call_id,
            reward=float(env_result.get("reward", 0) or 0),
            done=self.done,
            status=status,
            info={"environment_action": action},
        )

    def terminate(self, reason: str) -> TurnResult:
        """End a truncated or overlong rollout without another simulator action."""

        self.done = True
        return TurnResult(done=True, status=reason)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.done = True
        await asyncio.to_thread(self.env.close)


__all__ = ["ShopRolloutEnv", "TurnResult"]
