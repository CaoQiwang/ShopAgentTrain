"""最小可用的 ShopSimulator 工具调用 Agent 循环。"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from ..env.action_guard import ActionGuardError, validate_action
from ..env.client import ShopEnv, TaskId
from ..env.observation import ObservationConverter
from ..env.tools import SHOP_TOOL_SCHEMAS, ToolCallError, tool_call_to_action


DEFAULT_SYSTEM_PROMPT = """你是一个在 ShopSimulator 中执行购物任务的智能体。

你必须依据最新一条 observation 行动，不得点击历史页面中的目标。

决策规则：
1. 仅当最新 observation 的 `search_available` 为 true 时调用 search_products。搜索词应简洁，搜索词由你设计。
2. 搜索结果页使用 open_product，asin 必须原样来自最新 observation 的 `actions`。
3. 需要补充信息时可查看 Description、Features、Reviews 或 Attributes。进入这些信息子页后，必须先调用 prev_page 返回商品详情页，才能查看其他信息或购买。
4. 使用 select_option 选择用户要求的规格。value 必须逐字取自最新 observation 的 `actions`。
5. 确保至少点击了一个商品规格，并且最新 observation 的 `actions` 中存在 `buy now`，才可调用 buy_now。
6. 每轮只调用一个工具。只返回一个结构化工具调用，不得输出普通文本。"""


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _observation_source(
    result: Mapping[str, Any], legacy_source: Any
) -> Any:
    """优先返回环境的结构化观测，缺失时兼容旧版文本响应。"""

    state = result.get("observation_state")
    if state is None:
        return legacy_source
    if not isinstance(state, Mapping):
        raise TypeError("环境返回的 observation_state 必须是 JSON object")
    return state


def _serialise_tool_call(call: Any) -> Dict[str, Any]:
    function = _get(call, "function", {})
    arguments = _get(function, "arguments", "{}") or "{}"
    if not isinstance(arguments, str):
        arguments = _json(arguments)
    return {
        "id": str(_get(call, "id") or f"call_{uuid.uuid4().hex[:12]}"),
        "type": "function",
        "function": {
            "name": str(_get(function, "name", "")),
            "arguments": arguments,
        },
    }


class AgentLoop:
    """使用 OpenAI 兼容客户端运行一条 ShopSimulator 轨迹。

    ``client`` 只需提供 ``client.chat.completions.create(...)``。环境和观测转换器
    均可注入，方便离线测试。
    """

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        env: Optional[ShopEnv] = None,
        converter: Optional[ObservationConverter] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 40,
        temperature: float = 0.0,
        max_tokens: int = 512,
        model_options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not model or not model.strip():
            raise ValueError("model 不能为空")
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")

        self.client = client
        self.model = model
        self.env = env or ShopEnv()
        self.converter = converter or ObservationConverter()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_options = dict(model_options or {})

        self.messages: List[Dict[str, Any]] = []
        self.trajectory: List[Dict[str, Any]] = []
        self.latest_observation: Dict[str, Any] = {}

    def run(self, task_id: TaskId) -> Dict[str, Any]:
        """运行单个任务，直到购买、主动结束、环境结束或达到步数上限。"""

        self.converter.reset()
        self.trajectory = []
        reset_result = self.env.reset(task_id)
        task = " ".join(
            str(reset_result.get("instruction", "")).replace("[SEP]", " ").split()
        )
        set_instruction = getattr(self.converter, "set_instruction", None)
        if callable(set_instruction):
            set_instruction(task)

        # 新环境直接返回结构化首页；旧环境才需要补出搜索首页状态。
        initial_raw = (
            task
            + "\n\n搜索功能是否可用: True"
            + "\n\n可点击的按钮: []"
        )
        self.latest_observation = self.converter.convert(
            _observation_source(reset_result, initial_raw)
        )
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": _json(
                    {
                        "task": task,
                        "observation": self.latest_observation,
                    }
                ),
            },
        ]

        result = self._base_result(task_id, task)
        try:
            for model_step in range(1, self.max_steps + 1):
                assistant_message, call = self._call_model()
                self.messages.append(assistant_message)

                if call is None:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "请只调用一个已注册工具，不要输出普通文本。",
                        }
                    )
                    continue

                call_id = call["id"]
                function = call["function"]
                name = str(function["name"])
                arguments = function.get("arguments", "{}")

                try:
                    action = tool_call_to_action(name, arguments)
                except ToolCallError as exc:
                    self._append_tool_result(
                        call_id,
                        {
                            "error": {
                                "type": "tool_call_format_invalid",
                                "message": str(exc),
                            }
                        },
                    )
                    continue

                event = {
                    "model_step": model_step,
                    "tool": name,
                    "arguments": self._parse_arguments(arguments),
                    "environment_action": action,
                }

                if name == "finish_without_purchase":
                    event["done"] = True
                    self.trajectory.append(event)
                    self._append_tool_result(
                        call_id, {"status": "finished_without_purchase"}
                    )
                    result.update(
                        status="finished_without_purchase",
                        steps=len(self.trajectory),
                    )
                    return self._finish_result(result)

                if action is None:
                    event["done"] = False
                    self.trajectory.append(event)
                    self._append_tool_result(
                        call_id,
                        {"warning": "该工具不与环境交互，请调用购物工具。"},
                    )
                    continue

                try:
                    validate_action(action, self.latest_observation)
                except ActionGuardError as exc:
                    # 场景校验失败不应调用 env.step，否则 ShopSimulator 会静默忽略
                    # 无效点击并返回相同页面，agent 无法判断失败原因。
                    self._append_tool_result(call_id, exc.feedback())
                    continue

                env_result = self.env.step(action)
                self.latest_observation = self.converter.convert(
                    _observation_source(env_result, env_result),
                    action=action,
                )
                self._append_tool_result(call_id, self.latest_observation)

                done = bool(env_result.get("done", False))
                over = bool(env_result.get("over", False))
                event.update(done=done, over=over)
                self.trajectory.append(event)

                if done or over:
                    result.update(
                        status="purchased" if done else "environment_over",
                        steps=len(self.trajectory),
                        reward=float(env_result.get("reward", 0) or 0),
                        reward_detail=env_result.get("reward_detail", {}),
                        purchase=env_result.get("purchase", {}),
                        goal=env_result.get("goal", {}),
                    )
                    return self._finish_result(result)

            result.update(status="max_steps", steps=len(self.trajectory))
            return self._finish_result(result)
        finally:
            # 购买或环境上限会由服务端自动释放；其余退出路径在这里归还槽位。
            if self.env.active:
                try:
                    self.env.release()
                except Exception:
                    # 清理失败不应覆盖已经得到的轨迹结果或原始异常。
                    pass

    def _call_model(
        self,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        request = {
            "model": self.model,
            "messages": deepcopy(self.messages),
            "tools": SHOP_TOOL_SCHEMAS,
            "tool_choice": "required",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.model_options,
        }

        # Qwen3/vLLM 在未指定时默认开启 thinking。购物 agent 只需要一个简短的
        # 结构化工具调用，因此必须显式关闭，而不能依赖“没有开启”等同于关闭。
        extra_body = request.get("extra_body")
        if extra_body is None:
            extra_body_dict: Dict[str, Any] = {}
        elif isinstance(extra_body, Mapping):
            extra_body_dict = dict(extra_body)
        else:
            raise TypeError("model_options.extra_body 必须是 JSON object")

        # 显式传入顶层 enable_thinking 时采用 DashScope 等服务的参数格式；否则
        # 保持现有 vLLM/Qwen chat_template_kwargs 默认行为。
        if "enable_thinking" not in extra_body_dict:
            chat_template_kwargs = extra_body_dict.get("chat_template_kwargs")
            if chat_template_kwargs is None:
                chat_template_kwargs_dict: Dict[str, Any] = {}
            elif isinstance(chat_template_kwargs, Mapping):
                chat_template_kwargs_dict = dict(chat_template_kwargs)
            else:
                raise TypeError(
                    "model_options.extra_body.chat_template_kwargs 必须是 JSON object"
                )
            chat_template_kwargs_dict.setdefault("enable_thinking", False)
            extra_body_dict["chat_template_kwargs"] = chat_template_kwargs_dict
        request["extra_body"] = extra_body_dict

        completion = self.client.chat.completions.create(**request)
        choices = _get(completion, "choices", [])
        if not choices:
            raise RuntimeError("模型响应中没有 choices")

        choice = choices[0]
        message = _get(choice, "message")
        if message is None:
            raise RuntimeError("模型响应中没有 message")
        raw_calls = list(_get(message, "tool_calls", None) or [])
        calls = [_serialise_tool_call(raw_calls[0])] if raw_calls else []

        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": _get(message, "content", "") or "",
        }
        if calls:
            assistant_message["tool_calls"] = calls
        return assistant_message, calls[0] if calls else None

    def _append_tool_result(self, call_id: str, content: Any) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _json(content),
            }
        )

    @staticmethod
    def _parse_arguments(arguments: Any) -> Any:
        if isinstance(arguments, Mapping):
            return dict(arguments)
        if isinstance(arguments, str):
            try:
                return json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return arguments
        return arguments

    def _base_result(self, task_id: TaskId, task: str) -> Dict[str, Any]:
        return {
            "task_id": task_id,
            "task": task,
            "status": "running",
            "steps": 0,
            "reward": 0.0,
            "reward_detail": {},
            "purchase": {},
            "goal": {},
        }

    def _finish_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        result["final_observation"] = deepcopy(self.latest_observation)
        result["trajectory"] = deepcopy(self.trajectory)
        result["messages"] = deepcopy(self.messages)
        return result


def run_agent_loop(
    client: Any,
    model: str,
    task_id: TaskId,
    **kwargs: Any,
) -> Dict[str, Any]:
    """创建 :class:`AgentLoop` 并运行单个任务的便捷入口。"""

    return AgentLoop(client, model, **kwargs).run(task_id)


__all__ = ["AgentLoop", "DEFAULT_SYSTEM_PROMPT", "run_agent_loop"]
