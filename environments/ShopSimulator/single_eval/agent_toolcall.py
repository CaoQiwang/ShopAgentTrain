"""ShopSimulator evaluator using OpenAI-compatible structured tool calls.

Place this file next to ``single_eval/env.py`` and run it with a YAML config:

    python agent_toolcall.py --yaml_name configs/standard/qwen35_9b_toolcall.yaml

The LLM never sends free-form text to ShopSimulator. A tool call is validated
against the latest observation and then translated locally to the environment's
existing ``search[...]`` or ``click[...]`` protocol.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from openai import OpenAI
from tqdm import tqdm

from env import ShopEnv


DEFAULT_SYSTEM_PROMPT = """你是一个在 ShopSimulator 中执行购物任务的智能体。

    你必须依据最新一条 observation 行动，不得点击历史页面中的目标。

    决策规则：
    1. 仅当“搜索功能是否可用: True”时调用 search_products。搜索词应简洁，搜索词由你设计
    2. 搜索结果页使用 open_product，asin 必须原样来自最新 observation 的“可点击的按钮”。
    3. 需要补充信息时可查看 Description、Features、Reviews 或 Attributes。
    4. 使用 select_option 选择用户要求的规格。value 必须逐字取自最新 observation 的“可点击的按钮”。
    5. 确保至少点击了一个商品的规格，才可调用 buy_now。
    6. 所有推理都在内部完成，只返回一个结构化工具调用。
"""


def _tool_schema(
    name: str,
    description: str,
    properties: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


SHOP_TOOLS: List[Dict[str, Any]] = [
    _tool_schema(
        "search_products",
        "在当前页面允许搜索时搜索商品。query 应简短并包含最关键的商品类别、品牌、型号、功能或规格。",
        {"query": {"type": "string", "description": "商品搜索关键词"}},
        ["query"],
    ),
    _tool_schema(
        "open_product",
        "打开当前搜索结果中的一个商品。asin 必须原样来自最新 observation 的可点击按钮。",
        {"asin": {"type": "string", "description": "当前页面中的商品 ASIN"}},
        ["asin"],
    ),
    _tool_schema(
        "select_option",
        "选择当前商品的一项规格。value 必须原样来自最新 observation 的可点击按钮，且不能是导航按钮。",
        {"value": {"type": "string", "description": "当前商品页中的规格值"}},
        ["value"],
    ),
    _tool_schema("view_description", "查看当前商品的 Description；无参数，传入 {}。"),
    _tool_schema("view_features", "查看当前商品的 Features；无参数，传入 {}。"),
    _tool_schema("view_reviews", "查看当前商品的 Reviews；无参数，传入 {}。"),
    _tool_schema("view_attributes", "查看当前商品的 Attributes；无参数，传入 {}。"),
    _tool_schema("next_page", "当前页没有合适候选且存在 Next > 时进入下一页；无参数，传入 {}。"),
    _tool_schema("prev_page", "存在 < Prev 时返回上一页；无参数，传入 {}。"),
    _tool_schema("back_to_search", "存在 Back to Search 时返回搜索页；无参数，传入 {}。"),
    _tool_schema("buy_now", "不可撤销的购买动作。仅在完整核验商品、规格和最终价格都满足用户要求时调用；无参数，传入 {}。",
    ),
]


FIXED_CLICK_TARGETS = {
    "view_description": "description",
    "view_features": "features",
    "view_reviews": "reviews",
    "view_attributes": "attributes",
    "next_page": "next >",
    "prev_page": "< prev",
    "back_to_search": "back to search",
    "buy_now": "buy now",
}

NAVIGATION_TARGETS = set(FIXED_CLICK_TARGETS.values())
TOOL_ARGUMENTS = {
    tool["function"]["name"]: set(
        tool["function"]["parameters"].get("properties", {}).keys()
    )
    for tool in SHOP_TOOLS
}


class AgentTermination(RuntimeError):
    """Expected, bounded task termination rather than an infrastructure crash."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def clickable_buttons(observation: str) -> List[str]:
    """Read the JSON button list appended by ShopSimulator's API wrapper."""
    matches = re.findall(r"可点击的按钮:\s*(\[[^\n]*\])", observation or "")
    if not matches:
        return []
    try:
        values = json.loads(matches[-1])
    except json.JSONDecodeError:
        return []
    return [value for value in values if isinstance(value, str)]


def search_is_available(observation: str) -> bool:
    matches = re.findall(
        r"搜索功能是否可用:\s*(True|False)", observation or "", re.IGNORECASE
    )
    return bool(matches and matches[-1].casefold() == "true")


def canonical_button(target: str, observation: str) -> Optional[str]:
    """Return the exact current-page spelling for a case-insensitive target."""
    folded = str(target).strip().casefold()
    for button in clickable_buttons(observation):
        if button.strip().casefold() == folded:
            return button
    return None


def safe_json_arguments(raw_arguments: Any) -> Dict[str, Any]:
    if isinstance(raw_arguments, dict):
        value = raw_arguments
    elif isinstance(raw_arguments, str):
        try:
            value = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool arguments 不是合法 JSON: {exc}") from exc
    else:
        raise ValueError("tool arguments 必须是 JSON object")
    if not isinstance(value, dict):
        raise ValueError("tool arguments 必须是 JSON object")
    return value


def message_size(message: Dict[str, Any]) -> int:
    return len(json.dumps(message, ensure_ascii=False, default=str))


class Agent:
    """A serial tool-calling shopping agent with local action validation."""

    def __init__(self, task_id: int, config: Dict[str, Any]) -> None:
        self.task_id = task_id
        self.config = config
        self.model_name = config["model_name"]
        self.model_key = config.get("model_key") or "EMPTY"
        self.base_url = config.get("base_url")
        if not self.base_url:
            raise ValueError("agent_config.base_url 未设置")

        self.system_prompt = config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        self.max_tokens = int(config.get("max_tokens", 256))
        self.temperature = float(config.get("temperature", 0.0))
        self.max_api_retries = int(config.get("max_api_retries", 3))
        self.api_timeout = float(config.get("api_timeout", 120.0))
        self.max_steps = int(config.get("max_steps", 40))
        self.max_blocked_calls = int(config.get("max_blocked_calls", 3))
        self.tool_choice = config.get("tool_choice", "required")
        self.enable_thinking = bool(config.get("enable_thinking", False))
        self.max_context_chars = int(config.get("max_context_chars", 60000))
        self.keep_recent_observations = int(
            config.get("keep_recent_observations", 4)
        )

        self.client = OpenAI(
            api_key=self.model_key,
            base_url=self.base_url,
            timeout=self.api_timeout,
            max_retries=0,
        )
        self.shop_env: Optional[ShopEnv] = None
        self.messages: List[Dict[str, Any]] = []
        self.action_log: List[Dict[str, Any]] = []
        self.generation_log: List[Dict[str, Any]] = []
        self.latest_observation = ""
        self.env_steps = 0
        self.blocked_calls = 0
        self.saved = False
        self.task_complete = False
        self.env_idx: Optional[int] = None
        self.user_persona: Optional[Dict[str, Any]] = None
        self._seen_state_actions: set[Tuple[str, str]] = set()

    def set_shop_env(self, shop_env: ShopEnv) -> None:
        self.shop_env = shop_env

    def reset(self) -> str:
        if self.shop_env is None:
            raise AttributeError("self.shop_env not initialized")

        env_result = self.shop_env.reset(self.task_id)
        self.env_idx = self.shop_env.env_idx
        observation = (
            env_result["instruction"]
            + "\n\n搜索功能是否可用: True\n\n可点击的按钮: []"
        )

        system_content = self.system_prompt
        if self.shop_env.if_persona and "user_persona" in env_result:
            self.user_persona = env_result["user_persona"]
            system_content += "\n\n用户个人文档：" + json.dumps(
                self.user_persona, ensure_ascii=False
            )

        self.messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": observation},
        ]
        self.latest_observation = observation
        self.action_log = []
        self.generation_log = []
        self.env_steps = 0
        self.blocked_calls = 0
        self.saved = False
        self.task_complete = False
        self._seen_state_actions.clear()
        return observation

    def _messages_for_api(self) -> List[Dict[str, Any]]:
        """Compact old tool observations while preserving tool-call pairing."""
        projected = copy.deepcopy(self.messages)
        if sum(message_size(item) for item in projected) <= self.max_context_chars:
            return projected

        tool_indexes = [
            i for i, item in enumerate(projected) if item.get("role") == "tool"
        ]
        protected = set(tool_indexes[-self.keep_recent_observations :])
        for index in tool_indexes:
            if index not in protected:
                projected[index]["content"] = (
                    "[旧页面 observation 已压缩。不要使用历史页面目标，"
                    "仅依据最新 observation 调用工具。]"
                )
        return projected

    def _call_model(self) -> Tuple[Dict[str, Any], Optional[str]]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_api_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self._messages_for_api(),
                    tools=SHOP_TOOLS,
                    tool_choice=self.tool_choice,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": self.enable_thinking
                        }
                    },
                )
                choice = completion.choices[0]
                message = choice.message
                tool_calls = list(message.tool_calls or [])

                # Keep exactly one call. This also keeps the assistant/tool
                # message pair valid for the next OpenAI request.
                serialized_calls: List[Dict[str, Any]] = []
                if tool_calls:
                    call = tool_calls[0]
                    serialized_calls.append(
                        {
                            "id": call.id or f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments or "{}",
                            },
                        }
                    )

                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content or "",
                }
                if serialized_calls:
                    assistant_message["tool_calls"] = serialized_calls

                reasoning = getattr(message, "reasoning_content", None)
                finish_reason = getattr(choice, "finish_reason", None)
                self.generation_log.append(
                    {
                        "attempt": attempt,
                        "finish_reason": finish_reason,
                        "content": message.content,
                        "reasoning_content": reasoning,
                        "tool_call_count": len(tool_calls),
                    }
                )
                return assistant_message, finish_reason
            except Exception as exc:  # API/network errors only; bounded retries.
                last_error = exc
                print(
                    f"LLM call failed (task {self.task_id}, "
                    f"attempt {attempt}/{self.max_api_retries}): {exc}"
                )
                if attempt < self.max_api_retries:
                    time.sleep(min(2 ** (attempt - 1), 4))

        raise AgentTermination("llm_error", f"LLM 调用失败: {last_error}")

    def _tool_error_message(
        self, call_id: str, name: str, reason: str
    ) -> Dict[str, Any]:
        buttons = clickable_buttons(self.latest_observation)
        allowed = json.dumps(buttons[:30], ensure_ascii=False)
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": (
                f"工具调用被本地动作守卫拒绝，未发送给购物环境。原因: {reason}。"
                f"最新页面搜索可用: {search_is_available(self.latest_observation)}；"
                f"最新页面可点击按钮: {allowed}。"
                "请只调用一个合法工具，不要重复历史页面动作。"
            ),
        }

    def _action_from_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if name not in TOOL_ARGUMENTS:
            raise ValueError(f"未知工具: {name}")

        extra = sorted(set(arguments) - TOOL_ARGUMENTS[name])
        if extra:
            raise ValueError(f"存在 schema 外参数: {', '.join(extra)}")

        if name == "search_products":
            if not search_is_available(self.latest_observation):
                raise ValueError("当前页面不可搜索")
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query 不能为空")
            query = re.sub(r"[\r\n\[\]]+", " ", query).strip()
            if not query:
                raise ValueError("query 清理后为空")
            return f"search[{query[:200]}]"

        if name == "open_product":
            asin = arguments.get("asin")
            if not isinstance(asin, str) or not asin.strip():
                raise ValueError("asin 不能为空")
            target = canonical_button(asin, self.latest_observation)
            if target is None:
                raise ValueError("asin 不在最新页面的可点击按钮中")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", target):
                raise ValueError("目标不像有效商品 ASIN")
            action = f"click[{target}]"

        elif name == "select_option":
            value = arguments.get("value")
            if not isinstance(value, str) or not value.strip():
                raise ValueError("value 不能为空")
            target = canonical_button(value, self.latest_observation)
            if target is None:
                raise ValueError("规格值不在最新页面的可点击按钮中")
            if target.strip().casefold() in NAVIGATION_TARGETS:
                raise ValueError("select_option 不能用于导航按钮")
            action = f"click[{target}]"

        else:
            target = canonical_button(
                FIXED_CLICK_TARGETS[name], self.latest_observation
            )
            if target is None:
                raise ValueError(
                    f"最新页面没有 {FIXED_CLICK_TARGETS[name]!r} 按钮"
                )
            action = f"click[{target}]"

        state_hash = hashlib.sha1(
            self.latest_observation.encode("utf-8")
        ).hexdigest()
        signature = (state_hash, action.casefold())
        if signature in self._seen_state_actions:
            raise ValueError("相同页面上的相同动作已经执行过")
        return action

    def _next_valid_tool(self) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        while self.blocked_calls < self.max_blocked_calls:
            assistant_message, finish_reason = self._call_model()
            tool_calls = assistant_message.get("tool_calls", [])
            self.messages.append(assistant_message)

            if not tool_calls:
                self.blocked_calls += 1
                suffix = "，且输出因长度限制被截断" if finish_reason == "length" else ""
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"上一回复没有结构化工具调用{suffix}。"
                            "不要输出文字；现在只调用一个可用工具。"
                        ),
                    }
                )
                continue

            call = tool_calls[0]
            name = str(call.get("function", {}).get("name", ""))
            call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:12]}")
            try:
                arguments = safe_json_arguments(
                    call.get("function", {}).get("arguments", "{}")
                )
                action = self._action_from_tool(name, arguments)
                return call, arguments, action
            except Exception as exc:
                self.blocked_calls += 1
                self.messages.append(
                    self._tool_error_message(call_id, name, str(exc))
                )

        raise AgentTermination(
            "invalid_action_limit",
            f"连续 {self.max_blocked_calls} 次未产生可执行工具调用",
        )

    def act(self) -> Tuple[bool, str]:
        if self.shop_env is None:
            raise AttributeError("Agent 未连接 ShopEnv")
        if self.env_steps >= self.max_steps:
            raise AgentTermination(
                "max_steps", f"达到最大环境步数 {self.max_steps}"
            )

        call, arguments, action = self._next_valid_tool()
        call_id = str(call["id"])
        name = str(call["function"]["name"])
        observation_before = self.latest_observation
        state_hash = hashlib.sha1(observation_before.encode("utf-8")).hexdigest()

        env_response = self.shop_env.interact(action)
        self.env_steps += 1
        if "error" in env_response:
            raise AgentTermination(
                "environment_error", f"环境报错: {env_response['error']}"
            )

        observation = env_response.get("instruction", "")
        self.latest_observation = observation
        self._seen_state_actions.add((state_hash, action.casefold()))
        self.blocked_calls = 0
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": observation,
            }
        )
        self.action_log.append(
            {
                "step": self.env_steps,
                "tool": name,
                "arguments": arguments,
                "environment_action": action,
                "done": bool(env_response.get("done", False)),
                "over": bool(env_response.get("over", False)),
            }
        )

        done = bool(env_response.get("done", False))
        over = bool(env_response.get("over", False))
        if done or over:
            status = "completed" if done else "environment_limit"
            self.save_result(
                status=status,
                reward=float(env_response.get("reward", 0) or 0),
                reward_detail=env_response.get("reward_detail", {}),
                goal=env_response.get("goal", {}),
                purchase=env_response.get("purchase", {}),
                error=None if done else "ShopSimulator 达到内部历史长度限制",
            )
            self.task_complete = True

        return self.task_complete, observation

    def save_result(
        self,
        status: str,
        reward: float = 0.0,
        reward_detail: Optional[Dict[str, Any]] = None,
        goal: Optional[Dict[str, Any]] = None,
        purchase: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        if self.saved:
            return
        log_data = {
            "task_id": self.task_id,
            "status": status,
            "reward": reward,
            "reward_detail": reward_detail or {},
            "goal": goal or {},
            "purchase": purchase or {},
            "error": error,
            "env_steps": self.env_steps,
            "actions": self.action_log,
            "conversation": self.messages,
            "generation_log": self.generation_log,
        }

        output_dir = os.path.join(
            self.config["output_path"], self.config["model_name"]
        )
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{self.task_id}.json")
        temp_name = filename + ".tmp"
        with open(temp_name, "w", encoding="utf-8") as handle:
            json.dump(log_data, handle, ensure_ascii=False, indent=2)
        os.replace(temp_name, filename)
        self.saved = True
        print(f"[LOG] task={self.task_id} status={status} saved={filename}")


def get_finished_tasks(output_dir: str) -> List[int]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        return []
    result: List[int] = []
    for filename in os.listdir(output_dir):
        if not filename.endswith(".json"):
            continue
        try:
            result.append(int(filename.rsplit(".", 1)[0]))
        except ValueError:
            print(f"Warning: 无法从 {filename!r} 解析 task id，已跳过")
    return sorted(result)


def parse_task_ids(spec: str) -> List[int]:
    """Parse forms such as ``0,2,5-9``."""
    values: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"非法 task id 范围: {part}")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    return sorted(values)


def run_task(task_id: int, config: Dict[str, Any]) -> bool:
    agent: Optional[Agent] = None
    shop_env: Optional[ShopEnv] = None
    try:
        shop_env = ShopEnv(config["env_config"])
        agent = Agent(task_id=task_id, config=config["agent_config"])
        agent.set_shop_env(shop_env)
        agent.reset()
        while not agent.task_complete:
            agent.act()
        return True
    except AgentTermination as exc:
        print(f"Task {task_id} stopped [{exc.status}]: {exc}")
        if agent is not None:
            agent.save_result(status=exc.status, error=str(exc))
        return False
    except Exception as exc:
        print(f"Error (task {task_id}): {exc}")
        traceback.print_exc()
        if agent is not None:
            agent.save_result(status="exception", error=traceback.format_exc())
        return False
    finally:
        if shop_env is not None and shop_env.env_idx is not None:
            try:
                shop_env.release()
            except Exception as exc:
                print(f"Warning: task {task_id} 释放环境失败: {exc}")


def run_tasks_singlethreaded(task_ids: Sequence[int], config: Dict[str, Any]) -> None:
    for task_id in tqdm(task_ids, desc="执行任务"):
        print(f"**Begin task {task_id}**")
        run_task(task_id, config)


def run_tasks_multithreaded(
    task_ids: Sequence[int], config: Dict[str, Any], max_workers: int
) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_task, task_id, config): task_id
            for task_id in task_ids
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="执行任务"
        ):
            task_id = futures[future]
            try:
                ok = future.result()
                print(f"**Task {task_id} {'completed' if ok else 'stopped'}**")
            except Exception as exc:
                print(f"**Task {task_id} failed**: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ShopSimulator with structured tool calls."
    )
    parser.add_argument("--yaml_name", required=True, help="YAML config path")
    parser.add_argument("--multithread", action="store_true")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run selected tasks even when an existing JSON result is present.",
    )
    parser.add_argument(
        "--task_ids",
        default=None,
        help="Optional task selection, for example: 0,2,5-9",
    )
    args = parser.parse_args()

    with open(args.yaml_name, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "agent_config" not in config:
        raise ValueError("YAML 缺少 agent_config")
    if "env_config" not in config:
        raise ValueError("YAML 缺少 env_config")

    agent_config = config["agent_config"]
    if args.task_ids:
        all_tasks = parse_task_ids(args.task_ids)
    else:
        task_start = int(agent_config.get("task_start", 0))
        task_nums = int(agent_config.get("task_nums", 0))
        all_tasks = list(range(task_start, task_start + task_nums))

    output_dir = os.path.join(
        agent_config["output_path"], agent_config["model_name"]
    )
    finished = set(get_finished_tasks(output_dir))
    todo = all_tasks if args.overwrite else [i for i in all_tasks if i not in finished]
    print(
        f"Selected: {len(all_tasks)}, Existing: {len(set(all_tasks) & finished)}, "
        f"Pending: {len(todo)}, Overwrite: {args.overwrite}"
    )

    if args.multithread:
        print(f"Using multithreaded mode, max workers: {args.max_workers}")
        run_tasks_multithreaded(todo, config, args.max_workers)
    else:
        print("Using single-threaded mode")
        run_tasks_singlethreaded(todo, config)


if __name__ == "__main__":
    main()
