#!/usr/bin/env python3
"""对 GRPO 任务重复 rollout，只保存每个任务的奖励组。

默认读取 ``data/shop_grpo_tasks_10000.jsonl``，使用本地 OpenAI 兼容服务
``http://127.0.0.1:8061/v1`` 上的 ``qwen35-4b-sft-ckpt300``，对每个
``task_id >= 10000`` 的任务运行 4 次。输出每行形如：

    {"task_id":10000,"rewards":[1.0,1.0,0.0,1.0]}

脚本直接复用 ``run_rollout.sh`` 所调用的 AgentLoop 和模拟器原始 reward；模型
消息、工具调用、observation 等轨迹信息只存在于内存，不会写入输出文件。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = PROJECT_ROOT / "data/shop_grpo_tasks_10000.jsonl"
DEFAULT_MODEL = "qwen35-4b-sft-ckpt300"
DEFAULT_MODEL_BASE_URL = "http://127.0.0.1:8061/v1"
DEFAULT_ENV_BASE_URL = "http://127.0.0.1:5000"
SIMULATOR_REWARD = "simulator"
CUSTOM_REWARD = "custom"


def reward_from_agent_result(result: Mapping[str, Any], reward_mode: str) -> float:
    """从同一个 AgentLoop 结果选择原始或自定义终局奖励。"""

    if reward_mode == SIMULATOR_REWARD:
        return float(result.get("reward", 0.0) or 0.0)
    if reward_mode == CUSTOM_REWARD:
        from src.env.result_reward import exact_match_result_reward

        # AgentLoop 用 status 表示购买成功，不在最终结果中保留 env 的 done 字段。
        # 必须显式恢复 done，否则 exact_match_result_reward 会把所有结果判为 0。
        return exact_match_result_reward(
            {
                "done": result.get("status") == "purchased",
                "reward_detail": result.get("reward_detail", {}),
            }
        )
    raise ValueError(f"未知奖励口径：{reward_mode!r}")


async def run_reward_rollout(
    task_id: int,
    *,
    client: Any,
    model: str,
    env_base_url: str,
    env_timeout: float,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    reward_mode: str = SIMULATOR_REWARD,
    model_extra_body: Optional[Mapping[str, Any]] = None,
    diagnose_zero: bool = False,
    loop_factory: Optional[Callable[[], Any]] = None,
) -> float:
    """按 run_rollout.sh 的执行路径运行轨迹并返回模拟器原始 reward。"""

    # 延迟导入，保证 --help 和数据辅助函数不依赖环境运行时。
    from src.core.agent_loop import AgentLoop
    from src.env.client import ShopEnv

    env: Any = None
    if loop_factory is None:
        env = ShopEnv(base_url=env_base_url, timeout=env_timeout)
        loop = AgentLoop(
            client,
            model,
            env=env,
            max_steps=max_turns,
            temperature=temperature,
            max_tokens=max_tokens,
            model_options=(
                {"extra_body": dict(model_extra_body)}
                if model_extra_body is not None
                else None
            ),
        )
    else:
        loop = loop_factory()
        env = getattr(loop, "env", None)
    try:
        result = await asyncio.to_thread(loop.run, task_id)
        reward = reward_from_agent_result(result, reward_mode)
        if diagnose_zero and reward == 0.0:
            print(
                "[zero-diagnostic] "
                f"task={task_id} status={result.get('status')} "
                f"steps={result.get('steps')} "
                f"simulator_reward={float(result.get('reward', 0.0) or 0.0)} "
                f"selected_reward={reward} reward_detail="
                f"{json.dumps(result.get('reward_detail', {}), ensure_ascii=False)}"
            )
        return reward
    finally:
        if env is not None:
            await asyncio.to_thread(env.close)


def load_task_ids(path: Path, min_task_id: int) -> list[int]:
    """从 GRPO JSONL 读取并验证不重复的 task_id。"""

    task_ids: list[int] = []
    seen: set[int] = set()
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"无法读取输入文件 {path}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task_id = row["env_config"]["task_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: 缺少合法的 env_config.task_id") from exc
            if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
                raise ValueError(f"{path}:{line_number}: task_id 必须是非负整数")
            if task_id < min_task_id:
                continue
            if task_id in seen:
                raise ValueError(f"{path}:{line_number}: task_id={task_id} 重复")
            seen.add(task_id)
            task_ids.append(task_id)
    if not task_ids:
        raise ValueError(f"输入文件中没有 task_id >= {min_task_id} 的任务")
    return task_ids


def load_completed(path: Path, rollouts: int) -> dict[int, list[float]]:
    """读取可续跑结果；忽略可能由中断造成的最后一条残缺 JSON。"""

    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    completed: dict[int, list[float]] = {}
    nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty = nonempty_indexes[-1] if nonempty_indexes else -1
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == last_nonempty:
                print(f"警告：忽略输出文件末尾的残缺行：{path}:{index + 1}")
                continue
            raise ValueError(f"{path}:{index + 1}: 不是合法 JSON") from exc
        if not isinstance(row, dict) or set(row) != {"task_id", "rewards"}:
            raise ValueError(
                f"{path}:{index + 1}: 每行只能包含 task_id 和 rewards"
            )
        task_id = row["task_id"]
        rewards = row["rewards"]
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise ValueError(f"{path}:{index + 1}: task_id 必须是整数")
        if not isinstance(rewards, list) or len(rewards) != rollouts or not all(
            type(reward) in (int, float) and math.isfinite(float(reward))
            for reward in rewards
        ):
            raise ValueError(
                f"{path}:{index + 1}: rewards 必须包含 {rollouts} 个数值"
            )
        if task_id in completed:
            raise ValueError(f"{path}:{index + 1}: task_id={task_id} 重复")
        completed[task_id] = [float(reward) for reward in rewards]
    return completed


def rewrite_results(path: Path, results: Mapping[int, Sequence[float]]) -> None:
    """原子整理已有结果，也会清除恢复时忽略的末尾残行。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for task_id in sorted(results):
                row = {"task_id": task_id, "rewards": list(results[task_id])}
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_result(stream: Any, task_id: int, rewards: Sequence[float]) -> None:
    row = {"task_id": task_id, "rewards": list(rewards)}
    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())


def create_openai_client(api_key: str, base_url: str, timeout: float) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai 包，请先执行: pip install openai") from exc
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def run_task_group(
    task_id: int,
    *,
    rollouts: int,
    retries: int,
    retry_delay: float,
    rollout_kwargs: Mapping[str, Any],
) -> list[float]:
    """同一任务顺序运行多个独立环境；失败时只重试当前 rollout。"""

    rewards: list[float] = []
    for rollout_index in range(rollouts):
        for attempt in range(retries + 1):
            try:
                reward = asyncio.run(run_reward_rollout(task_id, **rollout_kwargs))
                rewards.append(float(reward))
                break
            except Exception:
                if attempt >= retries:
                    raise
                time.sleep(retry_delay * (2 ** attempt))
    return rewards


def parse_json_object(value: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"不是合法 JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("必须是 JSON object")
    return result


def build_parser(reward_mode: str = SIMULATOR_REWARD) -> argparse.ArgumentParser:
    if reward_mode == SIMULATOR_REWARD:
        description = __doc__
        output_help = "默认：data/<model>_rollout_rewards_<rollouts>.jsonl"
    elif reward_mode == CUSTOM_REWARD:
        description = (
            "对 GRPO 任务重复 rollout，只保存 exact_match_result_reward "
            "计算出的自定义奖励组。"
        )
        output_help = "默认：data/<model>_custom_rollout_rewards_<rollouts>.jsonl"
    else:
        raise ValueError(f"未知奖励口径：{reward_mode!r}")

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        help=output_help,
    )
    parser.add_argument("--min-task-id", type=int, default=10000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-base-url", default=DEFAULT_MODEL_BASE_URL)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--env-base-url", default=DEFAULT_ENV_BASE_URL)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--model-timeout", type=float, default=120.0)
    parser.add_argument("--env-timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--limit", type=int, help="仅处理前 N 个待运行任务，用于试跑")
    parser.add_argument("--model-extra-body", type=parse_json_object)
    parser.add_argument(
        "--diagnose-zero",
        action="store_true",
        help="零分时在终端打印 status、步数和 reward_detail；仍不保存轨迹",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="清空当前输出并重跑；奖励口径或生成参数变化后应使用",
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    reward_mode: str = SIMULATOR_REWARD,
) -> int:
    parser = build_parser(reward_mode)
    args = parser.parse_args(argv)
    if args.min_task_id < 0:
        parser.error("min-task-id 必须非负")
    if args.rollouts <= 0 or args.workers <= 0 or args.max_turns <= 0:
        parser.error("rollouts、workers 和 max-turns 必须大于 0")
    if args.max_tokens <= 0 or args.retries < 0 or args.retry_delay < 0:
        parser.error("max-tokens 必须大于 0；retries 和 retry-delay 必须非负")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit 必须大于 0")
    if args.output is None:
        reward_name = "" if reward_mode == SIMULATOR_REWARD else "custom_"
        args.output = (
            PROJECT_ROOT
            / f"data/{args.model}_{reward_name}rollout_rewards_{args.rollouts}.jsonl"
        )
    if args.input.resolve() == args.output.resolve():
        parser.error("输出文件不能覆盖输入文件")

    task_ids = load_task_ids(args.input, args.min_task_id)
    completed = {} if args.overwrite else load_completed(args.output, args.rollouts)
    completed_before_run = len(completed)
    unknown = sorted(set(completed) - set(task_ids))
    if unknown:
        parser.error(f"输出文件包含不在当前输入范围内的 task_id：{unknown[:5]}")

    # 启动前先原子重写一次，修复上次中断可能留下的残行并统一数值格式。
    rewrite_results(args.output, completed)
    pending = [task_id for task_id in task_ids if task_id not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]

    reward_label = (
        "ShopSimulator原始reward"
        if reward_mode == SIMULATOR_REWARD
        else "exact_match_result_reward自定义reward"
    )
    print(
        f"模型={args.model} 服务={args.model_base_url} 任务总数={len(task_ids)} "
        f"已完成={len(completed)} 本次待运行={len(pending)} 每任务={args.rollouts} 次 "
        f"并发任务={args.workers} 温度={args.temperature} max_tokens={args.max_tokens} "
        f"奖励口径={reward_label}"
    )
    if not pending:
        print(f"没有待运行任务；输出：{args.output}")
        return 0

    client = create_openai_client(args.api_key, args.model_base_url, args.model_timeout)
    rollout_kwargs = {
        "client": client,
        "model": args.model,
        "env_base_url": args.env_base_url,
        "env_timeout": args.env_timeout,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reward_mode": reward_mode,
        "model_extra_body": args.model_extra_body,
        "diagnose_zero": args.diagnose_zero,
    }

    failures: list[tuple[int, str]] = []
    write_lock = threading.Lock()
    started = time.monotonic()
    interrupted = False
    with args.output.open("a", encoding="utf-8", newline="\n") as output_stream:
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures: dict[Future[list[float]], int] = {}
        try:
            futures = {
                executor.submit(
                    run_task_group,
                    task_id,
                    rollouts=args.rollouts,
                    retries=args.retries,
                    retry_delay=args.retry_delay,
                    rollout_kwargs=rollout_kwargs,
                ): task_id
                for task_id in pending
            }
            for done_count, future in enumerate(as_completed(futures), 1):
                task_id = futures[future]
                try:
                    rewards = future.result()
                except Exception as exc:
                    failures.append((task_id, f"{type(exc).__name__}: {exc}"))
                    print(f"[{done_count}/{len(pending)}] task={task_id} 失败：{failures[-1][1]}")
                    continue
                with write_lock:
                    append_result(output_stream, task_id, rewards)
                    completed[task_id] = rewards
                print(f"[{done_count}/{len(pending)}] task={task_id} rewards={rewards}")
        except KeyboardInterrupt:
            interrupted = True
            for future in futures:
                future.cancel()
            print("收到中断，正在等待已开始的任务释放环境；已写入结果可直接续跑。")
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    rewrite_results(args.output, completed)
    elapsed = time.monotonic() - started
    completed_this_run = len(completed) - completed_before_run
    print(
        f"本次完成={completed_this_run} 失败={len(failures)} "
        f"累计结果={len(completed)}/{len(task_ids)} 用时={elapsed:.1f}s 输出={args.output}"
    )
    if interrupted:
        return 130
    if failures:
        print("失败任务未写入输出；用相同命令重启会自动补跑。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CUSTOM_REWARD",
    "SIMULATOR_REWARD",
    "load_completed",
    "load_task_ids",
    "reward_from_agent_result",
    "run_reward_rollout",
    "run_task_group",
]
