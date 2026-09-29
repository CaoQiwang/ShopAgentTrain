"""并发运行 ShopSimulator 单轮评测并保存每个任务的轨迹。"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from ..core.agent_loop import AgentLoop
from ..env.client import DEFAULT_BASE_URL, ShopEnv


DEFAULT_TASK_COUNT = 1495
DEFAULT_WORKERS = 4
DEFAULT_OUTPUT_DIR = Path("output")


def parse_task_ids(spec: str) -> List[int]:
    """解析 ``0,2,5-9`` 形式的任务 ID。"""

    task_ids = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"非法任务范围: {part!r}")
            task_ids.update(range(start, end + 1))
        else:
            task_id = int(part)
            if task_id < 0:
                raise ValueError("任务 ID 不能为负数")
            task_ids.add(task_id)
    if not task_ids:
        raise ValueError("没有解析到任务 ID")
    return sorted(task_ids)


def safe_model_name(model: str) -> str:
    """生成可在 Windows 和 Unix 上作为目录名使用的模型名称。"""

    name = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "__", model).strip(" .")
    return name or "unnamed-model"


def result_path(output_root: Path, model: str, task_id: int) -> Path:
    return output_root / safe_model_name(model) / f"{task_id}.json"


def save_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    """先写临时文件再替换目标，避免中断后留下半个 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, default=str)
    os.replace(temporary, path)


def finished_task_ids(output_root: Path, model: str) -> set[int]:
    directory = output_root / safe_model_name(model)
    if not directory.is_dir():
        return set()
    finished = set()
    for path in directory.glob("*.json"):
        try:
            finished.add(int(path.stem))
        except ValueError:
            continue
    return finished


def exception_task_ids(
    output_root: Path, model: str, task_ids: Iterable[int]
) -> set[int]:
    """找出指定任务中已有结果为 exception 的任务。"""

    exceptions = set()
    for task_id in task_ids:
        path = result_path(output_root, model, task_id)
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(result, dict) and result.get("status") == "exception":
            exceptions.add(task_id)
    return exceptions


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exception_result(
    task_id: int,
    model: str,
    started_at: str,
    started_time: float,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "model": model,
        "status": "exception",
        "steps": 0,
        "reward": 0.0,
        "reward_detail": {},
        "purchase": {},
        "goal": {},
        "trajectory": [],
        "messages": [],
        "error": traceback.format_exc(),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(time.monotonic() - started_time, 3),
    }


def run_one_task(
    task_id: int,
    *,
    client: Any,
    model: str,
    output_root: Path,
    env_factory: Callable[[], ShopEnv],
    max_steps: int,
    temperature: float,
    max_tokens: int,
    model_options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """创建独立 ShopEnv，运行并保存一个任务；异常也会保存为结果文件。"""

    started_at = _utc_now()
    started_time = time.monotonic()
    env: Optional[ShopEnv] = None
    try:
        env = env_factory()
        loop = AgentLoop(
            client,
            model,
            env=env,
            max_steps=max_steps,
            temperature=temperature,
            max_tokens=max_tokens,
            model_options=model_options,
        )
        result = loop.run(task_id)
        result.update(
            model=model,
            started_at=started_at,
            finished_at=_utc_now(),
            duration_seconds=round(time.monotonic() - started_time, 3),
            error=None,
        )
    except Exception:
        result = _exception_result(task_id, model, started_at, started_time)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                result["cleanup_error"] = str(exc)

    path = result_path(output_root, model, task_id)
    save_json_atomic(path, result)
    return {
        "task_id": task_id,
        "status": result["status"],
        "reward": result.get("reward", 0.0),
        "path": str(path),
    }


def run_rollouts(
    *,
    client: Any,
    model: str,
    task_ids: Iterable[int],
    output_root: Path = DEFAULT_OUTPUT_DIR,
    env_base_url: str = DEFAULT_BASE_URL,
    workers: int = DEFAULT_WORKERS,
    max_steps: int = 40,
    temperature: float = 0.0,
    max_tokens: int = 512,
    env_timeout: float = 60.0,
    persona: bool = False,
    overwrite: bool = False,
    retry_exceptions: bool = False,
    model_options: Optional[Mapping[str, Any]] = None,
    env_factory: Optional[Callable[[], ShopEnv]] = None,
) -> Dict[str, Any]:
    """并发运行一组任务，返回批次汇总。

    每个正在运行的 worker 都持有独立的 :class:`ShopEnv`，因此 ``workers`` 就是
    同时使用的环境实例上限。ShopSimulator 默认只有 20 个服务端槽位，不应把该值
    设置得超过服务端容量。
    """

    if workers <= 0:
        raise ValueError("workers 必须大于 0")
    if max_steps <= 0:
        raise ValueError("max_steps 必须大于 0")

    selected = sorted(set(int(task_id) for task_id in task_ids))
    if any(task_id < 0 for task_id in selected):
        raise ValueError("任务 ID 不能为负数")

    output_root = Path(output_root)
    if retry_exceptions:
        exceptions = exception_task_ids(output_root, model, selected)
        pending = [item for item in selected if item in exceptions]
    else:
        existing = finished_task_ids(output_root, model)
        pending = selected if overwrite else [item for item in selected if item not in existing]
    skipped = len(selected) - len(pending)

    if env_factory is None:
        env_factory = lambda: ShopEnv(
            base_url=env_base_url,
            timeout=env_timeout,
            persona=persona,
        )

    if retry_exceptions:
        print(
            f"模型={model} 选中={len(selected)} 待重试 exception={len(pending)} "
            f"跳过其他={skipped} 并发环境={workers}"
        )
    else:
        print(
            f"模型={model} 选中={len(selected)} 已存在={skipped} "
            f"待运行={len(pending)} 并发环境={workers}"
        )

    counts: Dict[str, int] = {}
    reward_sum = 0.0
    completed = 0
    started_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: Dict[Future[Dict[str, Any]], int] = {
            executor.submit(
                run_one_task,
                task_id,
                client=client,
                model=model,
                output_root=output_root,
                env_factory=env_factory,
                max_steps=max_steps,
                temperature=temperature,
                max_tokens=max_tokens,
                model_options=model_options,
            ): task_id
            for task_id in pending
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                # run_one_task 已处理正常轨迹异常；这里只防止保存阶段等外围异常。
                item = {
                    "task_id": task_id,
                    "status": "runner_exception",
                    "reward": 0.0,
                    "path": "",
                }
                print(f"[任务 {task_id}] runner 异常: {exc}")

            completed += 1
            status = str(item["status"])
            counts[status] = counts.get(status, 0) + 1
            reward_sum += float(item.get("reward", 0) or 0)
            print(
                f"[{completed}/{len(pending)}] task={task_id} "
                f"status={status} reward={item.get('reward', 0)}"
            )

    summary = {
        "model": model,
        "model_output_directory": str(output_root / safe_model_name(model)),
        "selected": len(selected),
        "skipped_existing": 0 if retry_exceptions else skipped,
        "skipped_not_exception": skipped if retry_exceptions else 0,
        "retry_exceptions_only": retry_exceptions,
        "executed": len(pending),
        "status_counts": counts,
        "reward_sum": reward_sum,
        "average_reward": reward_sum / len(pending) if pending else 0.0,
        "duration_seconds": round(time.monotonic() - started_time, 3),
        "finished_at": _utc_now(),
    }
    save_json_atomic(
        output_root / safe_model_name(model) / "summary.json", summary
    )
    return summary


def create_openai_client(
    *,
    api_key: str,
    base_url: Optional[str],
    timeout: float,
) -> Any:
    """延迟导入 OpenAI SDK，使辅助函数在未安装 SDK 时仍可被测试。"""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai 包，请先执行: pip install openai") from exc

    options: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout,
    }
    if base_url:
        options["base_url"] = base_url
    return OpenAI(**options)


def parse_json_object(value: str) -> Dict[str, Any]:
    """解析命令行中的 JSON object。"""

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"不是合法 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("必须是 JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量评测 ShopSimulator Agent")
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument(
        "--model-base-url",
        default=os.getenv("OPENAI_BASE_URL"),
        help="OpenAI 兼容模型服务地址；默认读取 OPENAI_BASE_URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="模型 API key；默认读取 OPENAI_API_KEY",
    )
    parser.add_argument(
        "--env-base-url",
        default=DEFAULT_BASE_URL,
        help="ShopSimulator 服务根地址",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=DEFAULT_TASK_COUNT)
    parser.add_argument(
        "--task-ids",
        help="覆盖 start/count，例如 0,2,5-9",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--model-timeout", type=float, default=120.0)
    parser.add_argument("--env-timeout", type=float, default=60.0)
    parser.add_argument(
        "--model-extra-body",
        type=parse_json_object,
        help="传给 OpenAI SDK extra_body 的 JSON object",
    )
    parser.add_argument("--persona", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retry-exceptions",
        action="store_true",
        help="只重跑已有 status=exception 的任务并覆盖其结果",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.task_ids:
        task_ids = parse_task_ids(args.task_ids)
    else:
        if args.task_start < 0 or args.task_count <= 0:
            raise ValueError("task-start 必须非负，task-count 必须大于 0")
        task_ids = range(args.task_start, args.task_start + args.task_count)

    client = create_openai_client(
        api_key=args.api_key,
        base_url=args.model_base_url,
        timeout=args.model_timeout,
    )
    summary = run_rollouts(
        client=client,
        model=args.model,
        task_ids=task_ids,
        output_root=args.output_dir,
        env_base_url=args.env_base_url,
        workers=args.workers,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        env_timeout=args.env_timeout,
        persona=args.persona,
        overwrite=args.overwrite,
        retry_exceptions=args.retry_exceptions,
        model_options=(
            {"extra_body": args.model_extra_body}
            if args.model_extra_body is not None
            else None
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failed = sum(
        count
        for status, count in summary["status_counts"].items()
        if status in {"exception", "runner_exception"}
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_TASK_COUNT",
    "finished_task_ids",
    "parse_task_ids",
    "parse_json_object",
    "result_path",
    "run_one_task",
    "run_rollouts",
    "safe_model_name",
]
