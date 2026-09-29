"""只运行 data/final_test_100.jsonl 中任务的独立最终测试。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .rollout import (
    create_openai_client,
    parse_json_object,
    result_path,
    run_rollouts,
    safe_model_name,
)
from ..env.client import DEFAULT_BASE_URL


DEFAULT_DATASET = Path("data/final_test_100.jsonl")
DEFAULT_OUTPUT_DIR = Path("output/final_test_100")
EXPECTED_TASK_COUNT = 100


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _query_from_record(record: Mapping[str, Any], line_number: int) -> str:
    instructions = record.get("instructions")
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(f"第 {line_number} 行缺少非空 instructions")
    instruction = instructions[0]
    if not isinstance(instruction, Mapping):
        raise ValueError(f"第 {line_number} 行 instructions[0] 不是 object")
    query = instruction.get("instruction")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"第 {line_number} 行缺少有效用户 query")
    return query.strip()


def load_final_test(
    path: Path,
    *,
    expected_count: int = EXPECTED_TASK_COUNT,
) -> List[Tuple[int, str]]:
    """读取最终测试集，返回保持文件顺序的 ``(task_id, query)``。"""

    if expected_count <= 0:
        raise ValueError("expected_count 必须大于 0")
    if not path.is_file():
        raise FileNotFoundError(f"最终测试集不存在：{path}")

    tasks: List[Tuple[int, str]] = []
    seen: set[int] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"最终测试集第 {line_number} 行不是合法 JSON：{exc}"
                ) from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"最终测试集第 {line_number} 行不是 object")

            value = record.get("task_id")
            if isinstance(value, bool):
                raise ValueError(f"第 {line_number} 行 task_id 无效：{value!r}")
            try:
                task_id = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"第 {line_number} 行 task_id 无效：{value!r}"
                ) from exc
            if task_id < 0 or str(task_id) != str(value).strip():
                raise ValueError(f"第 {line_number} 行 task_id 无效：{value!r}")
            if task_id in seen:
                raise ValueError(f"最终测试集包含重复 task_id：{task_id}")

            tasks.append((task_id, _query_from_record(record, line_number)))
            seen.add(task_id)

    if len(tasks) != expected_count:
        raise ValueError(
            f"最终测试集应包含 {expected_count} 题，实际为 {len(tasks)} 题"
        )
    return tasks


def load_rollout_result(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 rollout 结果 {path}：{exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"rollout 结果根节点不是 object：{path}")
    return result


def build_jsonl_record(
    task_id: int,
    query: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    """生成单行最终测试记录，保留完整 rollout 结果。"""

    return {
        "task_id": task_id,
        "query": query,
        "result": dict(result),
    }


def write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                )
                stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def save_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def build_summary(
    *,
    model: str,
    dataset: Path,
    results: Sequence[Mapping[str, Any]],
    trajectories_path: Path,
) -> Dict[str, Any]:
    statuses = Counter(str(result.get("status") or "unknown") for result in results)
    reward_sum = sum(_finite_float(result.get("reward")) for result in results)
    return {
        "model": model,
        "dataset": str(dataset),
        "task_count": len(results),
        "status_counts": dict(sorted(statuses.items())),
        "reward_sum": reward_sum,
        "average_reward": reward_sum / len(results) if results else 0.0,
        "full_reward_count": sum(
            math.isclose(
                _finite_float(result.get("reward")),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for result in results
        ),
        "trajectories_file": str(trajectories_path),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="仅评测 data/final_test_100.jsonl 中的 100 个任务",
    )
    parser.add_argument("--model", required=True, help="模型名称")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"最终测试集（默认：{DEFAULT_DATASET}）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"独立输出根目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
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
    parser.add_argument("--env-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=4)
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = load_final_test(args.dataset)
    task_ids = [task_id for task_id, _ in tasks]
    queries = dict(tasks)

    model_name = safe_model_name(args.model)
    run_dir = args.output_dir / model_name
    raw_root = run_dir / "raw"
    trajectories_path = run_dir / "trajectories.jsonl"
    summary_path = run_dir / "summary.json"

    client = create_openai_client(
        api_key=args.api_key,
        base_url=args.model_base_url,
        timeout=args.model_timeout,
    )
    run_rollouts(
        client=client,
        model=args.model,
        task_ids=task_ids,
        output_root=raw_root,
        env_base_url=args.env_base_url,
        workers=args.workers,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        env_timeout=args.env_timeout,
        persona=args.persona,
        overwrite=args.overwrite,
        model_options=(
            {"extra_body": args.model_extra_body}
            if args.model_extra_body is not None
            else None
        ),
    )

    jsonl_records: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    for task_id in task_ids:
        path = result_path(raw_root, args.model, task_id)
        result = load_rollout_result(path)
        results.append(result)
        jsonl_records.append(
            build_jsonl_record(task_id, queries[task_id], result)
        )

    write_jsonl_atomic(trajectories_path, jsonl_records)
    summary = build_summary(
        model=args.model,
        dataset=args.dataset,
        results=results,
        trajectories_path=trajectories_path,
    )
    save_json_atomic(summary_path, summary)

    print(f"最终测试集：{args.dataset}")
    print(f"测试任务数：{len(task_ids)}")
    print(f"轨迹 JSONL：{trajectories_path}")
    print(f"汇总结果：{summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    failed = sum(
        count
        for status, count in summary["status_counts"].items()
        if status in {"exception", "runner_exception"}
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc


__all__ = [
    "build_jsonl_record",
    "build_summary",
    "load_final_test",
    "load_rollout_result",
    "write_jsonl_atomic",
]
