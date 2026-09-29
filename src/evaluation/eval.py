"""汇总评估一个模型在 ShopSimulator 上生成的 rollout 结果。"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FULL_REWARD_TOLERANCE = 1e-9

FIXED_TOOL_TARGETS = {
    "view_description": "description",
    "view_features": "features",
    "view_reviews": "reviews",
    "view_attributes": "attributes",
    "next_page": "next >",
    "prev_page": "< prev",
    "back_to_search": "back to search",
    "buy_now": "buy now",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _rate(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _average(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _normalised_component(detail: Mapping[str, Any], key: str) -> float:
    default = 1.0 if key == "r_option" else 0.0
    return _safe_float(detail.get(key, default), default)


def _is_full_success(result: Mapping[str, Any]) -> bool:
    """最终正确：完整满足类型、属性、规格和价格要求。"""

    reward = _safe_float(result.get("reward"))
    if math.isclose(reward, 1.0, abs_tol=FULL_REWARD_TOLERANCE):
        return True

    detail = result.get("reward_detail")
    if not isinstance(detail, Mapping) or not detail:
        return False
    return all(
        math.isclose(
            _normalised_component(detail, key),
            1.0,
            abs_tol=FULL_REWARD_TOLERANCE,
        )
        for key in ("r_type", "r_att", "r_option", "r_price")
    )


def _is_purchase(result: Mapping[str, Any]) -> bool:
    return result.get("status") == "purchased" or bool(result.get("reward_detail"))


def _exact_product_match(result: Mapping[str, Any]) -> bool:
    purchase = result.get("purchase")
    goal = result.get("goal")
    if not isinstance(purchase, Mapping) or not isinstance(goal, Mapping):
        return False
    purchased_asin = str(purchase.get("asin") or "").strip().casefold()
    goal_asin = str(goal.get("asin") or "").strip().casefold()
    return bool(purchased_asin and goal_asin and purchased_asin == goal_asin)


def _arguments(function: Mapping[str, Any]) -> Dict[str, Any]:
    value = function.get("arguments", {})
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "{}")
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _tool_is_legal(
    name: str,
    arguments: Mapping[str, Any],
    observation: Optional[Mapping[str, Any]],
) -> Optional[bool]:
    """根据当时最新的结构化 observation 判断工具是否可执行。"""

    if name in {"think", "finish_without_purchase"}:
        return None
    if not isinstance(observation, Mapping):
        return None

    actions = {
        str(action).strip().casefold()
        for action in observation.get("actions", [])
    }
    if name == "search_products":
        return bool(observation.get("search_available", False))
    if name == "open_product":
        return str(arguments.get("asin") or "").strip().casefold() in actions
    if name == "select_option":
        return str(arguments.get("value") or "").strip().casefold() in actions
    if name in FIXED_TOOL_TARGETS:
        return FIXED_TOOL_TARGETS[name] in actions
    return False


def audit_messages(messages: Any) -> Dict[str, Any]:
    """统计模型空调用、工具合法性以及动作后页面是否保持不变。"""

    if not isinstance(messages, list):
        return {
            "model_turns": 0,
            "tool_call_turns": 0,
            "no_tool_call_turns": 0,
            "empty_no_tool_call_turns": 0,
            "checked_action_calls": 0,
            "legal_action_calls": 0,
            "illegal_action_calls": 0,
            "unchanged_observation_actions": 0,
        }

    current_observation: Optional[Mapping[str, Any]] = None
    pending: Optional[Tuple[Optional[bool], Optional[Mapping[str, Any]]]] = None
    counts = Counter()

    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        content = message.get("content") or ""

        if role == "user" and isinstance(content, str):
            try:
                body = json.loads(content)
            except json.JSONDecodeError:
                body = None
            if isinstance(body, Mapping) and isinstance(
                body.get("observation"), Mapping
            ):
                current_observation = body["observation"]

        elif role == "assistant":
            counts["model_turns"] += 1
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                counts["no_tool_call_turns"] += 1
                if not str(content).strip():
                    counts["empty_no_tool_call_turns"] += 1
                pending = None
                continue

            counts["tool_call_turns"] += 1
            call = tool_calls[0] if isinstance(tool_calls, list) else {}
            function = call.get("function", {}) if isinstance(call, Mapping) else {}
            if not isinstance(function, Mapping):
                function = {}
            name = str(function.get("name") or "")
            legal = _tool_is_legal(name, _arguments(function), current_observation)
            if legal is not None:
                counts["checked_action_calls"] += 1
                key = "legal_action_calls" if legal else "illegal_action_calls"
                counts[key] += 1
            pending = (legal, current_observation)

        elif role == "tool" and isinstance(content, str):
            try:
                body = json.loads(content)
            except json.JSONDecodeError:
                body = None
            if isinstance(body, Mapping) and body.get("observation_version"):
                if pending is not None and pending[1] is not None:
                    if dict(body) == dict(pending[1]):
                        counts["unchanged_observation_actions"] += 1
                current_observation = body
            pending = None

    return {
        key: counts[key]
        for key in (
            "model_turns",
            "tool_call_turns",
            "no_tool_call_turns",
            "empty_no_tool_call_turns",
            "checked_action_calls",
            "legal_action_calls",
            "illegal_action_calls",
            "unchanged_observation_actions",
        )
    }


def load_results(
    result_dir: Path, *, task_ids: Optional[range] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """读取数字文件名的任务结果，忽略 summary/evaluation 文件。"""

    results: List[Dict[str, Any]] = []
    errors: List[str] = []
    paths = sorted(
        (
            path
            for path in result_dir.glob("*.json")
            if path.stem.isdigit()
            and (task_ids is None or int(path.stem) in task_ids)
        ),
        key=lambda path: int(path.stem),
    )
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path.name}: 顶层 JSON 不是 object")
            continue
        value.setdefault("task_id", int(path.stem))
        results.append(value)
    return results, errors


def infer_expected_tasks(result_dir: Path, result_count: int) -> int:
    """优先参考 summary.selected，但绝不小于实际有效结果数。"""

    selected = 0
    summary_path = result_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            selected = int(summary.get("selected", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            selected = 0
    return max(selected, result_count)


def evaluate_results(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_tasks: Optional[int] = None,
) -> Dict[str, Any]:
    """计算整体指标；缺失任务在所有成功率和总体均分中按失败计。"""

    valid_count = len(results)
    total = valid_count if expected_tasks is None else int(expected_tasks)
    if total < valid_count:
        raise ValueError(
            f"expected_tasks={total} 不能小于有效结果文件数量={valid_count}"
        )
    if total < 0:
        raise ValueError("expected_tasks 不能为负数")

    rewards = [_safe_float(result.get("reward")) for result in results]
    purchase_results = [result for result in results if _is_purchase(result)]
    purchase_rewards = [_safe_float(result.get("reward")) for result in purchase_results]
    full_successes = [result for result in results if _is_full_success(result)]
    partial_successes = [
        result
        for result in results
        if 0.0 < _safe_float(result.get("reward")) < 1.0 - FULL_REWARD_TOLERANCE
    ]
    exact_matches = [result for result in results if _exact_product_match(result)]

    statuses = Counter(str(result.get("status") or "unknown") for result in results)
    tool_counts = Counter()
    environment_action_count = 0
    steps: List[float] = []
    durations: List[float] = []
    successful_steps: List[float] = []
    audits = Counter()

    component_values = {
        key: [] for key in ("r_type", "r_att", "r_option", "r_price")
    }
    component_passes = Counter()
    hard_score_sum = 0.0

    for result in results:
        step = _safe_float(result.get("steps"))
        duration = _safe_float(result.get("duration_seconds"))
        steps.append(step)
        durations.append(duration)
        if _is_full_success(result):
            successful_steps.append(step)

        trajectory = result.get("trajectory")
        if isinstance(trajectory, list):
            for event in trajectory:
                if isinstance(event, Mapping):
                    tool_name = str(event.get("tool") or "unknown")
                    tool_counts[tool_name] += 1
                    if (
                        tool_name not in {"think", "finish_without_purchase"}
                        and event.get("environment_action") is not None
                    ):
                        environment_action_count += 1

        audit = audit_messages(result.get("messages"))
        audits.update(audit)

        detail = result.get("reward_detail")
        detail = detail if isinstance(detail, Mapping) else {}
        components = {
            key: _normalised_component(detail, key) if detail else 0.0
            for key in component_values
        }
        for key, value in components.items():
            component_values[key].append(value)
            if math.isclose(value, 1.0, abs_tol=FULL_REWARD_TOLERANCE):
                component_passes[key] += 1
        hard_score_sum += math.prod(components.values())

    missing_count = total - valid_count
    reward_sum = sum(rewards)
    checked_actions = audits["checked_action_calls"]
    model_turns = audits["model_turns"]

    return {
        "dataset": {
            "expected_tasks": total,
            "valid_result_files": valid_count,
            "missing_result_files": missing_count,
            "result_coverage_rate": _rate(valid_count, total),
        },
        "outcome": {
            # 用户要求的核心指标：最终完全正确的任务占全部评测任务的比例。
            "task_completion_rate": _rate(len(full_successes), total),
            "full_success_count": len(full_successes),
            "purchase_rate": _rate(len(purchase_results), total),
            "purchase_count": len(purchase_results),
            "partial_success_rate": _rate(len(partial_successes), total),
            "partial_success_count": len(partial_successes),
            "exact_product_accuracy": _rate(len(exact_matches), total),
            "exact_product_match_count": len(exact_matches),
            "status_counts": dict(sorted(statuses.items())),
        },
        "reward": {
            "reward_sum": reward_sum,
            "average_reward_all_tasks": _rate(reward_sum, total),
            "average_reward_purchased_tasks": _average(purchase_rewards),
            "hard_score_average_all_tasks": _rate(hard_score_sum, total),
        },
        "components": {
            key: {
                "average_all_tasks": _rate(sum(values), total),
                "pass_rate_all_tasks": _rate(component_passes[key], total),
            }
            for key, values in component_values.items()
        },
        "efficiency": {
            "average_environment_steps": _average(steps),
            "median_environment_steps": median(steps) if steps else 0.0,
            "p95_environment_steps": _percentile(steps, 0.95),
            "average_steps_full_success": _average(successful_steps),
            "average_duration_seconds": _average(durations),
            "total_duration_seconds_sum": sum(durations),
        },
        "tooling": {
            "tool_counts": dict(sorted(tool_counts.items())),
            "environment_action_count": environment_action_count,
            "model_turn_count": model_turns,
            "no_tool_call_count": audits["no_tool_call_turns"],
            "no_tool_call_rate": _rate(audits["no_tool_call_turns"], model_turns),
            "empty_no_tool_call_count": audits["empty_no_tool_call_turns"],
            "checked_action_calls": checked_actions,
            "legal_action_calls": audits["legal_action_calls"],
            "illegal_action_calls": audits["illegal_action_calls"],
            "action_validity_rate": _rate(
                audits["legal_action_calls"], checked_actions
            ),
            "unchanged_observation_actions": audits[
                "unchanged_observation_actions"
            ],
            "unchanged_observation_rate": _rate(
                audits["unchanged_observation_actions"], environment_action_count
            ),
        },
        "task_ids": {
            "full_success": sorted(
                int(result.get("task_id")) for result in full_successes
            ),
            "partial_success": sorted(
                int(result.get("task_id")) for result in partial_successes
            ),
            "purchased": sorted(
                int(result.get("task_id")) for result in purchase_results
            ),
            "exact_product_match": sorted(
                int(result.get("task_id")) for result in exact_matches
            ),
        },
    }


def save_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def evaluate_directory(
    result_dir: Path,
    *,
    expected_tasks: Optional[int] = None,
    task_start: Optional[int] = None,
) -> Dict[str, Any]:
    result_dir = Path(result_dir)
    if not result_dir.is_dir():
        raise FileNotFoundError(f"结果目录不存在: {result_dir}")

    if task_start is not None:
        if expected_tasks is None:
            raise ValueError("--task-start 需要同时指定 --expected-tasks")
        if task_start < 0:
            raise ValueError("--task-start 不能为负数")
        task_ids = range(task_start, task_start + expected_tasks)
    else:
        task_ids = None

    results, load_errors = load_results(result_dir, task_ids=task_ids)
    if expected_tasks is None:
        expected_tasks = infer_expected_tasks(result_dir, len(results))
    evaluation = evaluate_results(results, expected_tasks=expected_tasks)
    evaluation["source"] = {
        "result_directory": str(result_dir),
        "load_error_count": len(load_errors),
        "load_errors": load_errors,
    }
    if task_ids is not None:
        evaluation["source"]["selected_task_id_range"] = {
            "start": task_ids.start,
            "end_exclusive": task_ids.stop,
        }
    return evaluation


def _format_rate(value: Any) -> str:
    return f"{100 * _safe_float(value):.2f}%"


def _safe_model_name(model: str) -> str:
    name = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "__", model).strip(" .")
    return name or "unnamed-model"


def print_evaluation(evaluation: Mapping[str, Any]) -> None:
    dataset = evaluation["dataset"]
    outcome = evaluation["outcome"]
    reward = evaluation["reward"]
    components = evaluation["components"]
    efficiency = evaluation["efficiency"]
    tooling = evaluation["tooling"]

    print("\n========== ShopSimulator 整体评估 ==========")
    print(
        f"结果覆盖: {dataset['valid_result_files']}/{dataset['expected_tasks']} "
        f"({_format_rate(dataset['result_coverage_rate'])})"
    )
    print(
        f"任务完成率（最终完全正确）: "
        f"{outcome['full_success_count']}/{dataset['expected_tasks']} "
        f"({_format_rate(outcome['task_completion_rate'])})"
    )
    print(
        f"购买完成率: {outcome['purchase_count']}/{dataset['expected_tasks']} "
        f"({_format_rate(outcome['purchase_rate'])})"
    )
    print(
        f"部分成功率: {outcome['partial_success_count']}/{dataset['expected_tasks']} "
        f"({_format_rate(outcome['partial_success_rate'])})"
    )
    print(
        f"精确商品命中率: {outcome['exact_product_match_count']}/"
        f"{dataset['expected_tasks']} "
        f"({_format_rate(outcome['exact_product_accuracy'])})"
    )
    print(f"全部任务平均 reward: {reward['average_reward_all_tasks']:.4f}")
    print(
        f"已购买任务平均 reward: "
        f"{reward['average_reward_purchased_tasks']:.4f}"
    )
    print(
        "分项通过率: "
        + ", ".join(
            f"{key}={_format_rate(value['pass_rate_all_tasks'])}"
            for key, value in components.items()
        )
    )
    print(
        f"平均环境步数: {efficiency['average_environment_steps']:.2f}；"
        f"满分任务平均步数: {efficiency['average_steps_full_success']:.2f}"
    )
    print(
        f"工具动作合法率: {_format_rate(tooling['action_validity_rate'])}；"
        f"无工具调用率: {_format_rate(tooling['no_tool_call_rate'])}"
    )
    print(f"状态分布: {json.dumps(outcome['status_counts'], ensure_ascii=False)}")
    print("=============================================\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估 ShopSimulator rollout 结果")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="模型名称，对应 output/<model>")
    group.add_argument("--result-dir", type=Path, help="直接指定模型结果目录")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--expected-tasks",
        type=int,
        help="评测任务总数；缺失结果按失败计。默认读取 summary.selected",
    )
    parser.add_argument(
        "--task-start",
        type=int,
        help="只评估从该 ID 开始的 expected-tasks 个任务；需同时指定 --expected-tasks",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        help="评估 JSON 保存路径；默认写入结果目录/evaluation.json",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result_dir = args.result_dir or (
        args.output_dir / _safe_model_name(args.model)
    )
    evaluation = evaluate_directory(
        result_dir,
        expected_tasks=args.expected_tasks,
        task_start=args.task_start,
    )
    print_evaluation(evaluation)

    save_path = args.save_path or (result_dir / "evaluation.json")
    save_json_atomic(save_path, evaluation)
    print(f"评估结果已保存: {save_path}")

    return 0 if evaluation["source"]["load_error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "audit_messages",
    "evaluate_directory",
    "evaluate_results",
    "load_results",
    "print_evaluation",
]
