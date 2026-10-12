"""ShopSimulator 购买结果和轨迹错误的自定义奖励。"""

from __future__ import annotations

from typing import Any, Mapping


RESULT_METRICS = ("r_type", "r_att", "r_option", "r_price")


def invalid_action_reward(already_penalized: bool) -> float:
    """轨迹中首次出现格式错误或不合法动作时扣 0.1。"""
    return 0.0 if already_penalized else -0.1


def exact_match_result_reward(result: Mapping[str, Any]) -> float:
    """购买后四项全满足给 1，仅规格不符给 0.3，其余给 0。

    使用模拟器终局返回的 ``reward_detail``，不依赖它的加权总分。
    缺少指标时按不满足处理，避免无购买或异常响应被误判为成功。
    """

    if not isinstance(result, Mapping) or result.get("done") is not True:
        return 0.0
    detail = result.get("reward_detail")
    if not isinstance(detail, Mapping):
        return 0.0
    for metric in RESULT_METRICS:
        value = detail.get(metric)
        if type(value) not in (bool, int, float):
            return 0.0
        if metric != "r_option" and value != 1:
            return 0.0

    option = detail["r_option"]
    if option == 1:
        return 1.0
    return 0.3 if 0 <= option < 1 else 0.0


__all__ = ["RESULT_METRICS", "invalid_action_reward", "exact_match_result_reward"]
