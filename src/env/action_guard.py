"""根据最新 observation 阻止 ShopSimulator 的无效动作。

工具参数能否转换为动作由 :mod:`src.env.tools` 负责；本模块负责第二层校验：
动作字符串是否合法，以及该动作在最新页面中是否可用。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping


# 与环境的 ``name[value]`` 语法保持一致，同时用 fullmatch 拒绝尾随垃圾文本。
# value 内部可能是商品规格的一部分，因此允许其中出现方括号。
_ACTION_RE = re.compile(r"^(search|click)\[(.+)\]$", re.DOTALL)


class ActionGuardError(ValueError):
    """动作在发送给环境前未通过校验。"""

    error_type = "action_guard_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details

    def feedback(self) -> Dict[str, Any]:
        """生成可直接作为 tool result 返回给 agent 的结构化错误。"""

        return {
            "error": {
                "type": self.error_type,
                "message": str(self),
                **self.details,
            }
        }


class ActionFormatError(ActionGuardError):
    """动作字符串不是环境支持的 ``search[...]``/``click[...]`` 格式。"""

    error_type = "action_format_invalid"


class ActionUnavailableError(ActionGuardError):
    """动作格式正确，但在最新 observation 所代表的页面中不可用。"""

    error_type = "action_unavailable"


def _available_actions(observation: Mapping[str, Any]) -> List[str]:
    actions = observation.get("actions", [])
    if not isinstance(actions, list):
        return []
    return [action for action in actions if isinstance(action, str)]


def validate_action(action: str, observation: Mapping[str, Any]) -> None:
    """校验动作格式及其在最新 observation 中的可用性。

    校验成功时返回 ``None``；失败时分别抛出
    :class:`ActionFormatError` 或 :class:`ActionUnavailableError`。
    点击目标采用与 ShopSimulator 一致的大小写不敏感比较，但不会对内容做模糊
    匹配，避免误点历史页面或相似规格。
    """

    if not isinstance(action, str) or not action.strip():
        raise ActionFormatError(
            "动作格式不合法：动作必须是非空字符串。",
            requested_action=action,
            expected_format="search[query] 或 click[target]",
        )

    match = _ACTION_RE.fullmatch(action.strip())
    if match is None:
        raise ActionFormatError(
            "动作格式不合法：环境只接受非空的 search[query] 或 click[target]。",
            requested_action=action,
            expected_format="search[query] 或 click[target]",
        )

    if not isinstance(observation, Mapping):
        raise ActionUnavailableError(
            "无法判断动作是否可用：最新 observation 不是 JSON object。",
            requested_action=action,
            available_actions=[],
        )

    action_name, target = match.groups()
    actions = _available_actions(observation)
    context = {
        "requested_action": action,
        "page_type": observation.get("page_type"),
        "search_available": observation.get("search_available") is True,
        "available_actions": actions,
    }

    if action_name == "search":
        if observation.get("search_available") is not True:
            raise ActionUnavailableError(
                "当前场景下动作不可用：最新 observation 不允许搜索。",
                **context,
            )
        return

    available_by_casefold = {item.casefold(): item for item in actions}
    if target.casefold() not in available_by_casefold:
        raise ActionUnavailableError(
            "当前场景下动作不可用：点击目标不在最新 observation.actions 中。",
            requested_target=target,
            **context,
        )


__all__ = [
    "ActionFormatError",
    "ActionGuardError",
    "ActionUnavailableError",
    "validate_action",
]
