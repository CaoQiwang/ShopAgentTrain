"""ShopSimulator 对模型公开的唯一工具定义。

这里同时保存两件必须保持同步的内容：模型看到的 JSON Schema，以及把工具调用
转换成环境 ``search[...]``、``click[...]``、``finish[...]`` 动作的映射。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Tuple, Union


CLICK_TOOL_ACTIONS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "open_product": ("asin", None),
    "select_option": ("value", None),
    "view_description": (None, "Description"),
    "view_features": (None, "Features"),
    "view_reviews": (None, "Reviews"),
    "view_attributes": (None, "Attributes"),
    "next_page": (None, "Next >"),
    "prev_page": (None, "< Prev"),
    "back_to_search": (None, "Back to Search"),
    "buy_now": (None, "Buy Now"),
}


class ToolCallError(ValueError):
    """工具调用无法转换为环境动作。"""


def _parameters_dict(
    parameters: Optional[Union[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    """兼容字典参数和 OpenAI tool call 中的 JSON 字符串参数。"""

    if parameters is None:
        return {}
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters or "{}")
        except json.JSONDecodeError as exc:
            raise ToolCallError(f"工具参数不是合法 JSON: {exc}") from exc
    if not isinstance(parameters, Mapping):
        raise ToolCallError("工具参数必须是 JSON object")
    return dict(parameters)


def _required_string(parameters: Mapping[str, Any], key: str, name: str) -> str:
    value = parameters.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolCallError(f"工具 {name!r} 的参数 {key!r} 必须是非空字符串")
    return value.strip()


def tool_call_to_action(
    name: str,
    parameters: Optional[Union[str, Mapping[str, Any]]] = None,
) -> str:
    """把一个标准 tool call 转成 ShopSimulator 能执行的字符串动作。

    未知工具、参数缺失或参数类型错误会抛出 :class:`ToolCallError`，避免把无效
    动作静默发送给环境。
    """

    if not isinstance(name, str) or not name.strip():
        raise ToolCallError("工具名称必须是非空字符串")
    name = name.strip()
    parameters_dict = _parameters_dict(parameters)

    if name == "search_products":
        query = _required_string(parameters_dict, "query", name)
        return f"search[{query}]"
    if name == "finish_without_purchase":
        reason = _required_string(parameters_dict, "reason", name)
        if reason != "no_suitable_product":
            raise ToolCallError(
                "finish_without_purchase.reason 只能是 'no_suitable_product'"
            )
        return f"finish[{reason}]"

    try:
        key, fixed_value = CLICK_TOOL_ACTIONS[name]
    except KeyError as exc:
        raise ToolCallError(f"未知工具: {name!r}") from exc

    value = (
        fixed_value
        if fixed_value is not None
        else _required_string(parameters_dict, str(key), name)
    )
    return f"click[{value}]"


def _schema(
    name: str,
    description: str,
    properties: Optional[Dict[str, Any]] = None,
    required: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """生成统一格式的 OpenAI function schema，并禁止额外参数。"""

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


_INTERACTION_TOOL_SCHEMAS = [
    _schema(
        "search_products",
        "仅当最新 observation 的 search_available 为 true 时搜索。query 应简洁，"
        "包含品类和最有区分度的品牌、型号、功能或规格；不得重复相同查询或机械"
        "复制整段需求。",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _schema(
        "open_product",
        "打开最新 observation 当前搜索结果中列出的候选商品以核验价格、功能和"
        "规格；asin 必须原样取自该页面。",
        {"asin": {"type": "string"}},
        ["asin"],
    ),
    _schema(
        "select_option",
        "为当前候选补齐所有影响可购买 variant 的必要规格轴；value 必须来自最新"
        " observation，同一规格轴只选一个值，不得选择导航按钮或历史页面中的值。"
        "选择后应按完整 variant 的实际价格重新检查预算。",
        {"value": {"type": "string"}},
        ["value"],
    ),
    _schema(
        "view_description",
        "仅当当前页面的 actions 包含 description 且需要补充核验商品时打开；"
        "无参数，必须传 {}。",
    ),
    _schema(
        "view_features",
        "仅当当前页面的 actions 包含 features 且需要核验核心功能时打开；"
        "无参数，必须传 {}。",
    ),
    _schema(
        "view_reviews",
        "仅当当前页面的 actions 包含 reviews 且需要补充判断时打开；"
        "无参数，必须传 {}。",
    ),
    _schema(
        "view_attributes",
        "仅当当前页面的 actions 包含 attributes 且需要核验关键属性时打开；"
        "无参数，必须传 {}。",
    ),
    _schema(
        "next_page",
        "仅当当前页面的 actions 包含 next > 且当前页没有合适候选时翻到下一页；"
        "无参数，必须传 {}。",
    ),
    _schema(
        "prev_page",
        "仅当当前页面的 actions 包含 < prev 时返回上一页；无参数，必须传 {}。",
    ),
    _schema(
        "back_to_search",
        "仅当当前页面的 actions 包含 back to search 时返回搜索页；"
        "无参数，必须传 {}。",
    ),
    _schema(
        "buy_now",
        "不可撤销的终止动作。仅当最新 observation 的 actions 包含 buy now、品类"
        "正确、完整 variant 的实际价格在预算内，且按品牌、型号与核心功能、规格"
        "属性比较后是已核验候选中的最佳选择时购买；无参数，必须传 {}。",
    ),
]

_FINISH_WITHOUT_PURCHASE_SCHEMA = _schema(
    "finish_without_purchase",
    "主动结束且不购买，这不是成功。经过多次有实质差异的搜索和多个候选核验，"
    "仍没有可接受商品，并且当前没有明显值得继续核验的候选时调用；是否达到资格"
    "由环境判断。",
    {
        "reason": {
            "type": "string",
            "enum": ["no_suitable_product"],
        }
    },
    ["reason"],
)

SHOP_TOOL_SCHEMAS = [
    *_INTERACTION_TOOL_SCHEMAS,
    _FINISH_WITHOUT_PURCHASE_SCHEMA,
]


__all__ = [
    "CLICK_TOOL_ACTIONS",
    "SHOP_TOOL_SCHEMAS",
    "ToolCallError",
    "tool_call_to_action",
]
