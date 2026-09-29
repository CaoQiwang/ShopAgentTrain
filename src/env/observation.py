"""将 ShopSimulator 的原始页面观测转换为面向 Agent 的结构化观测。"""

from __future__ import annotations

import json
import re
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union


OBSERVATION_VERSION = "my-shopping-observation-v1"
SEP_MARKER = "[SEP]"

_SEARCH_AVAILABLE_RE = re.compile(
    r"(?:^|\r?\n)\s*搜索功能是否可用:\s*(True|False)", re.IGNORECASE
)
_ACTIONS_RE = re.compile(
    r"(?:^|\r?\n)\s*可点击的按钮:\s*(\[[^\r\n]*\])"
)
_PAGE_RE = re.compile(r"^Page\s+\d+\s*\(Total results:\s*\d+\)$", re.IGNORECASE)
_ASIN_RE = re.compile(r"^[A-Za-z0-9]{8,20}$")
_PRICE_PREFIX_RE = re.compile(r"^价格\s*:\s*", re.IGNORECASE)
_SHOP_PREFIX_RE = re.compile(r"^店铺\s*:\s*", re.IGNORECASE)
_INSTRUCTION_TOKEN_RE = re.compile(r"^Instruction\s*:\s*", re.IGNORECASE)
_SEARCH_ACTION_RE = re.compile(r"^search\[(.*)\]$", re.IGNORECASE | re.DOTALL)
_CLICK_ACTION_RE = re.compile(r"^click\[(.*)\]$", re.IGNORECASE | re.DOTALL)
_HISTORY_ACTION_RE = re.compile(
    r"^(?:search|click)\[.*\]$", re.IGNORECASE | re.DOTALL
)

_NAVIGATION_ACTIONS = {
    "back to search",
    "next >",
    "< prev",
    "description",
    "features",
    "reviews",
    "attributes",
    "buy now",
}


def _clean_text(value: Any) -> str:
    """移除保留分隔符，并清理分隔符产生的多余空白。"""

    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.replace(SEP_MARKER, " ")).strip()


def _sanitize(value: Any) -> Any:
    """递归保证最终观测的任意字符串中都不包含 ``[SEP]``。"""

    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, Mapping):
        return {_clean_text(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _unique_strings(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        cleaned = _clean_text(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _looks_like_asin(value: str) -> bool:
    value = value.strip()
    return bool(_ASIN_RE.fullmatch(value) and any(char.isdigit() for char in value))


def _normalise_action(action: Optional[str]) -> str:
    if not action:
        return ""
    action = str(action).replace("\\n", "\n").strip()
    if "\nAction: " in action:
        action = action.rsplit("\nAction: ", 1)[1].strip()
    return action


def _extract_raw_observation(observation: Union[str, Mapping[str, Any]]) -> str:
    if isinstance(observation, str):
        return observation
    if isinstance(observation, Mapping):
        raw = observation.get("instruction", "")
        if not isinstance(raw, str):
            raise TypeError("observation['instruction'] 必须是字符串")
        return raw
    raise TypeError("observation 必须是字符串或包含 instruction 的映射")


def _extract_page(raw: str) -> tuple[str, bool, List[str]]:
    """分离页面正文和 API 附加的可搜索状态、动作列表。"""

    search_matches = list(_SEARCH_AVAILABLE_RE.finditer(raw))
    search_available = bool(
        search_matches and search_matches[-1].group(1).casefold() == "true"
    )

    action_matches = list(_ACTIONS_RE.finditer(raw))
    actions: List[str] = []
    if action_matches:
        try:
            parsed = json.loads(action_matches[-1].group(1))
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            actions = _unique_strings(item for item in parsed if isinstance(item, str))

    suffix_starts = []
    if search_matches:
        suffix_starts.append(search_matches[-1].start())
    if action_matches:
        suffix_starts.append(action_matches[-1].start())
    page = raw[: min(suffix_starts)] if suffix_starts else raw
    return page.strip(), search_available, actions


def _page_tokens(page: str) -> List[str]:
    """把页面拆成可见文本，同时丢弃协议层注入的任务与历史动作。

    ShopSimulator 会把 ``Instruction: <任务>`` 当作页面中的第一个可见文本，
    并用 ``[SEP]`` 与真正页面内容分隔。必须丢弃整个 token；若只移除
    ``Instruction:`` 前缀，任务正文就会被误当作 Description/Features 内容。
    某些环境配置还会附带历史 ``search[...]``/``click[...]``，这里一并过滤。
    """

    result: List[str] = []
    for raw_token in page.split(SEP_MARKER):
        token = _clean_text(raw_token)
        if not token:
            continue
        if _INSTRUCTION_TOKEN_RE.match(token):
            continue
        if _HISTORY_ACTION_RE.fullmatch(token):
            continue
        result.append(token)
    return result


def _product_record(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get("product")
    if isinstance(nested, Mapping):
        return dict(nested)
    return dict(value)


class JsonlProductCatalog:
    """按需从 ShopSimulator 的 ``documents.jsonl`` 中读取商品元数据。

    查询结果会缓存在内存中。一次 ``get_many`` 调用只扫描文件一次，因此搜索页应
    批量查询当前页的所有 ASIN。
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._missing = set()
        self._lock = threading.RLock()

    def get_many(self, asins: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        requested = _unique_strings(asins)
        with self._lock:
            pending = {
                asin
                for asin in requested
                if asin not in self._cache and asin not in self._missing
            }
            if pending:
                with self.path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            document = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        product = _product_record(document)
                        asin = _clean_text(product.get("asin") or document.get("id"))
                        if asin in pending:
                            self._cache[asin] = product
                            pending.remove(asin)
                            if not pending:
                                break
                self._missing.update(pending)
            return {
                asin: deepcopy(self._cache[asin])
                for asin in requested
                if asin in self._cache
            }

    def get(self, asin: str, default: Any = None) -> Any:
        return self.get_many([asin]).get(str(asin), default)


CatalogSource = Optional[
    Union[Mapping[str, Mapping[str, Any]], JsonlProductCatalog, str, Path]
]

_DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "environments"
    / "ShopSimulator"
    / "shop_env"
    / "search_engine"
    / "resources"
    / "documents.jsonl"
)
_default_catalog: Optional[JsonlProductCatalog] = None
_default_catalog_lock = threading.Lock()


def _get_default_catalog() -> Optional[JsonlProductCatalog]:
    """返回仓库内置商品目录的共享惰性读取器。"""

    global _default_catalog
    if not _DEFAULT_CATALOG_PATH.is_file():
        return None
    with _default_catalog_lock:
        if _default_catalog is None:
            _default_catalog = JsonlProductCatalog(_DEFAULT_CATALOG_PATH)
        return _default_catalog


class ObservationConverter:
    """把同一条执行轨迹中的环境观测转换为结构化 Agent 观测。

    转换器会保留最近一次搜索词、当前商品和已选规格。调用 ``convert`` 时应传入
    产生该观测的动作，例如 ``search[儿童乳胶枕]`` 或 ``click[商品ASIN]``。
    """

    def __init__(self, product_catalog: CatalogSource = None) -> None:
        if product_catalog is None:
            product_catalog = _get_default_catalog()
        if isinstance(product_catalog, (str, Path)):
            product_catalog = JsonlProductCatalog(product_catalog)
        self.product_catalog = product_catalog
        self.instruction_text = ""
        self.query = ""
        self.current_asin = ""
        self.selected_options: Dict[str, str] = {}
        self.available_options: Dict[str, List[str]] = {}
        self._recent_products: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def reset(self) -> None:
        """清除一次执行轨迹中积累的页面状态。"""

        with self._lock:
            self.instruction_text = ""
            self.query = ""
            self.current_asin = ""
            self.selected_options = {}
            self.available_options = {}
            self._recent_products = {}

    def set_instruction(self, instruction: Any) -> None:
        """记录当前任务文本，用于从后续页面观测中精确移除任务回显。

        ShopSimulator 的不同页面可能返回 ``Instruction: <任务>``，也可能只返回
        裸任务文本。仅依赖前缀无法稳定识别，因此 AgentLoop 在 reset 后应调用
        本方法。这里仅做完整文本匹配，不使用包含或模糊匹配，避免误删商品内容。
        """

        cleaned = _clean_text(instruction)
        cleaned = _INSTRUCTION_TOKEN_RE.sub("", cleaned, count=1).strip()
        with self._lock:
            self.instruction_text = cleaned

    def convert(
        self,
        observation: Union[str, Mapping[str, Any]],
        *,
        action: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """转换一条环境观测，返回可直接序列化为 JSON 的字典。"""

        with self._lock:
            if isinstance(observation, Mapping) and observation.get(
                "observation_version"
            ) == OBSERVATION_VERSION:
                return _sanitize(deepcopy(dict(observation)))

            self._remember_action(action)
            if query is not None:
                self.query = _clean_text(query)

            raw = _extract_raw_observation(observation)
            page, search_available, actions = _extract_page(raw)
            tokens = self._remove_instruction_echo(_page_tokens(page))
            action_folds = {item.casefold() for item in actions}
            asin_actions = [action for action in actions if _looks_like_asin(action)]
            has_page_marker = any(_PAGE_RE.fullmatch(token) for token in tokens)

            # 当前 actions 比可能带有历史内容的 tokens 更可靠。商品详情页优先，
            # 避免历史搜索页的 ``Page ...`` 令当前详情页被误判为搜索结果页。
            if "buy now" in action_folds:
                result = self._convert_product_detail(
                    tokens, search_available, actions
                )
            elif asin_actions or (has_page_marker and not self.current_asin):
                result = self._convert_search_results(
                    tokens, search_available, actions
                )
            elif search_available:
                result = {
                    "observation_version": OBSERVATION_VERSION,
                    "page_type": "search",
                    "query": self.query,
                    "search_available": True,
                    "actions": actions,
                }
            else:
                result = self._convert_other_page(
                    tokens, search_available, actions
                )
            return _sanitize(result)

    def _remove_instruction_echo(self, tokens: Sequence[str]) -> List[str]:
        """删除带前缀或无前缀的当前任务回显。"""

        instruction_folded = self.instruction_text.casefold()
        result: List[str] = []
        for token in tokens:
            without_prefix = _INSTRUCTION_TOKEN_RE.sub("", token, count=1).strip()
            if instruction_folded and without_prefix.casefold() == instruction_folded:
                continue
            result.append(token)
        return result

    def to_json(
        self,
        observation: Union[str, Mapping[str, Any]],
        *,
        action: Optional[str] = None,
        query: Optional[str] = None,
        indent: Optional[int] = 2,
    ) -> str:
        """转换观测并输出保留中文字符的 JSON 字符串。"""

        return json.dumps(
            self.convert(observation, action=action, query=query),
            ensure_ascii=False,
            indent=indent,
        )

    def _remember_action(self, action: Optional[str]) -> None:
        action = _normalise_action(action)
        search_match = _SEARCH_ACTION_RE.fullmatch(action)
        if search_match:
            self.query = _clean_text(search_match.group(1))
            self.current_asin = ""
            self.selected_options = {}
            self.available_options = {}
            return

        click_match = _CLICK_ACTION_RE.fullmatch(action)
        if not click_match:
            return
        target = _clean_text(click_match.group(1))
        if _looks_like_asin(target):
            self.current_asin = target
            self.selected_options = {}
            self.available_options = {}
            return

        for name, values in self.available_options.items():
            for value in values:
                if value.casefold() == target.casefold():
                    self.selected_options[name] = value
                    return

    def _lookup_many(self, asins: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        source = self.product_catalog
        if source is None:
            return {}
        if hasattr(source, "get_many"):
            records = source.get_many(asins)  # type: ignore[union-attr]
            return {
                str(asin): _product_record(record)
                for asin, record in records.items()
            }

        records: Dict[str, Dict[str, Any]] = {}
        for asin in asins:
            record = source.get(asin)  # type: ignore[union-attr]
            if record is not None:
                records[asin] = _product_record(record)
        return records

    def _convert_search_results(
        self,
        tokens: List[str],
        search_available: bool,
        actions: List[str],
    ) -> Dict[str, Any]:
        page_indices = [
            index for index, token in enumerate(tokens) if _PAGE_RE.fullmatch(token)
        ]
        # 环境若附带历史 observation，当前页面位于最后，因此使用最后一个锚点。
        page_index = page_indices[-1] if page_indices else -1
        body = tokens[page_index + 1 :]
        asin_actions = [action for action in actions if _looks_like_asin(action)]
        metadata = self._lookup_many(asin_actions)

        products = []
        for rank, asin in enumerate(asin_actions, start=1):
            try:
                index = next(
                    index
                    for index, token in enumerate(body)
                    if token.casefold() == asin.casefold()
                )
            except StopIteration:
                continue
            title = body[index + 1] if index + 1 < len(body) else ""
            price = body[index + 2] if index + 2 < len(body) else ""
            product = self._agent_product(
                asin, metadata.get(asin, {}), title=title, price=price
            )
            products.append({"rank": rank, **product})
            self._recent_products[asin] = metadata.get(asin, {})

        return {
            "observation_version": OBSERVATION_VERSION,
            "page_type": "search_results",
            "query": self.query,
            "products": products,
            "search_available": search_available,
            "actions": actions,
        }

    def _convert_product_detail(
        self,
        tokens: List[str],
        search_available: bool,
        actions: List[str],
    ) -> Dict[str, Any]:
        price_indices = [
            index for index, token in enumerate(tokens) if _PRICE_PREFIX_RE.match(token)
        ]
        # 与搜索页相同，历史观测存在时最后一个价格锚点属于当前详情页。
        price_index = price_indices[-1] if price_indices else -1
        title = tokens[price_index - 1] if price_index > 0 else ""
        price = (
            _PRICE_PREFIX_RE.sub("", tokens[price_index], count=1)
            if price_index >= 0
            else ""
        )

        asin = self.current_asin or self._asin_for_title(title)
        self.current_asin = asin
        metadata = self._lookup_many([asin]).get(asin, {}) if asin else {}
        if not metadata and asin in self._recent_products:
            metadata = self._recent_products[asin]

        available_options = self._metadata_options(metadata, actions)
        if not available_options:
            available_options = self._options_from_tokens(tokens, actions, price_index)
        self.available_options = available_options
        self.selected_options = {
            name: value
            for name, value in self.selected_options.items()
            if name in available_options and value in available_options[name]
        }

        return {
            "observation_version": OBSERVATION_VERSION,
            "page_type": "product_detail",
            "product": self._agent_product(
                asin, metadata, title=title, price=price
            ),
            "selected_options": dict(self.selected_options),
            "available_options": available_options,
            "search_available": search_available,
            "actions": actions,
        }

    def _convert_other_page(
        self,
        tokens: List[str],
        search_available: bool,
        actions: List[str],
    ) -> Dict[str, Any]:
        action_set = {action.casefold() for action in actions}
        content = [
            token
            for token in tokens
            if token.casefold() not in action_set
            and not _INSTRUCTION_TOKEN_RE.match(token)
            and not _HISTORY_ACTION_RE.fullmatch(token)
        ]
        return {
            "observation_version": OBSERVATION_VERSION,
            "page_type": "product_subpage" if self.current_asin else "unknown",
            "product": {"asin": self.current_asin} if self.current_asin else {},
            "content": content,
            "search_available": search_available,
            "actions": actions,
        }

    def _asin_for_title(self, title: str) -> str:
        title_folded = title.casefold()
        for asin, metadata in self._recent_products.items():
            candidate = _clean_text(metadata.get("title") or metadata.get("Title"))
            if candidate.casefold() == title_folded:
                return asin
        return ""

    def _agent_product(
        self,
        asin: str,
        metadata: Mapping[str, Any],
        *,
        title: str,
        price: str,
    ) -> Dict[str, Any]:
        title = title or _clean_text(metadata.get("title") or metadata.get("Title"))
        price = price or _clean_text(metadata.get("Price") or metadata.get("price"))
        category = _clean_text(
            metadata.get("product_category") or metadata.get("category")
        )
        if category:
            category = re.split(r"\s*[›>/]\s*", category)[-1]

        brand = _clean_text(metadata.get("brand"))
        if not brand:
            brand = self._brand_from_shop(metadata.get("shop_name"))

        attributes = (
            metadata.get("key_attributes")
            or metadata.get("Attributes")
            or metadata.get("attribute")
            or []
        )
        if isinstance(attributes, str):
            attributes = [attributes]

        return {
            "asin": asin,
            "title": title,
            "price": price,
            "brand": brand,
            "category": category,
            "key_attributes": _unique_strings(attributes),
        }

    @staticmethod
    def _brand_from_shop(shop_name: Any) -> str:
        shop = _clean_text(shop_name)
        return re.sub(
            r"(?:官方)?(?:旗舰店|专卖店|专营店|企业店|店铺|店)$", "", shop
        ).strip()

    @staticmethod
    def _metadata_options(
        metadata: Mapping[str, Any], actions: Sequence[str]
    ) -> Dict[str, List[str]]:
        raw_options = metadata.get("options") or metadata.get("available_options")
        if not isinstance(raw_options, Mapping):
            raw_options = metadata.get("customization_options")
        if not isinstance(raw_options, Mapping):
            return {}

        action_spellings = {action.casefold(): action for action in actions}
        result: Dict[str, List[str]] = {}
        for raw_name, raw_values in raw_options.items():
            if not isinstance(raw_values, Sequence) or isinstance(raw_values, str):
                continue
            values = []
            for item in raw_values:
                if isinstance(item, Mapping):
                    if item.get("is_available") is False:
                        continue
                    item = item.get("value")
                value = _clean_text(item)
                if value:
                    values.append(action_spellings.get(value.casefold(), value))
            values = _unique_strings(values)
            if values:
                result[_clean_text(raw_name)] = values
        return result

    @staticmethod
    def _options_from_tokens(
        tokens: Sequence[str], actions: Sequence[str], price_index: int
    ) -> Dict[str, List[str]]:
        option_actions = [
            action
            for action in actions
            if action.casefold() not in _NAVIGATION_ACTIONS
            and not _looks_like_asin(action)
        ]
        if not option_actions or price_index <= 0:
            return {}

        option_folds = {value.casefold() for value in option_actions}
        nav_folds = _NAVIGATION_ACTIONS
        start = 0
        for index, token in enumerate(tokens[:price_index]):
            if token.casefold() in nav_folds:
                start = index + 1

        result: Dict[str, List[str]] = {}
        current_name = ""
        for token in tokens[start : price_index - 1]:
            if token.casefold() in option_folds:
                if current_name:
                    result.setdefault(current_name, []).append(token)
            else:
                current_name = token
                result.setdefault(current_name, [])
        return {name: _unique_strings(values) for name, values in result.items() if values}


def convert_observation(
    observation: Union[str, Mapping[str, Any]],
    *,
    action: Optional[str] = None,
    query: Optional[str] = None,
    product_catalog: CatalogSource = None,
) -> Dict[str, Any]:
    """无状态的便捷接口；连续交互应复用 :class:`ObservationConverter`。"""

    return ObservationConverter(product_catalog).convert(
        observation, action=action, query=query
    )


def observation_to_json(
    observation: Union[str, Mapping[str, Any]],
    *,
    action: Optional[str] = None,
    query: Optional[str] = None,
    product_catalog: CatalogSource = None,
    indent: Optional[int] = 2,
) -> str:
    """无状态地转换观测并返回 JSON 字符串。"""

    return json.dumps(
        convert_observation(
            observation,
            action=action,
            query=query,
            product_catalog=product_catalog,
        ),
        ensure_ascii=False,
        indent=indent,
    )


__all__ = [
    "OBSERVATION_VERSION",
    "JsonlProductCatalog",
    "ObservationConverter",
    "convert_observation",
    "observation_to_json",
]
