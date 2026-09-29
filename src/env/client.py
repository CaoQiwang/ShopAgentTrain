"""ShopSimulator HTTP 环境的客户端封装。

每次执行轨迹都应创建一个 :class:`ShopEnv` 实例。每个实例在同一时间只持有一个
远程 ``env_idx``，因此不同智能体不会相互覆盖对方的环境状态。
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:5000"
API_PATH = "/api/shop_agent"
TaskId = Union[int, str]


class ShopEnvError(RuntimeError):
    """ShopSimulator 客户端抛出的基础异常。"""


class ShopEnvProtocolError(ShopEnvError):
    """当 ShopSimulator 返回无效响应或错误响应时抛出。"""


class ShopEnvStateError(ShopEnvError):
    """当操作不符合本地生命周期状态时抛出。"""


class ShopEnv:
    """用于 ShopSimulator 单次执行轨迹的客户端。

    ``base_url`` 可以是服务根地址，也可以是完整的 ``/api/shop_agent`` URL。
    每次执行轨迹都应创建一个客户端，并调用 ``close()``；也可以将其用作上下文
    管理器，以便提前退出时归还服务器槽位。

    设置 ``persona=True`` 后，会将 ``instruction_simple`` 作为 ``instruction``
    暴露，以匹配 ShopSimulator 的角色评估器。外部注入的 requests 会话仍由调用方
    负责管理。
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: Union[float, Tuple[float, float]] = 60,
        *,
        persona: bool = False,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = self._build_endpoint(base_url)
        self.timeout = timeout
        self.persona = persona

        self.task_id: Optional[TaskId] = None
        self.env_idx: Optional[int] = None
        self.instruction: Optional[str] = None

        self._session = session or requests.Session()
        self._owns_session = session is None
        self._closed = False
        self._lock = threading.RLock()

    @staticmethod
    def _build_endpoint(base_url: str) -> str:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty HTTP(S) URL")

        url = base_url.strip().rstrip("/")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"base_url must be an absolute HTTP(S) URL, got {base_url!r}"
            )
        if parsed.path in {"", "/"}:
            return f"{url}{API_PATH}"
        if parsed.path.rstrip("/") == API_PATH:
            return url
        raise ValueError(
            "base_url must be a service root or end with '/api/shop_agent'"
        )

    @property
    def active(self) -> bool:
        """该客户端当前是否持有远程环境槽位。"""

        return self.env_idx is not None

    def _ensure_open(self) -> None:
        if self._closed:
            raise ShopEnvStateError("ShopEnv is closed; create a new client")

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_open()
        try:
            response = self._session.post(
                self.base_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ShopEnvError(
                f"ShopSimulator request failed ({payload.get('action')}): {exc}"
            ) from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise ShopEnvProtocolError(
                f"ShopSimulator returned non-JSON data (HTTP {response.status_code})"
            ) from exc

        if not isinstance(body, dict) or "result" not in body:
            raise ShopEnvProtocolError(
                "ShopSimulator response must be an object containing 'result'"
            )
        result = body["result"]
        if not isinstance(result, dict):
            raise ShopEnvProtocolError("ShopSimulator 'result' must be an object")
        if result.get("error"):
            raise ShopEnvProtocolError(str(result["error"]))
        return result

    def reset(self, task_id: TaskId) -> Dict[str, Any]:
        """启动任务并返回 ShopSimulator 的初始观测。

        如果上一次执行轨迹仍占用槽位，会先将其释放，避免重复重置耗尽服务器端
        数量有限的槽位池。
        """

        if isinstance(task_id, bool) or not isinstance(task_id, (int, str)):
            raise TypeError("task_id must be an int or str")

        with self._lock:
            self._ensure_open()
            if self.active:
                self.release()

            result = self._request({"action": "reset", "idx": task_id})
            env_idx = result.get("env_idx")
            if isinstance(env_idx, bool) or not isinstance(env_idx, int):
                raise ShopEnvProtocolError(
                    "ShopSimulator reset response is missing an integer 'env_idx'"
                )

            self.task_id = task_id
            self.env_idx = env_idx
            self._apply_persona(result)
            self.instruction = result.get("instruction")
            return result

    def _apply_persona(self, result: Dict[str, Any]) -> None:
        if not self.persona:
            return
        simple = result.get("instruction_simple")
        if simple is None:
            raise ShopEnvProtocolError(
                "Persona mode requires 'instruction_simple' in the reset response"
            )
        result["instruction"] = simple
        result.pop("instruction_simple", None)

    def step(self, action: str) -> Dict[str, Any]:
        """执行一次 ``search[...]`` 或 ``click[...]`` 操作。

        原始结果包含 ``instruction``（当前观测）、``reward``、``done``、``over``
        和 ``reward_detail`` 等字段。保留 ``interact`` 别名是为了兼容评估器。
        """

        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")

        with self._lock:
            self._ensure_open()
            if self.env_idx is None:
                raise ShopEnvStateError("Call reset(task_id) before step(action)")

            result = self._request(
                {
                    "action": "interact",
                    "env_idx": self.env_idx,
                    "response": action,
                }
            )

            if self.persona and self.instruction:
                returned_instruction = result.get("instruction")
                if isinstance(returned_instruction, str):
                    parts = returned_instruction.split("[SEP]", 2)
                    if len(parts) == 3:
                        result["instruction"] = (
                            f"{parts[0]}[SEP] {self.instruction} [SEP]{parts[2]}"
                        )

            # 当 `over` 为真时，pack_api.py 会立即释放槽位。
            # 清除本地已失效的 ID，避免 close() 释放已被其他并发执行轨迹复用的槽位。
            if bool(result.get("over")):
                self._clear_rollout_state()
            return result

    interact = step

    def search(self, query: str) -> Dict[str, Any]:
        """对 ``step('search[...]')`` 的便捷封装。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return self.step(f"search[{query}]")

    def click(self, target: str) -> Dict[str, Any]:
        """对 ``step('click[...]')`` 的便捷封装。"""

        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a non-empty string")
        return self.step(f"click[{target}]")

    def release(self) -> Optional[Dict[str, Any]]:
        """释放当前远程槽位；如果没有活动槽位，则不执行任何操作。"""

        with self._lock:
            self._ensure_open()
            if self.env_idx is None:
                return None

            result = self._request(
                {"action": "release_one", "env_idx": self.env_idx}
            )
            self._clear_rollout_state()
            return result

    def _clear_rollout_state(self) -> None:
        self.env_idx = None
        self.task_id = None
        self.instruction = None

    def close(self) -> None:
        """尽力释放该客户端持有的资源。

        如果调用方需要获知服务器错误，请使用 ``release``。``close`` 可以安全地
        用在 finally 代码块和上下文管理器退出流程中。
        """

        with self._lock:
            if self._closed:
                return
            try:
                if self.active:
                    try:
                        self.release()
                    except ShopEnvError:
                        pass
            finally:
                if self._owns_session:
                    self._session.close()
                self._closed = True
                self._clear_rollout_state()

    def __enter__(self) -> "ShopEnv":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "ShopEnv",
    "ShopEnvError",
    "ShopEnvProtocolError",
    "ShopEnvStateError",
]
