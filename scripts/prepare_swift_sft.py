#!/usr/bin/env python3
"""将 OpenAI 工具调用轨迹转换为仓库内 ms-swift 支持的格式。"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env.tools import SHOP_TOOL_SCHEMAS


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: prepare_swift_sft.py INPUT_JSONL OUTPUT_JSONL")
    source, destination = map(Path, sys.argv[1:])
    if source.resolve() == destination.resolve():
        raise SystemExit("数据副本路径不能与原始数据相同")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    rows = calls_count = 0
    line_number = 0
    try:
        with source.open(encoding="utf-8-sig") as src, temporary.open("w", encoding="utf-8", newline="\n") as dst:
            for line_number, raw in enumerate(src, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                messages = row.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError("缺少非空 messages 数组")
                converted = []
                for message in messages:
                    role = message["role"]
                    content = message.get("content")
                    calls = message.get("tool_calls")
                    if calls:
                        if role != "assistant" or not isinstance(calls, list):
                            raise ValueError("tool_calls 必须是 assistant 消息中的数组")
                        if content:
                            converted.append({"role": "assistant", "content": content})
                        for call in calls:
                            function = call["function"]
                            arguments = function["arguments"]
                            if isinstance(arguments, str):
                                arguments = json.loads(arguments)
                            if not isinstance(arguments, dict) or not isinstance(function.get("name"), str):
                                raise ValueError("工具 name 必须为字符串，arguments 必须为 JSON object")
                            converted.append({"role": "tool_call", "content": json.dumps({
                                "name": function["name"], "arguments": arguments,
                            }, ensure_ascii=False)})
                            calls_count += 1
                    else:
                        if role not in {"system", "user", "assistant", "tool", "tool_response", "tool_call"}:
                            raise ValueError(f"不支持的消息角色：{role}")
                        if content is None:
                            raise ValueError(f"{role} 消息缺少 content")
                        converted.append({"role": role, "content": content})
                if not any(m["role"] in {"assistant", "tool_call"} and m["content"] for m in converted):
                    raise ValueError("没有可训练的 assistant/tool_call 内容")
                # tools 保存为 JSON 字符串，避免 Arrow 推断不同工具参数 schema 时不兼容。
                tools = row.get("tools") or SHOP_TOOL_SCHEMAS
                if isinstance(tools, str):
                    tools = json.loads(tools)
                if not isinstance(tools, list):
                    raise ValueError("tools 必须是工具定义数组")
                row["messages"] = converted
                row["tools"] = json.dumps(tools, ensure_ascii=False)
                dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                rows += 1
        if not rows:
            raise ValueError("数据没有有效记录")
        os.replace(temporary, destination)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise SystemExit(f"数据转换失败，第 {line_number} 行：{exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    print(f"准备完成：{rows} 条轨迹，转换 {calls_count} 次工具调用；原始数据未修改。", flush=True)


if __name__ == "__main__":
    main()
