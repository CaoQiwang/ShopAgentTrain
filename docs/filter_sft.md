# 满分购物轨迹质量过滤

`scripts/filter_sft.py` 判断轨迹是否值得用于 SFT。奖励为 1 仅代表候选集合的来源，不能证明过程优秀。

脚本仅需 Python 3.10+ 标准库，调用本地服务的 `POST /v1/chat/completions`，无需安装 OpenAI SDK，也无需启动购物环境。

## 使用

把采集结果放在 `data/qwen3.5-397b-a17b_reward1_sft.jsonl`。每行至少包含 `{"messages": [...]}`，格式与 `collect_sft.py` 输出一致。

```bash
python scripts/filter_sft.py --model YOUR_SERVED_MODEL_NAME --model-base-url http://127.0.0.1:8041/v1
```

`YOUR_SERVED_MODEL_NAME` 是本地评审服务实际注册的模型名，与待过滤数据的生成模型名可以不同。默认地址与项目 `run_rollout.sh` 一致，也可以用 `MODEL_BASE_URL` 设置；模型名可用 `JUDGE_MODEL` 设置。API key 读取 `OPENAI_API_KEY`，默认 `EMPTY`。脚本不自动加载 `.env`，也不隐式使用可能指向外部服务的 `OPENAI_BASE_URL`。

默认输出：

- `data/qwen3.5-397b-a17b_sft_filtered.jsonl`：只包含优秀样本，保留原始 JSON 行、消息和额外字段，不加入审核信息。
- `data/qwen3.5-397b-a17b_sft_filtered.reviews.jsonl`：每条样本的评审状态、是否优秀、简短原因、问题类型及消息编号。

默认输入/输出路径以项目根目录为基准；显式传入的相对路径以当前工作目录为基准。

建议先抽查少量结果并校准评审模型。试跑使用不同输出路径，避免把子集当成完整数据：

```bash
python scripts/filter_sft.py --model YOUR_SERVED_MODEL_NAME --limit 20 --output-path data/filter_trial.jsonl
```

完整运行可以配置并发、超时、输出预算和重试：

```bash
python scripts/filter_sft.py --model YOUR_SERVED_MODEL_NAME --workers 4 --timeout 180 --max-tokens 2048 --retries 2
```

对于支持 `chat_template_kwargs` 的 Qwen/vLLM 服务，可按需关闭 thinking：

```bash
python scripts/filter_sft.py --model YOUR_SERVED_MODEL_NAME --model-extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

默认不强制启停 thinking；若开启 thinking，需要给 `--max-tokens` 足够预算。默认要求 JSON object 响应；服务不支持 `response_format` 时加 `--response-format none`，仍会通过提示词要求 JSON，并严格验证返回字段。

## 审核标准

先执行确定性检查，再调用模型做语义审核：

1. 规则复用 `src/env/tools.py` 和 `src/env/action_guard.py`，拦截未知工具、错误/多余参数、每轮非单工具调用、tool_call_id 错配、当前页面不可用动作、明确的智能体错误反馈及不完整购买轨迹。这些样本直接记录为不优秀，节省模型调用。
2. 模型仅使用按时间顺序记录的消息，核验购买前的商品证据、用户硬约束、选中规格和实际价格、探索过程、任务指令遵循情况。任何实质问题均不保留。

完整中文提示词位于脚本的 `JUDGE_SYSTEM_PROMPT`，包含具体审核步骤、问题分类、输出格式及六个边界示例。尤其强调：

- 满分、购买正确、购买后才出现的信息，都不能补足购买前的证据缺口。
- 搜索词、任务回显和助手自己的猜测，不是商品属性证据。
- 非法工具调用即使后来被环境纠正、最终成功，仍不保留。
- 不强制查看每个信息页、比较多个商品、写解释或执行固定步数；短而有据的轨迹可以优秀。
- 合理搜索改词、探索不符候选、正常返回详情页，不等于无效循环。
- 不因为商品描述包含 `error` 等单词就判定环境纠错。

工具约定针对当前项目的 ShopSimulator 结构化 `messages/tool_calls` 轨迹，不适用于直接存储 `search[...]` 文本动作的其他格式。模型评审仍需人工抽查，尤其是“软偏好”“预算左右”等边界判断。

## 评审记录

例如：

```json
{"line_number":12,"input_sha256":"...","review_signature":"...","status":"reviewed","source":"llm","is_excellent":false,"reason":"购买前未确认目标容量规格的实际价格，无法证明满足预算","issues":[{"code":"unverified_variant_price","message_index":8,"detail":"m8直接购买；此前仅显示价格区间，未展示所选大容量规格价格"}]}
```

`line_number` 从 1 开始，对应源 JSONL 的物理行；`message_index` 从 0 开始，对应 `messages` 下标。空行跳过，非空坏 JSON 单独报错。

`source=rule` 表示规则拦截，`source=llm` 表示模型评审。`status=error`、`is_excellent=null` 表示服务故障/响应格式无效/输入无法读取等导致评审未完成，不等于不优秀。程序退出码为 0 表示本次选定范围完成，1 表示存在评审失败，2 表示输入/参数/文件错误，130 表示中断。

脚本打印保留数、淘汰数、失败数、缓存数以及各类缺陷数量。缺陷计数可重叠，因为一条样本可能有多个缺陷。

## 续跑与错误处理

```bash
python scripts/filter_sft.py --model YOUR_SERVED_MODEL_NAME --resume
```

保留原来的输入、输出、服务及模型参数，只增加 `--resume`。评审日志逐条写入并刷新；续跑根据源行号、内容哈希和评审配置签名复用已完成结果，失败项重新请求。修改样本、模型、服务地址、提示词/规则代码或生成参数后，相应缓存不再复用。若相同服务名背后的模型权重变了，使用 `--overwrite` 强制重新评审。

续跑日志为追加记录，同一行重试后可以有多条记录，使用匹配签名/内容的最新成功记录。脚本可修复中断留下的日志尾部半行，中间损坏会报错。

过滤输出始终按输入顺序重新生成；只有本次选定范围的每条记录都处理成功，才原子替换正式输出。若有评审失败，已有训练文件保持原样，首次运行则不会产生正式过滤文件，详见 `output_written`。使用 `--overwrite` 会重新建立评审日志；不要同时运行多个进程写同一组路径。

轨迹全量发送，不做自动截断。上下文过长、HTTP 错误、超时、JSON 无效、结论/缺陷矛盾、消息编号越界、输出被截断等都不会放行样本。网络临时故障和响应格式错误按 `--retries` 重试；HTTP 400/401/403/404 等配置错误直接记为评审失败。修复上下文/服务配置后续跑即可。

## 离线测试

```bash
python -m unittest discover -s tests -p test_filter_sft.py -v
```

测试覆盖真实本地 HTTP 请求协议、截断回复重试、错误结论拒绝、环境纠错、原始数据保留、语义淘汰、服务失败不替换输出，以及续跑缓存失效。测试不调用实际模型，因此不代表真实模型的审核准确率。
