# ShopSimulator 多轮 GRPO 适配

适配入口为 `swift_plugin.py`，其中 `ShopMultiTurnScheduler` 直接继承 ms-swift 的 `MultiTurnScheduler`，实现轨迹开始、每轮结束与下一轮请求构造，并注册为 `shop_tool_scheduler`。它对应当前仓库的 ms-swift 4.4.2 和 Qwen3.5 工具格式。每个 rollout 使用 `env_config.task_id` 重置独立的 `ShopEnv`，模型每轮生成一个工具调用，环境执行后返回最新 observation。购买、主动结束、达到 `max_turns`、生成截断以及环境异常时都会调用 `ShopEnv.close()` 归还槽位。server 模式的推理异常也会清理槽位。无效调用会反馈给模型，供下一轮纠正。

从仓库根目录启动，并使仓库与 ms-swift 可导入：

```bash
export PYTHONPATH="$PWD/ms-swift:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SHOP_ENV_URL=http://127.0.0.1:5000
```

在 `swift rollout` 和 `swift rlhf --rlhf_type grpo` 两侧都加载插件并设置：

```text
--external_plugins src/train/swift_plugin.py
--agent_template qwen3_5
--enable_thinking false
--multi_turn_scheduler shop_tool_scheduler
--gym_env shop_simulator
--use_gym_env true
--max_turns 40
```

训练侧使用 `--dataset data/shop_grpo_tasks_10000.jsonl`。server 模式还需要 `--use_vllm true --vllm_mode server` 及对应的 vLLM 服务地址；colocate 模式设置 `--vllm_mode colocate`。rollout 与训练应使用相同的 Qwen3.5 模型、模板和 thinking 设置。`--max_completion_length` 控制**每轮**生成长度，40 轮的上下文还需要足够的 `--max_length` / `--vllm_max_model_len`。

当前通过 `src/env/result_reward.py` 对模拟器终局结果计算奖励：购买后 `r_type`、`r_att`、`r_option`、`r_price` 全部为 1 给 1；`r_type`、`r_att`、`r_price` 均为 1，但 `r_option` 不为 1 时给 0.3；其余情况给 0。调度器将每步结果累加为 `rollout_infos.total_reward`，交给 `--use_gym_env true` 使用。若以后要改用独立的 ms-swift 奖励函数，应显式设置 `--use_gym_env false`；若同时启用两者，ms-swift 会把环境奖励与自定义奖励一起计入。模拟器同时可用的槽位数有限；并发轨迹数应在服务容量内。

colocate 模式在推理引擎直接抛异常且尚未进入 `on_turn_end` 时，ms-swift 4.4.2 不会调用调度器的清理钩子。需要严格保证这类失败立即归还槽位时，请使用 server 模式。
