#!/usr/bin/env python3
"""使用自定义 exact-match 奖励重复采集 RL rollout 分数。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_rl_rewards import CUSTOM_REWARD, main


if __name__ == "__main__":
    raise SystemExit(main(reward_mode=CUSTOM_REWARD))
