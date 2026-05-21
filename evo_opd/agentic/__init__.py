"""Agentic-OPD: ReAct rollouts + reward composition.

  - trajectory.py: trajectory dataclass + per-token role tracking
  - rollout.py:    ReAct rollout loop with tool dispatch
  - rewards.py:    composite reward (lineage + struct + arena + efficiency)
"""
from .trajectory import Trajectory, ActionStep, TokenSpan
from .rollout import run_rollout, ROLLOUT_SYS_PROMPT
from .rewards import compute_trajectory_reward, AgenticRewardConfig

__all__ = [
    "Trajectory", "ActionStep", "TokenSpan",
    "run_rollout", "ROLLOUT_SYS_PROMPT",
    "compute_trajectory_reward", "AgenticRewardConfig",
]
