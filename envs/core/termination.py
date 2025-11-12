from typing import List

from envs.param import OVERTIME_PENALTY, STEP_PER_SLOT, TASK_COMPLETION_REWARD
from envs.snapshot.node import Node
from envs.snapshot.task import Task


def check_termination(
    terminated: bool, 
    truncated: bool, 
    action_reward: float, 
    nodes: List[Node], 
    tasks: List[Task]
    ):

    fail_reason = None
    reward = action_reward

    if any(t.t_end >= STEP_PER_SLOT for t in tasks):
        truncated = True
        reward = OVERTIME_PENALTY
        fail_reason = "over_time"

    if any(n.energy <= 0 for n in nodes):
        terminated = True
        reward = OVERTIME_PENALTY
        fail_reason = "energy_depleted"

    elif all(t.is_done for t in tasks):
        terminated = True
        reward = TASK_COMPLETION_REWARD

    return terminated, truncated, fail_reason, reward
