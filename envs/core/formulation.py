from typing import List

from envs.param import B_MAX, STEP_PER_SLOT, WEIGHT_DELAY, WEIGHT_ENERGY
from envs.snapshot.node import Node
from envs.snapshot.task import Task


def compute_delay_penalty(tasks: List[Task]):
    done_tasks = [t for t in tasks if t.is_done]
    
    if not done_tasks:
        return 0.0
    
    if STEP_PER_SLOT == 0:
        return 0.0

    max_time_cost = max(t.t_end - t.t_start for t in done_tasks)
    delay_penalty = max_time_cost / STEP_PER_SLOT
    return max(delay_penalty, 0.0)  # 防止出现负值


def compute_energy_penalty(nodes: List[Node]):
    if not nodes:
        return 0.0

    if B_MAX == 0:
        return 0.0

    max_energy_cost = max(B_MAX - n.energy for n in nodes)
    energy_penalty = max_energy_cost / B_MAX
    return min(energy_penalty, 1.0)  # 限制在 [0, 1]


def compute_aim_reward( 
    delay_penalty: float, 
    energy_penalty: float, 
    ):

    ratio =  WEIGHT_DELAY * delay_penalty + WEIGHT_ENERGY * energy_penalty

    return 100 * ratio

def settle_reward(delay_penalty: float, energy_penalty: float, scale: float = 100.0) -> float:
    """
    Lexicographic objective: prioritize one objective first, break ties with the other.

    - If primary == 'delay': minimize delay first (maximize 1-delay), then among equal delay
      prefer lower energy. Implemented as a large weight on the primary objective plus
      a small secondary bonus.
    - Reverse if primary == 'energy'.
    """
    d = min(max(delay_penalty, 0.0), 1.0)
    e = min(max(energy_penalty, 0.0), 1.0)
    return scale * (WEIGHT_ENERGY * (1.0 - e)) + (WEIGHT_DELAY * (1.0 - d))
