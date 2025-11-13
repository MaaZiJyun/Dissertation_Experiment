from typing import List

from envs.param import B_MAX, STEP_PER_SLOT, WEIGHT_DELAY, WEIGHT_ENERGY
from envs.snapshot.node import Node
from envs.snapshot.task import Task


def compute_delay_penalty(tasks: List[Task]):
    done_tasks = [t for t in tasks if t.is_done]
    
    if not done_tasks:
        return 0.0

    avg_delay = sum(min(t.t_end - t.t_start, STEP_PER_SLOT) for t in done_tasks) / len(done_tasks)
    delay_penalty = (STEP_PER_SLOT - avg_delay) / STEP_PER_SLOT
    return max(delay_penalty, 0.0)  # 防止出现负值


def compute_energy_penalty(nodes: List[Node]):
    avg_energy = sum(min(n.energy, B_MAX) for n in nodes) / len(nodes)
    energy_penalty = avg_energy / B_MAX
    return min(energy_penalty, 1.0)  # 限制在 [0, 1]


def compute_aim_reward( 
    delay_penalty: float, 
    energy_penalty: float, 
    ):

    ratio =  WEIGHT_DELAY * delay_penalty + WEIGHT_ENERGY * energy_penalty

    return 100 * ratio
