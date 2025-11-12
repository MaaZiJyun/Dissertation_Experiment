from typing import List

from envs.param import B_MAX, STEP_PER_SLOT, WEIGHT_DELAY, WEIGHT_ENERGY
from envs.snapshot.node import Node
from envs.snapshot.task import Task


def compute_delay_penalty(tasks: List[Task]):
    """
    计算任务平均延迟惩罚
    :param tasks: 任务列表，每个任务对象需要有 t_start、t_end、is_done 属性
    :param step_per_slot: 每个时间槽的步长
    :return: delay_penalty (float)
    """
    done_tasks = [t for t in tasks if t.is_done]
    if not done_tasks:
        return 0.0

    avg_delay = sum(max(t.t_end - t.t_start, STEP_PER_SLOT) for t in done_tasks) / len(done_tasks)
    delay_penalty = (STEP_PER_SLOT - avg_delay) / STEP_PER_SLOT
    return max(delay_penalty, 0.0)  # 防止出现负值


def compute_energy_penalty(nodes: List[Node]):
    """
    计算节点平均能量惩罚
    :param topology_manager: 包含节点信息的拓扑管理器对象
    :param b_max: 节点最大能量
    :return: energy_penalty (float)
    """
    
    avg_energy = sum(min(n.energy, B_MAX) for n in nodes) / len(nodes)
    energy_penalty = avg_energy / B_MAX
    return min(energy_penalty, 1.0)  # 限制在 [0, 1]


def compute_aim_reward( 
    delay_penalty: float, 
    energy_penalty: float, 
    ):

    ratio =  WEIGHT_DELAY * delay_penalty + WEIGHT_ENERGY * energy_penalty

    return 100 * ratio
