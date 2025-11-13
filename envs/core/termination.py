from typing import List

from envs.snapshot.node import Node
from envs.snapshot.task import Task


def all_tasks_completed(tasks: List[Task]):
    """
    检查所有任务是否完成
    :param tasks: 任务列表，每个任务对象需要有 is_done 属性
    :return: 所有任务是否完成 (bool)
    """
    result = False
    if len(tasks) > 0:
        result = all(t.is_done for t in tasks)
    return result

def any_satellite_depleted(nodes: List[Node]):
    """
    检查是否有卫星能量耗尽
    :param nodes: 节点列表，每个节点对象需要有 energy 属性
    :return: 是否有卫星能量耗尽 (bool)
    """
    result = False
    if len(nodes) > 0:
        result = any(n.energy <= 0 for n in nodes)
    return result

