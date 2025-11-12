from typing import List

from envs.param import STEP_PER_SLOT
from envs.snapshot.node import Node
from envs.snapshot.task import Task


def all_tasks_overtimed(tasks: List[Task]):
    """
    检查所有任务是否超时
    :param tasks: 任务列表，每个任务对象需要有 is_done 属性
    :return: 所有任务是否超时 (bool)
    """
    return all(t.t_end > STEP_PER_SLOT for t in tasks)

# def any_wrong_link(tasks: List[Task], t: int, decision_rho):
#     """
#     检查是否有错误链接
#     :param decision_rho: 决策 Rho 字典
#     :return: 是否有错误链接 (bool)
#     """
#     links = [decision_rho.get_uv_by_tm_true(t=t, m=task.id) for task in tasks]
    