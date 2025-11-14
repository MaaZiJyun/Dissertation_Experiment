from typing import List

from envs.IO.decision_manager import DecisionManager
from envs.IO.state_manager import StateManager
from envs.param import STEP_PER_SLOT
from envs.snapshot.task import Task
import numpy as np


def all_tasks_overtimed(tasks: List[Task]):
    result = False
    if len(tasks) > 0:
        result = all((t.t_end - t.t_start) > STEP_PER_SLOT for t in tasks)
    return result

def any_illegal_link(sm: StateManager, dm: DecisionManager):
    result = False
    
    if sm.is_empty() or dm.is_empty():
        return result
    
    operating_links = np.argwhere(dm.rho != 0)

    for up, uo, vp, vo, m_idx, n_idx in operating_links:
        u = (int(up), int(uo))
        v = (int(vp), int(vo))
        if sm.get_comm(u, v) == 0:
            result = True
            
    return result
