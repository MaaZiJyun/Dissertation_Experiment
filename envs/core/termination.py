from typing import List

from envs.param import STEP_PER_SLOT
from envs.snapshot.node import Node
from envs.snapshot.task import Task


def check_termination(nodes: List[Node], tasks: List[Task]):
    
    done = False
    success = False
    fail_reason = None
    
    is_all_done = all(t.is_done for t in tasks)
    
    if any(t.t_end >= STEP_PER_SLOT for t in tasks):
        done = True
        fail_reason = "time_limit"
        
    elif any(n.energy <= 0 for n in nodes):
        done = True
        fail_reason = "energy_depleted"
        
    elif is_all_done:
        done = True
        success = True
        
    return done, success, fail_reason
