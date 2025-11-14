from typing import Dict
from pydantic import BaseModel

"""
Each task record Z = {m, n, p, o, start, end} 
- m: task index
- n: layer index
- p: plane index
- o: satellite index
- start: int
- end: int
"""

class Task(BaseModel):
    id: int
    layer_id: int
    plane_at: int
    order_at: int
    t_start: int
    t_end: int
    
    # functional counters
    act: int = 0 # to record action taken
    workload_done: int = 0 # to record computation progress
    data_sent : int = 0 # to record transmission progress
    is_done: bool = False