from pydantic import BaseModel

"""
Each task record h = {id, layer_id, plane_at, order_at, t_start, t_end} 
- id: task index
- layer_id: current layer index
- plane_at: plane index
- order_at: satellite index
- t_start: float
- t_end: float
"""

class Task(BaseModel):
    id: int
    act: int = 0
    layer_id: int
    plane_at: int
    order_at: int
    t_start: float
    t_end: float
    is_done: bool
    
    # functional counters
    workload_done: int = 0 # to record computation progress
    data_sent : int = 0 # to record transmission progress