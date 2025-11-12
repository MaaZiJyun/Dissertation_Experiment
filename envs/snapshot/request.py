from pydantic import BaseModel
from envs.snapshot.task import Task

class TransReq(BaseModel):
    task_id: int
    src: tuple[int, int]
    dst: tuple[int, int]
    target_file_size: float
    data_sent: float = 0.0
    step: int = 0

class CompReq(BaseModel):
    task_id: int
    node_id: int
    layer_id: int
    target_workload: float
    workload_done : float = 0.0
    step: int = 0
    
    