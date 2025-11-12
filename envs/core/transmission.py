from typing import Dict, Tuple, List
from envs.param import STEP_PER_SLOT
from envs.snapshot.request import TransReq
from envs.snapshot.task import Task
from envs.snapshot.edge import Edge

def process_transfers(
    transfer_reqs: List[TransReq], 
    edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Edge], 
    tasks: List[Task]
):
    edge_transfers: Dict[Tuple[Tuple[int,int],Tuple[int,int]], List[Tuple[Task, float]]] = {}
    rewards = {}
    for req in transfer_reqs:
        
        task = req.task_id
        src = req.src
        dst = req.dst
        target_file_size = req.target_file_size
        
        task = next((t for t in tasks if t.id == task), None)
        if task is None:
            continue

        edge_key = (src, dst)
        edge_transfers.setdefault(edge_key, []).append((task, target_file_size))

    for edge_key, task_list in edge_transfers.items():
        edge = edges.get(edge_key)
        
        if not edge or edge.rate is None:
            continue

        capacity = edge.rate / STEP_PER_SLOT

        if capacity is None:
            continue
        
        total_data = sum(bits for _, bits in task_list)
        
        for task, target_file_size in task_list:
            
            # 分配比例
            r = target_file_size / total_data
            
            # 分配容量
            c = capacity * r
            task.data_sent  += c
            
            if task.data_sent  >= target_file_size:
                task.plane_at, task.order_at = edge_key[1]
                task.data_sent  = 0
                rewards[task.id] = 'complete'
            else:
                # remain_ratio = 1 - c / target_file_size
                rewards[task.id] = 'incomplete'
    return rewards
