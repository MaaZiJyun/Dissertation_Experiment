from typing import Dict, Tuple, List
from envs.IO.decision_manager import DecisionManager
from envs.IO.state_manager import StateManager
from envs.param import LAYER_OUTPUT_DATA_SIZE, STEP_PER_SECOND, STEP_PER_SLOT, T_SLOT, TRANS_COMPLETION_REWARD
from envs.snapshot.request import TransReq
from envs.snapshot.task import Task
from envs.snapshot.edge import Edge

def process_transfers(
    trans_reqs: List[TransReq], 
    edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Edge], 
    tasks: List[Task]
):
    edge_transfers: Dict[Tuple[Tuple[int,int],Tuple[int,int]], List[Tuple[Task, float]]] = {}
    rewards = {}
    for req in trans_reqs:
        
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

        capacity = edge.rate / STEP_PER_SECOND

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


def do_transferring(
    tasks: List[Task],
    trans_reqs: List[TransReq], 
    sm: StateManager,
    dm: DecisionManager,
    t: int,
):
    rewards = 0.0
    
    for req in trans_reqs: 
        m = req.task_id
        n = sm.get_progress(m)
        src = sm.get_location(m)
        dst = req.dst
        target = LAYER_OUTPUT_DATA_SIZE[n]
        
        task = next((task for task in tasks if task.id == m), None)
        if task is None:
            continue

        # 获取当前通信速率
        comm_rate = sm.get_comm(u=src, v=dst)
        if comm_rate == 0:
            continue
        
        # 分配带宽比例
        users = dm.get_rho_by_uv(u=src, v=dst)
        sum_of_data = sum(LAYER_OUTPUT_DATA_SIZE[n] for (m, n), value in users.items() if value)
        bandwidth_allocation = target / sum_of_data if sum_of_data > 0 else 0

        # 计算每步可传输的数据量
        data_per_step = comm_rate / STEP_PER_SECOND * bandwidth_allocation

        # 更新传输进度
        req.data_sent += data_per_step
        sm.write_size(m=m, n=n, value=data_per_step)
        task.data_sent += data_per_step

        # 当前已经传输的数据量
        sent_data = sm.sum_size_before(m=m, n=n, T=t)

        # 检查是否完成传输
        if sent_data >= target:
            # 传输完成，更新任务位置 (write destination coordinates)
            sm.write_location(m=m, value=dst)
            task.plane_at, task.order_at = dst
            task.data_sent = 0.0

            rewards += TRANS_COMPLETION_REWARD

    return rewards
