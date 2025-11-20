from typing import Dict, Tuple, List
from envs.IO.decision_manager import DecisionManager
from envs.IO.state_manager import StateManager
from envs.param import LAYER_OUTPUT_DATA_SIZE, STEP_PER_SECOND, STEP_PER_SLOT, T_SLOT, TRANS_COMPLETION_REWARD
from envs.snapshot.request import TransReq
from envs.snapshot.task import Task

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
        
        target_data_to_send = LAYER_OUTPUT_DATA_SIZE[n]
        
        task = next((task for task in tasks if task.id == m), None)
        if task is None:
            continue

        # 获取当前通信速率
        comm_capacity = sm.get_comm(u=src, v=dst)
        if comm_capacity == 0:
            continue
        
        # 分配带宽比例
        users_uv = dm.get_rho_by_uv(u=src, v=dst)
        users_vu = dm.get_rho_by_uv(u=dst, v=src)
        users = {**users_uv, **users_vu}
        
        # 计算带宽分配比例
        sum_of_data = sum(LAYER_OUTPUT_DATA_SIZE[n] for (m, n), value in users.items() if value)
        bandwidth_ratio = target_data_to_send / sum_of_data if sum_of_data > 0 else 0

        # 计算每步可传输的数据量
        data_per_step = comm_capacity / STEP_PER_SECOND * bandwidth_ratio

        # 更新传输进度
        req.data_sent += data_per_step
        sm.write_size(m=m, n=n, value=data_per_step)
        task.data_sent += data_per_step

        # 当前已经传输的数据量
        sent_data = sm.sum_size_before(m=m, n=n, T=t)
        
        # if m == 0:
        #     print(f"Task {m} Layer {n} Workload Done: {task.data_sent}, Sum Workload Before T={t}: {sent_data}, Target: {target_data_to_send}")

        # 检查是否完成传输
        if sent_data >= target_data_to_send:
            # 传输完成，更新任务位置 (write destination coordinates)
            sm.write_location(m=m, value=dst)
            task.plane_at, task.order_at = dst
            task.data_sent = 0.0
            trans_reqs.remove(req)

            rewards += TRANS_COMPLETION_REWARD

    return rewards
