from typing import List
from envs.IO.decision_manager import DecisionManager
from envs.IO.state_manager import StateManager
from envs.param import LAYER_COMPLETION_REWARD, LAYER_PROCESS_STEP_COST, MAX_NUM_LAYERS, TASK_COMPLETION_REWARD, TASK_COMPLETION_REWARD
from envs.snapshot.request import CompReq
from envs.snapshot.task import Task

def do_computing(
    comp_reqs: List[CompReq],
    tasks: List[Task],
    sm: StateManager,
    dm: DecisionManager,
    t: int,
):
    rewards = 0.0
    
    for req in comp_reqs: 
        m = req.task_id
        n = sm.get_progress(m)
        target = LAYER_PROCESS_STEP_COST[n]
        
        task = next((task for task in tasks if task.id == m), None)
        if task is None:
            continue

        # 更新计算进度
        sm.write_workload(m=m, n=n, value=1)
        task.workload_done += 1

        # 当前已经计算的量
        sent_data = sm.sum_workload_before(m=m, n=n, T=t)

        # if n == 0:
        #     print(f"Task {m} Layer {n} Workload Done: {task.workload_done}, Sum Workload Before T={t}: {sent_data}, Target: {target}")

        # 检查是否完成计算
        if sent_data >= target:
        # if task.workload_done >= target:
            # 计算完成，更新任务位置
            sm.write_progress(m=m, value=n+1)
            task.layer_id += 1
            task.workload_done = 0
            rewards += LAYER_COMPLETION_REWARD
            
            if task.layer_id >= MAX_NUM_LAYERS:
                task.is_done = True
                rewards += TASK_COMPLETION_REWARD

    return rewards
