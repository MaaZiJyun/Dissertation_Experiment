from envs.param import LAYER_PROCESS_STEP_COST, NUM_LAYERS
from envs.snapshot.task import Task

def process_computation(task: Task):
    
    state = 'in_progress'
    task.workload_done += 1

    if task.workload_done >= LAYER_PROCESS_STEP_COST[task.layer_id]:
        
        task.layer_id += 1
        task.workload_done = 0
        state = 'layer_complete'
        
        if task.layer_id >= NUM_LAYERS:
            
            task.is_done = True
            state = 'done'
        
    
    return state
