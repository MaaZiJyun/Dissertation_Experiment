from envs.snapshot.task import Task

def process_computation(task: Task, layer_process_step_cost, num_layers, global_time_counter):
    task.workload_done += 1
    if task.workload_done >= layer_process_step_cost[task.layer_id]:
        task.layer_id += 1
        task.workload_done = 0
        if task.layer_id >= num_layers:
            task.t_end = global_time_counter
            task.is_done = True
            return 'done'
        return 'layer_complete'
    return 'in_progress'
