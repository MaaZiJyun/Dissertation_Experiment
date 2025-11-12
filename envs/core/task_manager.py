

# TaskManager: 任务生成与管理
import numpy as np
from typing import List
from envs.core.topology_manager import TopologyManager
from envs.param import NUM_TASKS
from envs.snapshot.task import Task

class TaskManager:
    def __init__(self, topology: TopologyManager):
        self.num_tasks = NUM_TASKS
        self.topology = topology
        self.tasks: List[Task] = []
        self._load_tasks()

    def _load_tasks(self):
        self.tasks = []
        node_keys = list(self.topology.nodes.keys())
        for _id in range(self.num_tasks):
            idx = np.random.randint(0, len(node_keys))
            (_plane_at, _order_at) = node_keys[idx]
            _t_start = np.random.randint(0, 5)
            task_obj = Task(
                id=_id,
                layer_id=0,
                layer_process=0,
                data_sent =0,
                plane_at=_plane_at,
                order_at=_order_at,
                t_start=_t_start,
                t_end=_t_start,
                is_done=False,
            )
            self.tasks.append(task_obj)

    def reset_tasks(self):
        self._load_tasks()
        
    def get_tasks(self, t: int) -> List[Task]:
        # 已经完成的任务或者未开始的任务不处理
        return [task for task in self.tasks if task.t_start <= t]