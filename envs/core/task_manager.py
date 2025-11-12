

# TaskManager: 任务生成与管理
import numpy as np
from typing import List
from envs.core.topology_manager import TopologyManager
from envs.snapshot.task import Task

class TaskManager:
    def __init__(self, num_tasks: int, topology: TopologyManager):
        self.num_tasks = num_tasks
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
                t_end=0,
                is_done=False,
            )
            self.tasks.append(task_obj)

    def reset_tasks(self):
        self._load_tasks()

    def get_tasks(self) -> List[Task]:
        return self.tasks
