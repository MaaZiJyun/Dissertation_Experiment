

# TaskManager: 任务生成与管理
import numpy as np
from typing import List
from envs.core.json_manager import JsonManager
from envs.param import MAX_TASKS
from envs.snapshot.task import Task

class TaskManager:
    def __init__(self, json_manager: JsonManager):
        self.jm = json_manager
        self.tasks: List[Task] = []
        self._generate()

    def _generate(self):
        node_keys = list(self.jm.nodes.keys())
        for _id in range(MAX_TASKS):
            idx = np.random.randint(0, len(node_keys))
            (_plane_at, _order_at) = node_keys[idx]
            _t_start = np.random.randint(0, 5)
            task_obj = Task(
                id=_id,
                layer_id=0,
                plane_at=_plane_at,
                order_at=_order_at,
                t_start=0,
                t_end=0,
            )
            # print(f"Generated Task ID {task_obj.id} start time {task_obj.t_start} end time {task_obj.t_end}")
            self.tasks.append(task_obj)

    def reset(self):
        self.tasks.clear()
        self._generate()

    def get_tasks_at(self, step: int) -> List[Task]:
        return [t for t in self.tasks if not t.t_start <= step]
    
    def get_tasks(self) -> List[Task]:
        return self.tasks
    
    