

# TaskManager: 任务生成与管理
import numpy as np
from typing import List
from envs.core.json_manager import JsonManager
from envs.param import INTERRUPTION_PENALTY, MAX_NUM_TASKS
from envs.snapshot.task import Task

class TaskManager:
    def __init__(self, json_manager: JsonManager):
        self.jm = json_manager
        self.tasks: List[Task] = []
        self._generate()

    def _generate(self):
        node_keys = list(self.jm.nodes.keys())
        for _id in range(MAX_NUM_TASKS):
            idx = np.random.randint(0, len(node_keys))
            (_plane_at, _order_at) = node_keys[idx]
            _t = np.random.randint(0, 10)
            task_obj = Task(
                id=_id,
                layer_id=0,
                plane_at=_plane_at,
                order_at=_order_at,
                t_start=_t,
                t_end=_t,
                acted=0,
                workload_done=0,
                data_sent=0,
                is_done=False
            )
            # print(f"Generated Task ID {task_obj.id} start time {task_obj.t_start} end time {task_obj.t_end}")
            self.tasks.append(task_obj)

    def reset(self):
        self.tasks.clear()
        self._generate()

    def get_tasks_at(self, step: int) -> List[Task]:
        return [t for t in self.tasks if not t.t_start > step and not t.is_done]

    def get_tasks(self) -> List[Task]:
        return self.tasks
    
    def task_state(self, task: Task):
        
        # idle → anything allowed
        # under_processing → only allow: compute (5), no-op (0)
        # under_transferring → only allow: same direction transfer (1/2/3/4), no-op (0)
        # done → only allow: no-op (0)
        
        # --- Case 1: 已完成 ---
        if task.is_done:
            return "done"

        # --- Case 2: computing ---
        # 条件：上一动作是 compute 且已计算过部分
        if task.acted == 5 and task.workload_done > 0:
            return "under_processing"

        # --- Case 3: transferring ---
        # 条件：上一动作是传输且方向一致
        if task.acted in [1, 2, 3, 4] and task.data_sent > 0:
            return "under_transferring"

        # --- Case 4: idle ---
        return "idle"
    
    def build_action_mask(self, task: Task):
        mask = np.zeros(6, dtype=np.int8)

        state = self.task_state(task)

        if state == "idle":
            mask[:] = 1  # 全部可用

        elif state == "under_processing":
            mask[0] = 1  # no-op
            mask[5] = 1  # compute

        elif state == "under_transferring":
            prev = task.acted
            mask[0] = 1
            mask[prev] = 1   # 只能继续原方向传输

        elif state == "done":
            mask[0] = 1

        return mask

    def build_action_mask_for_tasks(self, tasks):
        """Return an array of shape (MAX_NUM_TASKS, 6) boolean mask for given tasks.
        Inactive slots (beyond provided tasks) will allow no-op (action 0) to avoid
        all-zero rows.
        """
        mask = np.zeros((MAX_NUM_TASKS, 6), dtype=bool)
        # default: allow no-op for all slots
        mask[:, 0] = True
        for i, task in enumerate(tasks):
            row = self.build_action_mask(task).astype(bool)
            mask[i, :] = row
        return mask

    def validate_action(self, task: Task, act: int):
        """
        Returns (valid: bool, penalty: float, truncated: bool, reason: str)
        """
        
        task_state = self.task_state(task)
        # if task.is_done:
        # # only no-op is allowed
        #     return (act == 0, WRONG_ACT_PENALTY, True, "done_task_illegal_action")

        if task_state == "under_processing":
            allowed = {0, 5}
            if act not in allowed:
                return (False, INTERRUPTION_PENALTY, True, "processing_interrupted")

        elif task_state == "under_transferring":
            allowed = {0, task.acted}
            if act not in allowed:
                return (False, INTERRUPTION_PENALTY, True, "transfer_interrupted")

        # else:  # idle
        #     allowed = {0, 1, 2, 3, 4, 5}

        return (True, 0.0, False, "None")




    
    