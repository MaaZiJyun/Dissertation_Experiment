import numpy as np
from typing import Dict, List, Tuple

from envs.param import MAX_NUM_LAYERS, MAX_NUM_TASKS
from envs.snapshot.edge import Edge
from envs.snapshot.node import Node
from envs.snapshot.task import Task

class StateManager:
    def __init__(self, p_max: int, o_max: int):
        # active task count
        self._M = 0
        # history buffer for snapshots
        self._beta: Dict[int, Dict[str, np.ndarray]] = {}

        # size limits
        self.P_MAX = p_max
        self.O_MAX = o_max
        self.N_MAX = MAX_NUM_LAYERS
        self.M_MAX = MAX_NUM_TASKS

        # For topology (full-shape)
        self.energy = np.zeros((self.P_MAX, self.O_MAX), dtype=np.float32)
        self.sunlight = np.zeros((self.P_MAX, self.O_MAX), dtype=np.int8)
        self.comm = np.zeros((self.P_MAX, self.O_MAX, self.P_MAX, self.O_MAX), dtype=np.float32)

        # For tasks: preallocate using M_MAX (active task count is tracked by self._M)
        self.location = np.zeros((self.M_MAX, 2), dtype=np.int32)
        self.progress = np.zeros(self.M_MAX, dtype=np.int32)
        self.size = np.zeros((self.M_MAX, self.N_MAX), dtype=np.float32)
        self.workload = np.zeros((self.M_MAX, self.N_MAX), dtype=np.int32)


    def setup(self, all_nodes: List[Node], all_edges: List[Edge], all_tasks: List[Task]):
        
        for n in all_nodes:
            pp, oo = n.plane_id, n.order_id
            self.energy[pp, oo] = n.energy
            self.sunlight[pp, oo] = n.gamma

        for e in all_edges:
            src_p, src_o = e.u.plane_id, e.u.order_id
            dst_p, dst_o = e.v.plane_id, e.v.order_id
            self.comm[src_p, src_o, dst_p, dst_o] = e.rate
            self.comm[dst_p, dst_o, src_p, src_o] = e.rate

        for t in all_tasks:
            self.location[t.id, 0] = t.plane_at
            self.location[t.id, 1] = t.order_at
            self.progress[t.id] = t.layer_id
            
    def update(self, current_task_length: int):
        self._M = current_task_length
        
    def report(self, step: int) -> Dict[str, np.ndarray]:
        beta_t = self._to_beta()
        self._beta[step] = beta_t
        return beta_t

    def reset(self):
        self.energy.fill(0)
        self.sunlight.fill(0)
        self.comm.fill(0)
        self.location.fill(0)
        self.progress.fill(0)
        self.size.fill(0)
        self.workload.fill(0)

    def write_energy(self, p: int, o: int, value: float):
        self.energy[(p, o)] = value
        
    def write_sunlight(self, p: int, o: int, value: int):
        self.sunlight[(p, o)] = value
        
    def write_comm(self, u: Tuple[int, int], v: Tuple[int, int], value: float):
        up, uo = u
        vp, vo = v
        self.comm[up, uo, vp, vo] = value
        
    def write_location(self, m: int, value: Tuple[int, int]):
        self.location[m,0] = value[0]
        self.location[m,1] = value[1]

    def write_progress(self, m: int, value: int):
        self.progress[m] = value
        
    def write_size(self, m: int, n: int, value: float):
        # accumulate size within the current step (per-step increments)
        self.size[m, n] = float(self.size[m, n]) + float(value)
            
    def write_workload(self, m: int, n: int, value: int):
            # accumulate workload within the current step (per-step increments)
        self.workload[m, n] = int(self.workload[m, n]) + int(value)

    def clear_step_counters(self):
        """Clear per-step increment counters (call at the start of each env.step)."""
        # workload and size may represent per-step increments
        try:
            self.workload.fill(0)
            self.size.fill(0.0)
        except Exception:
            pass
        
    def get_comm(self, u: Tuple[int, int], v: Tuple[int, int]) -> float:
        up, uo = u
        vp, vo = v
        return float(self.comm[up, uo, vp, vo])
    
    def get_size(self, m: int, n: int) -> float:
        return self.size[(m, n)]
    
    def get_progress(self, m: int) -> int:
        return self.progress[m]
    
    def get_location(self, m: int) -> Tuple[int, int]:
        return tuple(self.location[m])
    
    def _to_beta(self) -> Dict[str, np.ndarray]:
        """Return a snapshot of current state (beta) with topology full-shape
        and task arrays limited to the first self._M active tasks.
        """
        # topology arrays (full shape)
        valid_energy = self.energy.copy()
        valid_sunlight = self.sunlight.copy()
        valid_comm = self.comm.copy()

        # task arrays: take first self._M rows
        if self._M <= 0:
            # return empty task arrays with shape (0,...)
            valid_location = np.zeros((0, 2), dtype=self.location.dtype)
            valid_progress = np.zeros((0,), dtype=self.progress.dtype)
            valid_size = np.zeros((0, self.N_MAX), dtype=self.size.dtype)
            valid_workload = np.zeros((0, self.N_MAX), dtype=self.workload.dtype)
        else:
            valid_location = self.location[: self._M].copy()
            valid_progress = self.progress[: self._M].copy()
            valid_size = self.size[: self._M].copy()
            valid_workload = self.workload[: self._M].copy()

        beta_t = {
            "energy": valid_energy,
            "sunlight": valid_sunlight,
            "comm": valid_comm,
            "location": valid_location,
            "progress": valid_progress,
            "size": valid_size,
            "workload": valid_workload,
        }

        return beta_t

    def sum_size_before(self, m: int, n: int, T: int) -> float:
        """
        获取t时间之前所有size[m, n]历史值，返回float字典。
        """
        result = 0.0
        # include time step T (sum of increments up to and including T)
        for t in range(T + 1):
            beta = self._beta.get(t)
            if beta is None:
                continue
            arr = beta.get("size")
            if arr is None:
                continue
            # ensure indices are in range
            if getattr(arr, 'ndim', 0) >= 2 and 0 <= m < arr.shape[0] and 0 <= n < arr.shape[1]:
                result += float(arr[m, n])
        # if current step T is not yet recorded in _beta (called during step), include current buffer
        if self._beta.get(T) is None:
            try:
                if 0 <= m < self.size.shape[0] and 0 <= n < self.size.shape[1]:
                    result += float(self.size[m, n])
            except Exception:
                pass
        return result

    def sum_workload_before(self, m: int, n: int, T: int) -> int:
        """
        获取t时间之前所有workload[m, n]历史值，返回int字典。
        """
        result = 0
        # include time step T (sum of increments up to and including T)
        for t in range(T + 1):
            beta = self._beta.get(t)
            if beta is None:
                continue
            arr = beta.get("workload")
            if arr is None:
                continue
            # ensure indices are in range
            if getattr(arr, 'ndim', 0) >= 2 and 0 <= m < arr.shape[0] and 0 <= n < arr.shape[1]:
                result += int(arr[m, n])
        # if current step T is not yet recorded in _beta (called during step), include current buffer
        if self._beta.get(T) is None:
            try:
                if 0 <= m < self.workload.shape[0] and 0 <= n < self.workload.shape[1]:
                    result += int(self.workload[m, n])
            except Exception:
                pass
        return result
    
    def is_empty(self) -> bool:
        """
        检查状态管理器是否为空（没有初始化数据）。
        :return: 是否为空 (bool)
        """
        return self._M == 0
