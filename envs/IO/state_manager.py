import numpy as np
from typing import Dict, List, Tuple

from envs.param import LAYER_OUTPUT_DATA_SIZE, LAYER_PROCESS_STEP_COST, NUM_LAYERS
from envs.snapshot.edge import Edge
from envs.snapshot.node import Node
from envs.snapshot.task import Task

class StateManager:
    def __init__(self):

        self.P = 0
        self.O = 0
        self.M = 0
        self.N = NUM_LAYERS
        self.beta_history = {}
        
        
    def initialize(self, p: int, o: int, nodes: List[Node], edges: List[Edge], tasks: List[Task]):
        self.P = p
        self.O = o
        self.M = len(tasks)
        self.N = NUM_LAYERS

        self.energy = np.zeros((self.P, self.O), dtype=np.float32)
        self.sunlight = np.zeros((self.P, self.O), dtype=np.int8)
        # comm is a 4-D array indexed by [src_p, src_o, dst_p, dst_o]
        self.comm = np.zeros((self.P, self.O, self.P, self.O), dtype=np.float32)
        # location: for each task store (plane, order)
        self.location = np.zeros((self.M, 2), dtype=np.int32)
        self.progress = np.zeros(self.M, dtype=np.int32)
        self.size = np.zeros((self.M, self.N), dtype=np.float32)
        self.workload = np.zeros((self.M, self.N), dtype=np.int32)

        for n in nodes:
            pp, oo = n.plane_id, n.order_id
            self.energy[pp, oo] = n.energy
            self.sunlight[pp, oo] = n.gamma

        for e in edges:
            src_p, src_o = e.u.plane_id, e.u.order_id
            dst_p, dst_o = e.v.plane_id, e.v.order_id
            self.comm[src_p, src_o, dst_p, dst_o] = e.rate

        for t in tasks:
            if t.id < self.M:
                self.location[t.id, 0] = t.plane_at
                self.location[t.id, 1] = t.order_at
                self.progress[t.id] = t.layer_id
            
    def reset(self):
        self.beta_history.clear()

    def update(self, step: int):
        self.beta_history[step] = self._to_beta()
        
    def write_energy(self, p: int, o: int, value: float):
        self.energy[(p, o)] = value
        
    def write_sunlight(self, p: int, o: int, value: int):
        self.sunlight[(p, o)] = value
        
    def write_comm(self, u: Tuple[int, int], v: Tuple[int, int], value: float):
        up, uo = u
        vp, vo = v
        self.comm[up, uo, vp, vo] = value
        
    def write_location(self, m: int, value: Tuple[int, int]):
        self.location[m] = value
        
    def write_progress(self, m: int, value: int):
        self.progress[m] = value
        
    def write_size(self, m: int, n: int, value: float):
        self.size[(m, n)] = value
        
    def write_workload(self, m: int, n: int, value: int):
        self.workload[(m, n)] = value
        
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
        beta_t = {
            "energy": self.energy.copy(),
            "sunlight": self.sunlight.copy(),
            "comm": self.comm.copy(),
            "location": self.location.copy(),
            "progress": self.progress.copy(),
            "size": self.size.copy(),
            "workload": self.workload.copy(),
        }
        return beta_t
    
    def beta_at(self, step: int) -> Dict[str, np.ndarray]:
        return self.beta_history.get(step, None)

    def sum_size_before(self, m: int, n: int, T: int) -> float:
        """
        获取t时间之前所有size[m, n]历史值，返回float字典。
        """
        result = 0.0
        for t in range(T):
            beta = self.beta_history.get(t)
            if beta is not None:
                result += beta["size"][m, n]
        return result

    def sum_workload_before(self, m: int, n: int, T: int) -> float:
        """
        获取t时间之前所有workload[m, n]历史值，返回float字典。
        """
        result = 0.0
        for t in range(T):
            beta = self.beta_history.get(t)
            if beta is not None:
                result += beta["workload"][m, n]
        return result
    
    def is_empty(self) -> bool:
        """
        检查状态管理器是否为空（没有初始化数据）。
        :return: 是否为空 (bool)
        """
        return self.P == 0 or self.O == 0 or self.M == 0 or self.N == 0
