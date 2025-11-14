import numpy as np
from typing import Dict, Tuple

from envs.param import MAX_NUM_LAYERS, MAX_NUM_TASKS


class DecisionManager:
    def __init__(self, p_max: int, o_max: int):
        
        self._M = 0
        self._alpha = {}

        self.P_MAX = p_max
        self.O_MAX = o_max
        self.N_MAX = MAX_NUM_LAYERS
        self.M_MAX = MAX_NUM_TASKS
        
        self.pi = np.zeros((self.P_MAX, self.O_MAX, self.M_MAX, self.N_MAX), dtype=np.int8)
        self.rho = np.zeros((self.P_MAX, self.O_MAX, self.P_MAX, self.O_MAX, self.M_MAX, self.N_MAX), dtype=np.int8)

    def update(self, current_task_length: int):
        self._M = current_task_length
        
    def report(self, step: int) -> Dict[str, np.ndarray]:
        alpha_t = self._to_alpha()
        self._alpha[step] = alpha_t
        return alpha_t
        
    def reset(self):
        self.pi.fill(0)
        self.rho.fill(0)

    def write_pi(self, p: int, o: int, n: int, m: int, value: int):
        # p,o are plane and order indices; m is task index; n is layer index
        self.pi[p, o, m, n] = int(value)

    def write_rho(self, u: Tuple[int, int], v: Tuple[int, int], n: int, m: int, value: int):
        # u and v are tuples (plane, order)
        up, uo = u
        vp, vo = v
        self.rho[up, uo, vp, vo, m, n] = int(value)
        
    def get_rho(self, u: Tuple[int, int], v: Tuple[int, int], n: int, m: int):
        return self.rho[u[0], u[1], v[0], v[1], m, n]

    def get_rho_by_uv(self, u: Tuple[int, int], v: Tuple[int, int]) -> Dict[Tuple[int, int], bool]:
        result = {}
        up, uo = u
        vp, vo = v
        # iterate over tasks and layers
        for mm in range(self._M):
            for nn in range(MAX_NUM_LAYERS):
                value = self.rho[up, uo, vp, vo, mm, nn]
                result[(mm, nn)] = bool(value)
        return result

    def _to_alpha(self) -> Dict[str, np.ndarray]:
        # slice by task count on the M axis and copy to make a snapshot
        if self._M <= 0:
            valid_pi = np.zeros((self.P_MAX, self.O_MAX, 0, self.N_MAX), dtype=np.int8)
            valid_rho = np.zeros((self.P_MAX, self.O_MAX, self.P_MAX, self.O_MAX, 0, self.N_MAX), dtype=np.int8)
        else:
            valid_pi = self.pi[:, :, :self._M, :].copy()   # shape (P_MAX,O_MAX,_M,N_MAX)
            valid_rho = self.rho[:, :, :, :, :self._M, :].copy()  # shape (P_MAX,O_MAX,P_MAX,O_MAX,_M,N_MAX)
        alpha_t = {
            "pi": valid_pi,
            "rho": valid_rho,
        }
        return alpha_t
    
    def is_empty(self) -> bool:
        """
        检查决策管理器是否为空（没有初始化数据）。
        :return: 是否为空 (bool)
        """
        return self._M == 0