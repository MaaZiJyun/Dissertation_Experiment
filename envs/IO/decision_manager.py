import numpy as np
from typing import Dict, Tuple

from envs.param import NUM_LAYERS


class DecisionManager:
    def __init__(self):
        
        # placeholders for shapes
        self.P = 0
        self.O = 0
        self.M = 0
        self.N = NUM_LAYERS
        self.alpha_history = {}


    def initialize(self, p: int, o: int, m: int):
        """Initialize internal arrays.

        pi shape: (P, O, M, N)    -> per-satellite compute allocation
        rho shape: (P, O, P, O, M, N) -> per-link (src p,o -> dst p,o) per-task per-layer
        """
        self.P = p
        self.O = o
        self.M = m
        self.N = NUM_LAYERS
        self.pi = np.zeros((self.P, self.O, self.M, self.N), dtype=np.int8)
        self.rho = np.zeros((self.P, self.O, self.P, self.O, self.M, self.N), dtype=np.int8)

    def reset(self):
        self.alpha_history.clear()

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
        for mm in range(self.M):
            for nn in range(NUM_LAYERS):
                value = self.rho[up, uo, vp, vo, mm, nn]
                result[(mm, nn)] = bool(value)
        return result

    def update(self, step: int):
        self.alpha_history[step] = self._to_alpha()

    def _to_alpha(self) -> Dict[str, np.ndarray]:
        alpha_t = {
            "pi": self.pi.copy(),
            "rho": self.rho.copy(),
        }
        return alpha_t

    def alpha_at(self, step: int) -> Dict[str, np.ndarray]:
        return self.alpha_history.get(step, None)
    
    def is_empty(self) -> bool:
        """
        检查决策管理器是否为空（没有初始化数据）。
        :return: 是否为空 (bool)
        """
        return self.P == 0 or self.O == 0 or self.M == 0 or self.N == 0