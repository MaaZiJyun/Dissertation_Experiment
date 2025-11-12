from typing import Dict, Tuple


class Decision:
    def __init__(self):
        self.decision_rho: Dict[Tuple[Tuple[int, int], Tuple[int, int], int, int, int], bool] = {}
        self.decision_pi: Dict[Tuple[Tuple[int, int], int, int, int], bool] = {}
        
    def set_rho(self, u: Tuple[int, int], v: Tuple[int, int], n: int, m: int, t: int, value: bool):
        self.decision_rho[(u, v, n, m, t)] = value

    def set_pi(self, i: Tuple[int, int], n: int, m: int, t: int, value: bool):
        self.decision_pi[(i, n, m, t)] = value
        
    def get_rho(self, u: Tuple[int, int], v: Tuple[int, int], n: int, m: int, t: int) -> bool:
        return self.decision_rho.get((u, v, n, m, t), False)

    def get_pi(self, i: Tuple[int, int], n: int, m: int, t: int) -> bool:
        return self.decision_pi.get((i, n, m, t), False)
    
    def reset(self):
        self.decision_rho.clear()
        self.decision_pi.clear()
        
    def get_uv_by_tm_true(self, t: int, m: int):
        return [
            (u, v)
            for (u, v, n, mm, tt), value in self.decision_rho.items()
            if mm == m and tt == t and value
        ]