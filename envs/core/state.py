from typing import Dict, Tuple


class State:
    def __init__(self):
        # satellite sunlit status
        self.state_gamma: Dict[Tuple[int, int], bool] = {}
        # satellite energy status
        self.state_energy: Dict[Tuple[int, int], float] = {}
        # link rate status
        self.state_C: Dict[Tuple[int, int], Tuple[int, int], float] = {}
        # task data size status
        self.state_h: Dict[int, int, float] = {}
        
    def set_gamma(self, i: Tuple[int, int], value: bool):
        self.state_gamma[i] = value
        
    def set_energy(self, i: Tuple[int, int], value: float):
        self.state_energy[i] = value
        
    def set_C(self, u: Tuple[int, int], v: Tuple[int, int], value: float):
        self.state_C[(u, v)] = value

    def set_h(self, m: int, n: int, value: float):
        self.state_h[(m, n)] = value
        
    def get_gamma(self, i: Tuple[int, int]) -> bool:
        return self.state_gamma.get(i, False)

    def get_energy(self, i: Tuple[int, int]) -> float:
        return self.state_energy.get(i, 0.0)

    def get_C(self, u: Tuple[int, int], v: Tuple[int, int]) -> float:
        return self.state_C.get((u, v), 0.0)

    def get_h(self, m: int, n: int) -> float:
        return self.state_h.get((m, n), 0.0)
    
    def reset(self):
        self.state_gamma.clear()
        self.state_energy.clear()
        self.state_C.clear()
        self.state_h.clear()