from typing import Dict, Tuple

from envs.param import ENERGY_HARVEST_AMOUNT, STATIC_ENERGY_COST
from envs.snapshot.node import Node


def update_static_energy(nodes: Dict[Tuple[int, int], Node]):
    for n in nodes.values():
        n.energy = max(n.energy + STATIC_ENERGY_COST, 0.0)
        if n.gamma:
            n.energy = min(n.energy + ENERGY_HARVEST_AMOUNT, 100.0)
