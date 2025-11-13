from typing import Dict, List, Tuple

from envs.IO.state_manager import StateManager
from envs.param import ENERGY_HARVEST_AMOUNT, STATIC_ENERGY_COST
from envs.snapshot.node import Node


def update_static_energy(nodes: List[Node], sm: StateManager):
    for n in nodes:
        n.energy = max(n.energy + STATIC_ENERGY_COST, 0.0)
        sm.write_energy(n.plane_id, n.order_id, n.energy)
        
        if n.gamma:
            n.energy = min(n.energy + ENERGY_HARVEST_AMOUNT, 100.0)
            sm.write_energy(n.plane_id, n.order_id, n.energy)
