import numpy as np
from typing import List
from envs.snapshot.task import Task

def get_observation(nodes, num_planes, sats_per_plane, global_time_counter, tasks: List[Task]):
    parts: List[float] = []
    for p in range(num_planes):
        for s in range(sats_per_plane):
            n = nodes.get((p, s))
            parts.append(n.energy if n is not None else 0.0)
    parts.append(float(global_time_counter))
    for t in tasks:
        parts.extend([
            t.id,
            t.layer_id,
            t.plane_at,
            t.order_at,
            t.t_start,
            t.t_end,
            t.is_done,
            t.workload_done,
            t.data_sent,
        ])
    return np.array(parts, dtype=np.float32)
