import numpy as np
from typing import List
from envs.IO.decision_manager import DecisionManager
from envs.IO.state_manager import StateManager
from envs.snapshot.task import Task
from envs.param import MAX_TASKS


def _flatten_matrix(mat: np.ndarray) -> List[float]:
    return list(mat.ravel().astype(np.float32))


def get_obs(sm: StateManager, dm: DecisionManager, tasks: List[Task]) -> np.ndarray:
    """Construct observation vector from StateManager and DecisionManager.

    Layout (concatenated):
    - energy: P*O
    - sunlight: P*O
    - comm (flattened): (P*O)*(P*O) (log-scaled)
    - per-task states: for each task: id, layer_id, plane_at, order_at, t_start, t_end, is_done, workload_done, data_sent
    - pi (flattened)
    - rho (flattened)
    """
    parts: List[float] = []

    # energy and sunlight
    parts.extend(list(sm.energy.ravel().astype(np.float32)))
    parts.extend(list(sm.sunlight.ravel().astype(np.float32)))

    # comm: apply small log scaling to stabilize magnitudes (add eps)
    comm = sm.comm.astype(np.float32)
    eps = 1e-9
    comm_log = np.log1p(comm + eps)
    parts.extend(_flatten_matrix(comm_log))

    # tasks: include fixed MAX_TASKS slots (pad with zeros)
    num_slots = getattr(dm, 'M', MAX_TASKS)
    for idx in range(num_slots):
        if idx < len(tasks):
            t = tasks[idx]
            parts.extend([
                int(t.id),
                int(t.layer_id),
                int(t.plane_at),
                int(t.order_at),
                int(t.t_start),
                int(t.t_end),
                1.0 if t.is_done else 0.0,
                int(getattr(t, 'workload_done', 0)),
                float(getattr(t, 'data_sent', 0)),
            ])
        else:
            # padding for empty task slots
            parts.extend([0.0] * 9)

    # decision pi and rho: flatten
    # decision pi and rho: flatten using their defined shapes to keep obs size stable
    if hasattr(dm, 'pi') and dm.pi is not None:
        parts.extend(list(dm.pi.ravel().astype(np.float32)))
    else:
        parts.extend([0.0] * (MAX_TASKS * 1 * 1 * getattr(dm, 'N', 0)))

    if hasattr(dm, 'rho') and dm.rho is not None:
        parts.extend(list(dm.rho.ravel().astype(np.float32)))
    else:
        # shape (P,O,P,O,M,N) -> length = P*O*P*O*M*N but fallback to 0
        parts.extend([0.0] * 0)

    return np.array(parts, dtype=np.float32)
