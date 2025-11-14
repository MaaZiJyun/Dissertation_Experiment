import numpy as np
from typing import Dict, Tuple
from envs.IO.decision_manager import DecisionManager
from envs.IO.state_manager import StateManager
from envs.snapshot.info import Info

def get_obs(sm: StateManager, dm: DecisionManager, step: int) -> Tuple[Dict[str, np.ndarray], Info]:
    # ======== 从状态管理器提取状态空间 β_t ========
    beta_t = sm.report(step)
    
    # --- 可选的归一化与转换 ---
    # 这些超参数可在外部定义，如 B_MAX、R_MAX、MAX_NUM_LAYERS 等

    def _safe_normalize(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.size == 0:
            return a
        maxv = float(np.max(a)) if a.size > 0 else 0.0
        if maxv == 0.0:
            return a.astype(np.float32)
        return (a / (maxv + 1e-6)).astype(np.float32)

    # pad task arrays to fixed M_MAX to keep observation shapes stable
    M_MAX = getattr(sm, "M_MAX", beta_t.get("size", np.zeros((0,))).shape[0])

    def _pad(arr: np.ndarray, target_shape: tuple, fill=0.0):
        a = np.asarray(arr)
        out = np.full(target_shape, fill, dtype=np.float32)
        # compute slice sizes
        slices = tuple(slice(0, min(s, t)) for s, t in zip(a.shape, target_shape))
        out[slices] = a[tuple(slice(0, s) for s in a.shape)]
        return out

    obs = {
        "energy": np.asarray(_safe_normalize(beta_t["energy"]), dtype=np.float32),
        "sunlight": np.asarray(beta_t["sunlight"], dtype=np.float32),
        "comm": np.asarray(_safe_normalize(beta_t["comm"]), dtype=np.float32),
        "location": _pad(beta_t.get("location", np.zeros((0, 2))), (M_MAX, 2)),
        "progress": _pad(beta_t.get("progress", np.zeros((0,))), (M_MAX,)),
        "size": _pad(beta_t.get("size", np.zeros((0, sm.N_MAX))), (M_MAX, sm.N_MAX)),
        "workload": _pad(beta_t.get("workload", np.zeros((0, sm.N_MAX))), (M_MAX, sm.N_MAX)),
    }

    # ======== 从决策管理器提取动作空间 α_t ========
    alpha_t = dm.report(step)

    # return obs and a plain debug dict; env will build a snapshot Info object later
    info = {
        "alpha": alpha_t,
        "beta": beta_t,
        "step": step,
    }

    return obs, info


    