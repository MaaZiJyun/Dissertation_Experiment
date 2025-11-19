import numpy as np
from typing import Dict, Tuple
from envs.IO.decision_manager import DecisionManager
from envs.IO.state_manager import StateManager
from envs.core.task_manager import TaskManager
from envs.param import MAX_NUM_TASKS
from envs.snapshot.info import Info

def get_obs(sm: StateManager, dm: DecisionManager, tm: TaskManager, step: int) -> Tuple[Dict[str, np.ndarray], Info]:
    # ======== 从状态管理器提取状态空间 β_t ========
    beta_t = sm.report(step)
    
    # -------- 安全归一化 --------
    def _safe_normalize(arr: np.ndarray) -> np.ndarray:
        if arr is None:
            return arr
        a = np.asarray(arr, dtype=np.float32)
        v = np.max(np.abs(a))
        if v < 1e-9:
            return a
        return a / (v + 1e-6)

    # -------- 通用 padding --------
    def _pad(arr: np.ndarray, shape: tuple, fill: float = 0.0) -> np.ndarray:
        out = np.full(shape, fill, dtype=np.float32)
        if arr is None:
            return out
        a = np.asarray(arr, dtype=np.float32)
        slices_out = tuple(slice(0, min(a_dim, o_dim)) for a_dim, o_dim in zip(a.shape, shape))
        slices_in  = tuple(slice(0, s.stop) for s in slices_out)
        out[slices_out] = a[slices_in]
        return out
    
    
    # ======== 固定缩放常量（基于状态管理器的最大可能值） ========
    # 这些常量确保不同观测维度在合理尺度上，便于学习器和归一化器工作。
    # P_MAX / O_MAX 可能反映平面/轨道维度，但我们只需要保证非归一化量的量纲一致。
    # 如果某个值为 0，则使用 1.0 作为保护除数。
    p_max = max(getattr(sm, "P_MAX", 1), 1)
    o_max = max(getattr(sm, "O_MAX", 1), 1)
    n_max = max(getattr(sm, "N_MAX", 1), 1)

    # sunlight: 假设为能量采集强度，归一化到 [0,1] 的近似范围
    sunlight_scale = 1.0 if p_max == 0 else float(p_max)

    # location: 假设为二维坐标，按最大节点数归一化
    location_scale = float(max(n_max, 1))

    # progress/size/workload: 按 n_max 或其他 task 相关最大值缩放
    progress_scale = float(max(n_max, 1))
    size_scale = float(max(n_max, 1))
    workload_scale = float(max(n_max, 1))

    obs = {
        "energy": np.asarray(_safe_normalize(beta_t["energy"]), dtype=np.float32),
        # apply deterministic scaling to fields that were previously unscaled
        "sunlight": (np.asarray(beta_t.get("sunlight", 0.0), dtype=np.float32) / (sunlight_scale + 1e-9)).astype(np.float32),
        "comm": np.asarray(_safe_normalize(beta_t["comm"]), dtype=np.float32),
        "location": (_pad(beta_t.get("location", None), (MAX_NUM_TASKS, 2)) / (location_scale + 1e-9)).astype(np.float32),
        "progress": (_pad(beta_t.get("progress", None), (MAX_NUM_TASKS,)) / (progress_scale + 1e-9)).astype(np.float32),
        "size": (_pad(beta_t.get("size", None), (MAX_NUM_TASKS, sm.N_MAX)) / (size_scale + 1e-9)).astype(np.float32),
        "workload": (_pad(beta_t.get("workload", None), (MAX_NUM_TASKS, sm.N_MAX)) / (workload_scale + 1e-9)).astype(np.float32),
    }
    
    # ========= 动作掩码 =========
    
    # Build action_mask for all tasks via TaskManager API
    tasks = tm.get_tasks_at(step)
    mask = tm.build_action_mask_for_tasks(tasks)  # shape (MAX_NUM_TASKS, 6), dtype=bool
    obs["action_mask"] = mask.astype(np.int8)

    # ======== 从决策管理器提取动作空间 α_t ========
    alpha_t = dm.report(step)

    # return obs and a plain debug dict; env will build a snapshot Info object later
    info = {
        "alpha": alpha_t,
        "beta": beta_t,
        "step": step,
    }

    return obs, info
