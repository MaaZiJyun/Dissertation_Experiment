import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List
from envs.IO.decision_manager import DecisionManager
from envs.IO.state_manager import StateManager
from envs.core.termination import all_tasks_completed, any_satellite_depleted
from envs.core.truncation import all_tasks_overtimed, any_illegal_link
from envs.param import COMPUTE_ENERGY_COST, INTERRUPTION_PENALTY, LAYER_OUTPUT_DATA_SIZE, LAYER_PROCESS_STEP_COST, NO_ACTION_PENALTY, MAX_NUM_TASKS, TRANSMIT_ENERGY_COST
from envs.core.json_manager import JsonManager
from envs.core.task_manager import TaskManager
from envs.snapshot.request import CompReq, TransReq
from envs.core.formulation import compute_aim_reward, compute_delay_penalty, compute_energy_penalty
from envs.core.energy import update_static_energy
from envs.core.transmission import do_transferring
from envs.core.computation import do_computing
from envs.core.observation import get_obs
from envs.snapshot.info import Info

class LEOEnv(gym.Env):
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, json_path: str):
        super().__init__()
        
        # Entities Managers
        self.JM = JsonManager(json_path)
        self.TM = TaskManager(self.JM)
        
        # Attributes Managers
        self.SM = StateManager(self.JM.N_PLANE, self.JM.N_SAT)
        self.DM = DecisionManager(self.JM.N_PLANE, self.JM.N_SAT)
        
        # 初始化基本参数
        self.step_counter = 0
        self.action_space = spaces.MultiDiscrete([6] * MAX_NUM_TASKS)

        # 初始化观察空间 based on manager max sizes (stable shape independent of current _M)
        p = self.SM.P_MAX
        o = self.SM.O_MAX
        n = self.SM.N_MAX
        m = self.SM.M_MAX

        obs_spaces = {
            "energy": spaces.Box(low=-np.inf, high=np.inf, shape=(p, o), dtype=np.float32),
            "sunlight": spaces.Box(low=-np.inf, high=np.inf, shape=(p, o), dtype=np.float32),
            "comm": spaces.Box(low=-np.inf, high=np.inf, shape=(p, o, p, o), dtype=np.float32),
            "location": spaces.Box(low=-np.inf, high=np.inf, shape=(m, 2), dtype=np.float32),
            "progress": spaces.Box(low=-np.inf, high=np.inf, shape=(m,), dtype=np.float32),
            "size": spaces.Box(low=-np.inf, high=np.inf, shape=(m, n), dtype=np.float32),
            "workload": spaces.Box(low=-np.inf, high=np.inf, shape=(m, n), dtype=np.float32),
        }
        self.observation_space = spaces.Dict(obs_spaces)

    def _align_obs(self, obs: dict) -> dict:
        """Pad or trim observation arrays to match self.observation_space shapes."""
        aligned = {}
        for k, space in self.observation_space.spaces.items():
            wanted_shape = space.shape
            arr = np.asarray(obs.get(k, np.zeros(wanted_shape)), dtype=np.float32)
            # if arr has fewer dims, left-pad with batch dim removal
            if arr.shape == wanted_shape:
                aligned[k] = arr
                continue
            # create output container
            out = np.zeros(wanted_shape, dtype=np.float32)
            # compute slices
            slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, wanted_shape))
            out[slices] = arr[tuple(slice(0, s) for s in arr.shape)]
            aligned[k] = out
        return aligned

    def reset(self, seed=None, options=None):
        # Accept the `options` kwarg used by Gymnasium wrappers (Monitor, VecEnv).
        super().reset(seed=seed)
        
        self.step_counter = 0
        
        self.TM.reset()
        self.JM.reset()

        all_nodes, all_edges, all_tasks = self.JM.get_nodes(), self.JM.get_edges(), self.TM.get_tasks()
        
        self.SM.reset()
        self.DM.reset()
        
        self.SM.setup(
            all_nodes=all_nodes,
            all_edges=all_edges,
            all_tasks=all_tasks
        )

        obs, info = get_obs(sm=self.SM, dm=self.DM, step=self.step_counter)
        return self._align_obs(obs), info

    def step(self, actions):
        # 初始化数值
        action_reward = 0.0
        trans_reqs: List[TransReq] = []
        comp_reqs: List[CompReq] = []

        terminated, truncated = False, False
        terminated_reason, truncated_reason = "None", "None"

        # 获取当前任务和节点状态
        nodes, edges = self.JM.get_nodes(), self.JM.get_edges()
        tasks = self.TM.get_tasks_at(step=self.step_counter)
        n_tasks = len(tasks)

        # 获取有效动作
        valid_actions = list(actions)[:n_tasks]

        # 初始化状态管理器和决策管理器的活跃任务数
        self.SM.update(n_tasks)
        self.DM.update(n_tasks)

        assert len(valid_actions) == n_tasks

        # update energy for all nodes by default
        update_static_energy(nodes, self.SM)

        for task, act in zip(tasks, valid_actions):
            if task.is_done:
                continue

            p, o = task.plane_at, task.order_at
            node = self.JM.nodes.get((p, o))
            if node is None:
                continue

            # movement actions
            if act in [1, 2, 3, 4]:
                if task.workload_done > 0:
                    action_reward += INTERRUPTION_PENALTY
                if task.data_sent > 0 and task.act != act:
                    action_reward += INTERRUPTION_PENALTY

                if act == 1:
                    dst = ((p + 1) % self.JM.N_PLANE, o)
                elif act == 2:
                    dst = ((p - 1) % self.JM.N_PLANE, o)
                elif act == 3:
                    dst = (p, (o + 1) % self.JM.N_SAT)
                else:
                    dst = (p, (o - 1) % self.JM.N_SAT)

                self.DM.write_rho(u=(p, o), v=dst, n=task.layer_id, m=task.id, value=True)

                if ((p, o), dst) in self.JM.edges:
                    node.energy = max(node.energy + TRANSMIT_ENERGY_COST, 0.0)
                    self.SM.write_energy(p=p, o=o, value=node.energy)
                    data_bits = LAYER_OUTPUT_DATA_SIZE[task.layer_id]
                    trans_reqs.append(
                        TransReq(task_id=task.id, src=(p, o), dst=dst, target_file_size=data_bits, step=self.step_counter)
                    )

            elif act == 0:
                action_reward += NO_ACTION_PENALTY

            elif act == 5:
                # compute action
                data_bits = LAYER_OUTPUT_DATA_SIZE[task.layer_id]
                self.DM.write_pi(p=p, o=o, n=task.layer_id, m=task.id, value=True)
                node.energy = max(node.energy + COMPUTE_ENERGY_COST, 0.0)
                self.SM.write_energy(p=p, o=o, value=node.energy)
                comp_reqs.append(
                    CompReq(
                        task_id=task.id,
                        node_id=(p, o),
                        layer_id=task.layer_id,
                        target_workload=LAYER_PROCESS_STEP_COST[task.layer_id],
                        workload_done=task.workload_done,
                    )
                )
                self.SM.write_size(m=task.id, n=task.layer_id, value=data_bits)

            task.t_end += 1
            task.act = act

        # 执行传输与计算
        action_reward += do_transferring(tasks=tasks, trans_reqs=trans_reqs, sm=self.SM, dm=self.DM, t=self.step_counter)
        action_reward += do_computing(comp_reqs=comp_reqs, tasks=tasks, sm=self.SM, dm=self.DM, t=self.step_counter)

        # 计算目标与最终 reward
        aim_reward = compute_aim_reward(delay_penalty=compute_delay_penalty(tasks), energy_penalty=compute_energy_penalty(nodes))
        reward = action_reward + aim_reward

        # 终止/截断判定
        if all_tasks_completed(tasks):
            terminated = True
            terminated_reason = "all_tasks_completed"
        elif any_satellite_depleted(nodes):
            terminated = True
            terminated_reason = "satellite_energy_depleted"

        if all_tasks_overtimed(tasks):
            truncated = True
            truncated_reason = "all_tasks_overtimed"
        elif any_illegal_link(self.SM, self.DM):
            truncated = True
            truncated_reason = "any_illegal_link"

        # 产出观测和可序列化的 info
        obs, dbg_info = get_obs(sm=self.SM, dm=self.DM, step=self.step_counter)

        info_obj = Info(
            num_nodes=len(nodes),
            num_edges=len(edges),
            num_tasks=n_tasks,
            step=self.step_counter,
            alpha=dbg_info['alpha'],
            beta=dbg_info['beta'],
        )

        info_obj.reward = float(reward)
        info_obj.is_truncated = bool(truncated)
        info_obj.truncated_reason = truncated_reason
        info_obj.is_terminated = bool(terminated)
        info_obj.terminated_reason = terminated_reason

        info_serial = info_obj.to_serializable()

        self.step_counter += 1

        return obs, reward, terminated, truncated, info_serial

    def render(self):
        from envs.renderer.visualizer import render_satellite_network
        tasks = self.TM.get_tasks_at(step=self.step_counter)
        nodes, edges = self.JM.get_nodes(), self.JM.get_edges()
        render_satellite_network(
            nodes=nodes,
            edges=edges,
            tasks=tasks,
            step_counter=self.step_counter,
        )
