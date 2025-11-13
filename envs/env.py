import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.IO.decision_manager import DecisionManager
from envs.IO.state_manager import StateManager
from envs.core.termination import all_tasks_completed, any_satellite_depleted
from envs.core.truncation import all_tasks_overtimed, any_illegal_link
from envs.param import COMPUTE_ENERGY_COST, INTERRUPTION_PENALTY, LAYER_COMPLETION_REWARD, LAYER_OUTPUT_DATA_SIZE, LAYER_PROCESS_STEP_COST, NO_ACTION_PENALTY, MAX_TASKS, T_STEP, TASK_COMPLETION_REWARD, TRANSMIT_ENERGY_COST, WRONG_EDGE_PENALTY
from envs.core.json_manager import JsonManager
from envs.core.task_manager import TaskManager
from envs.snapshot.request import CompReq, TransReq
from envs.core.formulation import compute_aim_reward, compute_delay_penalty, compute_energy_penalty
from envs.core.energy import update_static_energy
from envs.core.transmission import do_transferring
from envs.core.computation import do_computing
from envs.core.observation import get_obs

class LEOEnv(gym.Env):
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, json_path: str):
        super().__init__()
        
        # Entities Managers
        self.JM = JsonManager(json_path)
        self.TM = TaskManager(self.JM)
        self.SM = StateManager()
        self.DM = DecisionManager()
        
        # 初始化基本参数
        self.step_counter = 0
        self.action_space = spaces.MultiDiscrete([6] * MAX_TASKS)

        # 初始化观察空间
        obs, _ = self.reset()
        self.observation_space = spaces.Box(low=0.0, high=1e9, shape=obs.shape, dtype=np.float32)

    def reset(self, seed=None):
        
        super().reset(seed=seed)
        
        self.step_counter = 0
        
        for key, n in self.JM.nodes.items():
            n.energy = float(np.random.uniform(80, 100))
        
        self.TM.reset()
        self.JM.reset()
        self.SM.reset()
        self.DM.reset()

        nodes, edges = self.JM.get_nodes(), self.JM.get_edges()
        tasks = self.TM.get_tasks_at(step=self.step_counter)

        self.SM.initialize(self.JM.N_PLANE, self.JM.N_SAT, nodes, edges, tasks)
        self.DM.initialize(self.JM.N_PLANE, self.JM.N_SAT, MAX_TASKS)
            
        return get_obs(
            sm=self.SM, 
            dm=self.DM, 
            tasks=self.TM.get_tasks_at(step=self.step_counter)
        ), {}

    def step(self, actions):
        
        # 设置初始数值
        action_reward = 0.0
        trans_reqs = []
        comp_reqs = []
        
        terminated, truncated = False, False
        terminated_reason, truncated_reason = "None", "None"

        # 获取当前任务和节点状态
        nodes, edges = self.JM.get_nodes(), self.JM.get_edges()
        tasks = self.TM.get_tasks_at(step=self.step_counter)

        # 获取有效动作
        valid_actions = actions[:len(tasks)]

        # 初始化状态管理器和决策管理器
        self.SM.initialize(self.JM.N_PLANE, self.JM.N_SAT, nodes, edges, tasks)
        self.DM.initialize(self.JM.N_PLANE, self.JM.N_SAT, MAX_TASKS)

        # 检查动作数量是否与任务数量一致
        assert len(valid_actions) == len(tasks)
        
        # update energy for all nodes by default
        update_static_energy(nodes, self.SM)
        
        for task, act in zip(tasks, valid_actions):

            if task.is_done:
                continue
            
            # 获取任务当前位置的节点
            p, o = task.plane_at, task.order_at
            node = self.JM.nodes.get((p, o))
            
            # 节点不存在则跳过
            if node is None:
                continue

            # 处理节点的动作
            if act in [1, 2, 3, 4]:
                
                # 移动动作，重置计算进度
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
                    
                self.DM.write_rho(
                    u=(p, o),
                    v=dst,
                    n=task.layer_id,
                    m=task.id,
                    value=True
                )

                if ((p, o), dst) in self.JM.edges:

                    node.energy = max(node.energy + TRANSMIT_ENERGY_COST, 0.0)
                    self.SM.write_energy(
                        p=p,
                        o=o,
                        value=node.energy
                    )

                    data_bits = LAYER_OUTPUT_DATA_SIZE[task.layer_id]
                    
                    trans_reqs.append(
                        TransReq(
                            task_id=task.id,
                            src=(p, o),
                            dst=dst,
                            target_file_size=data_bits,
                            step=self.step_counter
                        )
                    )

            elif act == 0:
                action_reward += NO_ACTION_PENALTY

            elif act == 5:
                
                self.DM.write_pi(
                    p=p,
                    o=o,
                    n=task.layer_id,
                    m=task.id,
                    value=True
                )
                
                node.energy = max(node.energy + COMPUTE_ENERGY_COST, 0.0)
                self.SM.write_energy(
                        p=p,
                        o=o,
                        value=node.energy
                )

                comp_reqs.append(
                    CompReq(
                        task_id=task.id,
                        node_id=(p, o),
                        layer_id=task.layer_id,
                        target_workload=LAYER_PROCESS_STEP_COST[task.layer_id],
                        workload_done=task.workload_done
                    )
                )
                
                self.SM.write_size(
                        m=task.id,
                        n=task.layer_id,
                        value=data_bits
                )

            task.t_end += 1
            task.act = act
                    
        # 处理数据传输请求
        # process_transfers(
        #     trans_reqs=trans_reqs, 
        #     edges=self.JM.edges, 
        #     tasks=tasks
        # )
        
        action_reward += do_transferring(
            tasks=tasks,
            trans_reqs=trans_reqs,
            sm=self.SM,
            dm=self.DM,
            t=self.step_counter
        )

        action_reward += do_computing(
            comp_reqs=comp_reqs,
            tasks=tasks,
            sm=self.SM,
            dm=self.DM,
            t=self.step_counter
        )

        # 更新系统状态变量
        self.SM.update(step=self.step_counter)
        self.DM.update(step=self.step_counter)

        aim_reward = compute_aim_reward(
            delay_penalty=compute_delay_penalty(tasks),
            energy_penalty=compute_energy_penalty(nodes)
        )

        reward = action_reward + aim_reward

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
            
        obs = get_obs(self.SM, self.DM, tasks)
        
        info = {
            'global_time': self.step_counter,
            'reward': reward,
            'truncated': truncated,
            'truncated_reason': truncated_reason,
            'terminated': terminated,
            'fail_reason': terminated_reason
        }
        print(f"Step {self.step_counter}: Reward={reward}, Terminated={terminated}, Truncated={truncated}, Info={info}")

        self.step_counter += 1
        
        return obs, reward, terminated, truncated, info

    def render(self):
        from envs.renderer.visualizer import render_satellite_network
        tasks = self.TM.get_tasks()
        render_satellite_network(
            topology=self.JM,
            tasks=tasks,
            step_counter=self.step_counter,
        )
