from typing import Dict, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.core.decision import Decision
from envs.core.state import State
from envs.core.termination import all_tasks_completed, any_satellite_depleted
from envs.core.truncation import all_tasks_overtimed
from envs.param import COMPUTE_ENERGY_COST, INTERRUPTION_PENALTY, LAYER_COMPLETION_REWARD, LAYER_OUTPUT_DATA_SIZE, NO_ACTION_PENALTY, NUM_TASKS, T_STEP, TASK_COMPLETION_REWARD, TRANSMIT_ENERGY_COST, WRONG_EDGE_PENALTY
from envs.core.topology_manager import TopologyManager
from envs.core.task_manager import TaskManager
from envs.snapshot.request import TransReq
from envs.core.formulation import compute_aim_reward, compute_delay_penalty, compute_energy_penalty
from envs.core.energy import update_static_energy
from envs.core.transmission import process_transfers
from envs.core.computation import process_computation
from envs.core.observation import get_observation

class LEOEnv(gym.Env):
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, json_path: str):
        super().__init__()
        self.topology_manager = TopologyManager(json_path)
        self.task_manager = TaskManager(self.topology_manager)
        
        # 初始化基本参数
        self.step_counter = 0
        self.action_space = spaces.MultiDiscrete([6] * NUM_TASKS)

        # state space
        self.state = State()
        
        # decision space
        self.decision = Decision()
        
        # 初始化观察空间
        obs, _ = self.reset()
        self.observation_space = spaces.Box(low=0.0, high=1e9, shape=obs.shape, dtype=np.float32)
        

    def reset(self, seed=None):
        
        super().reset(seed=seed)
        
        self.step_counter = 0
        
        for key, n in self.topology_manager.nodes.items():
            n.energy = float(np.random.uniform(80, 100))
            
        self.task_manager.reset_tasks()
        
        self.decision.reset()
        
        return get_observation(
            self.topology_manager.nodes, 
            self.topology_manager.num_planes, 
            self.topology_manager.sats_per_plane, 
            self.step_counter, 
            self.task_manager.get_tasks(t=self.step_counter)
        ), {}

    def step(self, actions):
        
        # 设置初始数值
        action_reward = 0.0
        fail_reason = None
        trans_reqs = []
        
        terminated, truncated = False, False

        # 获取当前任务和节点状态
        tasks = self.task_manager.get_tasks(t=self.step_counter)
        nodes = list(self.topology_manager.nodes.values())

        # 检查动作数量是否与任务数量一致
        assert len(actions) == len(tasks)
        
        # update energy for all nodes by default
        update_static_energy(self.topology_manager.nodes)
        
        for task, act in zip(tasks, actions):
            
            if task.is_done:
                continue
            
            # 获取任务当前位置的节点
            p, o = task.plane_at, task.order_at
            node = self.topology_manager.nodes.get((p, o))
            
            # 节点不存在则跳过
            if node is None:
                continue

            # 处理节点的动作
            if act in [1, 2, 3, 4]:
                
                # action_reward = DATA_TRANSFER_PENALTY
                
                # 移动动作，重置计算进度
                if task.workload_done > 0:
                    action_reward += INTERRUPTION_PENALTY
                    task.workload_done = 0
                
                if task.data_sent > 0 and task.act != act:
                    action_reward += INTERRUPTION_PENALTY
                
                if act == 1:
                    dst = ((p + 1) % self.topology_manager.num_planes, o)
                    
                elif act == 2:
                    dst = ((p - 1) % self.topology_manager.num_planes, o)
                    
                elif act == 3:
                    dst = (p, (o + 1) % self.topology_manager.sats_per_plane)
                    
                else:
                    dst = (p, (o - 1) % self.topology_manager.sats_per_plane)
                    
                self.decision.set_rho(
                    u=(p, o),
                    v=dst,
                    n=task.layer_id,
                    m=task.id,
                    t=self.step_counter,
                    value=True
                )
                    
                if ((p, o), dst) in self.topology_manager.edges:

                    node.energy = max(node.energy + TRANSMIT_ENERGY_COST, 0.0)

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
                else:
                    action_reward += WRONG_EDGE_PENALTY
                    truncated = True
                    fail_reason = "wrong_link"

            elif act == 0:
                action_reward += NO_ACTION_PENALTY

            elif act == 5:
                
                self.decision.set_pi(
                    i=(p, o),
                    n=task.layer_id,
                    m=task.id,
                    t=self.step_counter,
                    value=True
                )
                
                # 计算动作，重置发送进度
                if task.data_sent > 0:
                    action_reward += INTERRUPTION_PENALTY
                    task.data_sent = 0

                state = process_computation(task)

                node.energy = max(node.energy + COMPUTE_ENERGY_COST, 0.0)

                if state == 'layer_complete':
                    action_reward += LAYER_COMPLETION_REWARD

                elif state == 'done':
                    action_reward += TASK_COMPLETION_REWARD

            task.t_end += 1
            task.act = act
                    
        # 处理数据传输请求
        process_transfers(
            trans_reqs=trans_reqs, 
            edges=self.topology_manager.edges, 
            tasks=tasks
        )

        aim_reward = compute_aim_reward(
            delay_penalty=compute_delay_penalty(tasks),
            energy_penalty=compute_energy_penalty(nodes)
        )

        reward = action_reward + aim_reward

        if all_tasks_completed(tasks):
            terminated = True
            fail_reason = "all_tasks_completed"
        elif any_satellite_depleted(nodes):
            terminated = True
            fail_reason = "satellite_energy_depleted"
            
        if all_tasks_overtimed(tasks):
            truncated = True
            fail_reason = "all_tasks_overtimed"
        
        obs = get_observation(self.topology_manager.nodes, self.topology_manager.num_planes, self.topology_manager.sats_per_plane, self.step_counter, tasks)
        
        info = {
            'global_time': self.step_counter,
            'reward': reward,
            'truncated': truncated,
            'fail_reason': fail_reason
        }
        
        self.step_counter += 1
        
        return obs, reward, terminated, truncated, info

    def render(self):
        from envs.renderer.visualizer import render_satellite_network
        tasks = self.task_manager.get_tasks()
        render_satellite_network(
            topology=self.topology_manager,
            tasks=tasks,
            step_counter=self.step_counter,
        )
