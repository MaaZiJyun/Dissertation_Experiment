import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.param import COMPUTE_ENERGY_COST, DATA_COMPUTE_PENALTY, DATA_TRANSFER_PENALTY, LAYER_COMPLETION_REWARD, LAYER_OUTPUT_DATA_SIZE, NO_ACTION_PENALTY, NUM_TASKS, T_STEP, TASK_COMPLETION_REWARD, TRANSMIT_ENERGY_COST, WRONG_EDGE_PENALTY
from envs.core.topology_manager import TopologyManager
from envs.core.task_manager import TaskManager
from envs.snapshot.request import TransReq
from envs.core.formulation import compute_aim_reward, compute_delay_penalty, compute_energy_penalty
from envs.core.energy import update_static_energy
from envs.core.transmission import process_transfers
from envs.core.computation import process_computation
from envs.core.observation import get_observation
from envs.core.termination import check_termination

class LEOEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, json_path: str):
        super().__init__()
        self.topology_manager = TopologyManager(json_path)
        self.task_manager = TaskManager(NUM_TASKS, self.topology_manager)
        
        # 初始化基本参数
        self.step_counter = 0
        self.total_reward = 0.0
        self.action_space = spaces.MultiDiscrete([6] * NUM_TASKS)

        # 初始化观察空间
        obs, _ = self.reset()
        self.observation_space = spaces.Box(low=0.0, high=1e9, shape=obs.shape, dtype=np.float32)

    # 任务生成逻辑已迁移到TaskManager

    def reset(self, seed=None):
        
        super().reset(seed=seed)
        
        self.step_counter = 0
        self.total_reward = 0.0
        
        for key, n in self.topology_manager.nodes.items():
            n.energy = float(np.random.uniform(50, 100))
            
        self.task_manager.reset_tasks()
        
        return get_observation(
            self.topology_manager.nodes, 
            self.topology_manager.num_planes, 
            self.topology_manager.sats_per_plane, 
            self.step_counter, 
            self.task_manager.get_tasks()
        ), {}

    def step(self, actions):
        
        # 设置初始数值
        action_reward = 0.0
        truncated, terminated = False, False
        fail_reason = None
        trans_reqs = []

        # 获取当前任务和节点状态
        tasks = self.task_manager.get_tasks()
        nodes = list(self.topology_manager.nodes.values())

        # 检查动作数量是否与任务数量一致
        assert len(actions) == len(tasks)
        
        # update energy for all nodes by default
        update_static_energy(self.topology_manager.nodes)
        
        for task, act in zip(tasks, actions):
            
            # 已经完成的任务或者未开始的任务不处理
            if task.t_start > self.step_counter or task.is_done:
                continue
            
            # 获取任务当前位置的节点
            p, o = task.plane_at, task.order_at
            node = self.topology_manager.nodes.get((p, o))
            
            # 节点不存在则跳过
            if node is None:
                continue

            # 处理节点的动作
            if act in [1, 2, 3, 4]:
                
                action_reward = DATA_TRANSFER_PENALTY
                
                # 移动动作，重置计算进度
                task.workload_done = 0
                
                if act == 1:
                    dst = ((p + 1) % self.topology_manager.num_planes, o)
                    
                elif act == 2:
                    dst = ((p - 1) % self.topology_manager.num_planes, o)
                    
                elif act == 3:
                    dst = (p, (o + 1) % self.topology_manager.sats_per_plane)
                    
                else:
                    dst = (p, (o - 1) % self.topology_manager.sats_per_plane)
                    
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
                    # 传输路径不合法
                    action_reward = WRONG_EDGE_PENALTY
                    truncated = True
                    fail_reason = "wrong_edge"
        
            elif act == 0:
                action_reward = NO_ACTION_PENALTY
                
            elif act == 5:
                
                state = process_computation(task)

                node.energy = max(node.energy + COMPUTE_ENERGY_COST, 0.0)

                if state == 'layer_complete':
                    action_reward = LAYER_COMPLETION_REWARD
                    
                elif state == 'done':
                    action_reward = TASK_COMPLETION_REWARD

                else:
                    action_reward = DATA_COMPUTE_PENALTY
                    
        # 处理数据传输请求
        process_transfers(
            trans_reqs=trans_reqs, 
            edges=self.topology_manager.edges, 
            tasks=tasks
        )

        terminated, truncated, fail_reason, action_reward = check_termination(
            terminated=terminated, 
            truncated=truncated, 
            action_reward=action_reward, 
            nodes=nodes, 
            tasks=tasks
        )

        aim_reward = compute_aim_reward(
            delay_penalty=compute_delay_penalty(tasks),
            energy_penalty=compute_energy_penalty(nodes)
        )

        self.total_reward += action_reward * aim_reward

        obs = get_observation(self.topology_manager.nodes, self.topology_manager.num_planes, self.topology_manager.sats_per_plane, self.step_counter, tasks)
        
        info = {
            'global_time': self.step_counter,
            'reward': self.total_reward,
            'truncated': truncated,
            'fail_reason': fail_reason
        }
        
        self.step_counter += 1
        
        return obs, self.total_reward, terminated, truncated, info

    def render(self):
        from envs.renderer.visualizer import render_satellite_network
        tasks = self.task_manager.get_tasks()
        render_satellite_network(self.topology_manager, tasks, self.step_counter, T_STEP)
