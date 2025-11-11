import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List
from envs.object.node import Node
from envs.object.edge import Edge
from envs.object.task import Task


class LEOEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, json_path: str):
        super().__init__()
        
        # obtain the file path of the JSON topology
        self.json_path = Path(json_path)
        
        # parameters
        self.num_layers = 5
        self.num_tasks = 1
        self.t_slot = 32 #seconds
        self.isl_bit_per_slot = 1_000_000_000 * self.t_slot # bits/time_slot
        self.x = 2_400_000_000 # image raw data -> bits

        # time cost per layer updating
        self.layer_time_cost = [25, 15, 5, 5, 1] # in seconds
        self.layer_output_bit = [self.x, self.x / 10e1, self.x / 10e4, self.x / 10e8, 8] # bits per layer

        # precision
        self.step_per_slot = 100 # steps per slot
        
        # step variables
        self.t_step = self.t_slot / self.step_per_slot
        self.isl_bit_per_step = self.isl_bit_per_slot / self.step_per_slot
        self.layer_process_step_cost = [i * self.step_per_slot for i in self.layer_time_cost]

        # network topology
        self.nodes: Dict[Tuple[int, int], Node] = {}
        self.edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Edge] = {}
        
        self.global_time_counter = 0

        self._load_topology()

        # derive dimensions from loaded nodes
        self.planes = sorted({p for (p, s) in self.nodes.keys()})
        self.sats_per_plane = max(s for (p, s) in self.nodes.keys()) + 1 if self.nodes else 0
        self.num_planes = max(self.planes) + 1 if self.planes else 0

        self.tasks: List[Task] = []
        self._load_tasks()

        # runtime: perform an initial reset and then derive action/observation
        # spaces from the actual returned observation (prevents mismatches).
        obs, _ = self.reset()

        # action space per task (6 discrete actions per task)
        self.action_space = spaces.MultiDiscrete([6] * len(self.tasks))

        # observation: infer shape from the reset observation
        self.observation_space = spaces.Box(low=0.0, high=1e9, shape=obs.shape, dtype=np.float32)

    def _load_topology(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
            
        # index -> (plane_id, order_id)
        index_map = {}
        
        for node in data.get('nodes', []):
            if 'plane_id' in node and 'order_id' in node:
                pid = int(node['plane_id'])
                sid = int(node['order_id'])
            else:
                continue

            node_obj = Node.model_validate({
                'id': int(node.get('index', -1)),
                'plane_id': pid,
                'order_id': sid,
                'energy': float(node.get('energy', np.random.uniform(50, 100))),
                'gamma': bool(int(node.get('gamma', 0))),
                'x': float(node.get('x', 0.0)),
                'y': float(node.get('y', 0.0)),
                'z': float(node.get('z', 0.0)),
            })

            self.nodes[(pid, sid)] = node_obj
            if 'index' in node:
                index_map[int(node['index'])] = (pid, sid)
        
        for edge in data.get('edges', []):
            rate = edge.get('rate')
            uid = edge.get('u')
            vid = edge.get('v')
            if isinstance(uid, int) and isinstance(vid, int) and uid in index_map and vid in index_map:
                ua = index_map[uid]
                va = index_map[vid]
            else:
                # cannot resolve endpoints -> skip
                continue
            u = self.nodes.get(ua)
            v = self.nodes.get(va)

            # create Edge model instances (use index if present)
            self.edges[(ua, va)] = Edge.model_validate({
                'id': int(edge.get('index', -1)),
                'u': u,
                'v': v,
                'rate': float(rate) if rate is not None else None,
            })

            self.edges[(va, ua)] = Edge.model_validate({
                'id': int(edge.get('index', -1)),
                'u': v,
                'v': u,
                'rate': float(rate) if rate is not None else None,
            })

    def _load_tasks(self):
        while len(self.tasks) < self.num_tasks:
            _id = len(self.tasks)
            node_keys = list(self.nodes.keys())
            idx = np.random.randint(0, len(node_keys))
            (_plane_at, _order_at) = node_keys[idx]
            _t_start = np.random.randint(0, 5)
            task_obj = Task(
                id=_id,
                layer_id=0, # layer id 0 means the data is raw data
                layer_process=self.layer_process_step_cost[0],
                plane_at=_plane_at,
                order_at=_order_at,
                t_start=_t_start,
                t_end=0,
                is_done=False,
            )
            self.tasks.append(task_obj)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.global_time_counter = 0
        
        # reset energies to random (50,100]
        for key, n in self.nodes.items():
            n.energy = float(np.random.uniform(50, 100))
            
        # sample num_tasks distinct satellites
        self.tasks = []
        self._load_tasks()
        
        return self._get_obs(), {}

    def step(self, actions: List[int]):
        assert len(actions) == len(self.tasks)
        self.global_time_counter += 1
        total_reward = 0.0
        total_energy_cost = 0.0  # 用于记录本step能耗
        total_delay_cost = 0.0   # 用于记录本step延迟
        
        # ========== (1) 更新能量 ==========
        # solar recharge
        for k, node in self.nodes.items():
            # minus small energy for static cost
            node.energy -= 0.01
            
            # add some energy if sunlit
            if int(node.gamma) == 1:
                node.energy = min(node.energy + 1.0, 100.0)
                
                
        # ========== (2) 执行动作 ==========
        # apply actions
        for task, act in zip(self.tasks, actions):
            
            # not started yet
            if task.t_start > self.global_time_counter:
                continue
            
            # already finished
            if task.is_done:
                continue
            
            # get current position
            p, o = task.plane_at, task.order_at
            
            # get current node
            node = self.nodes.get((p, o))
            if node is None:
                continue

            # set reward for this step
            reward = 0.0
            energy_cost = 0.0
            
            # doing nothing, we don't want it to be lazy
            if act == 0:
                # we don't want it to be lazy
                reward = -1.0
            
            # move to next plane
            elif act == 1:
                # get next pos
                next_p = (p + 1) % self.num_planes
                
                # get the destination
                dest = (next_p, o)
                
                # ensure the destination is correct
                if ((p, o), dest) in self.edges:
                    # update pos
                    task.plane_at = next_p
                    # clear the layer process progress
                    task.layer_process = 0
                    # minus energy for moving
                    energy_cost = 0.2
                    # and reward, we don't want it move randomly
                    reward = -1
                else:
                    reward = -2
                    
            # move to previous plane
            elif act == 2:
                prev_p = (p - 1) % self.num_planes
                dest = (prev_p, o)
                if ((p, o), dest) in self.edges:
                    # update pos
                    task.plane_at = prev_p
                    # clear the layer process progress
                    task.layer_process = 0
                    energy_cost = 0.2
                    reward = -1
                else:
                    reward = -2
                    
            # move to next satellite in the same plane
            elif act == 3:
                next_o = (o + 1) % self.sats_per_plane
                dest = (p, next_o)
                if ((p, o), dest) in self.edges:
                    task.order_at = next_o
                    # clear the layer process progress
                    task.layer_process = 0
                    energy_cost = 0.2
                    reward = -1
                else:
                    reward = -2
                    
            # move to previous satellite in the same plane
            elif act == 4:
                next_y = (o - 1) % self.sats_per_plane
                dest = (p, next_y)
                if ((p, o), dest) in self.edges:
                    task.order_at = next_y
                    # clear the layer process progress
                    task.layer_process = 0
                    energy_cost = 0.2
                    reward = -1
                else:
                    reward = -2
                    
            # perform data processing
            elif act == 5:
                
                # minus the energy cost on processing each time
                energy_cost = 0.5
                reward = -1
                
                # make process plus 1
                task.layer_process += 1
                
                
                # if process reach the condition, the layer_id will be updated
                if task.layer_process > self.layer_process_step_cost[task.layer_id]:
                    task.layer_id += 1
                    task.layer_process = 0
                    reward = 1.0
                
                if task.layer_id >= self.num_layers:
                    task.is_done = True
                    reward = 5.0
                    
            else:
                reward = -5.0
            
            node.energy -= energy_cost
            total_energy_cost += energy_cost
            task.t_end += 1
            total_delay_cost += 1  # 每一步都算延迟
            
            total_reward += reward
            
        # ========== (3) 计算归一化Reward ==========
        num_tasks = len(self.tasks)
        max_proc_cost = 3.0  # 每任务最大处理能耗
        normE = max(1e-6, num_tasks * max_proc_cost)
        normD = max(1e-6, num_tasks * 1.0)

        # 延迟惩罚和能耗惩罚
        w1, w2 = 0.4, 0.6  # 权重
        delay_penalty = total_delay_cost / normD
        energy_penalty = total_energy_cost / normE

        total_reward -= (w1 * delay_penalty + w2 * energy_penalty)

        # ========== (4) 输出 ==========
        is_all_done = all(t.is_done for t in self.tasks)
        obs = self._get_obs()
        info = {'global_time': self.global_time_counter,
                'delay': delay_penalty,
                'energy': energy_penalty}
        return obs, total_reward, is_all_done, False, info

    def _get_obs(self):
        parts: List[float] = []
        for p in range(self.num_planes):
            for s in range(self.sats_per_plane):
                n = self.nodes.get((p, s))
                parts.append(n.energy if n is not None else 0.0)
        parts.append(float(self.global_time_counter))
        for t in self.tasks:
            parts.extend([
                t.id,
                t.layer_id,
                t.layer_process,
                t.plane_at,
                t.order_at,
                t.t_start,
                t.t_end,
                t.is_done,
            ])
        return np.array(parts, dtype=np.float32)

    def _connected(self, u: Tuple[int, int], v: Tuple[int, int]) -> bool:
        return (u, v) in self.edges
    
    def render(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.cm as cm
        # 可选：静态变量保存fig避免重复创建
        if not hasattr(self, '_fig') or not hasattr(self, '_ax'):
            self._fig = plt.figure(figsize=(8, 6))
            self._ax = self._fig.add_subplot(111, projection='3d')
        else:
            self._ax.cla()
        ax = self._ax
        # 1. 绘制所有卫星节点，并显示energy、xyz、gamma
        xs, ys, zs, colors = [], [], [], []
        for (p, s), node in self.nodes.items():
            xs.append(node.x)
            ys.append(node.y)
            zs.append(node.z)
            colors.append('green' if node.gamma else 'gray')
            # 显示卫星属性
            ax.text(node.x, node.y, node.z-2,
                f'{node.energy:.1f}:[{node.plane_id},{node.order_id}]',
                color='black', fontsize=8, ha='center', va='top', alpha=0.7)
        ax.scatter(xs, ys, zs, c=colors, s=40, label='Satellites')

        # 2. 绘制所有边
        for (ua, va), edge in self.edges.items():
            u = edge.u
            v = edge.v
            ax.plot([u.x, v.x], [u.y, v.y], [u.z, v.z], color='gray', alpha=0.5)

        # 3. 绘制任务（不同颜色）
        task_colors = cm.rainbow(np.linspace(0, 1, len(self.tasks)))
        for i, task in enumerate(self.tasks):
            node = self.nodes.get((task.plane_at, task.order_at))
            if node is not None:
                ax.scatter([node.x], [node.y], [node.z], color=task_colors[i], s=120, marker='o', label=f'Task {task.id}')
                # 任务状态文本
                ax.text(node.x, node.y, node.z+2, f'T{task.id} L{task.layer_id} {"Done" if task.is_done else ""}', color=task_colors[i], fontsize=9)

        # 4. 其他美化
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Satellite Network at {self.global_time_counter * self.t_step * self.step_per_slot:.2f} seconds')
        # 图例只显示一次
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.pause(0.05)
        plt.show(block=False)
        