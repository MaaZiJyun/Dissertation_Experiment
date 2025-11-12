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
from envs.param import DATA_COMPUTE_PENALTY, DATA_TRANSFER_PENALTY, LAYER_COMPLETION_REWARD, NO_ACTION_PENALTY, TASK_COMPLETION_REWARD, TIME_PENALTY


class LEOEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, json_path: str):
        super().__init__()
        
        # obtain the file path of the JSON topology
        self.json_path = Path(json_path)
        
        # parameters
        self.num_layers = 5
        self.num_tasks = 2
        self.t_slot = 32 #seconds per slot
        self.isl_bit_per_slot = 1_000_000_000 * self.t_slot # bits/time_slot
        self.x = 2_400_000_000 # image raw data -> bits

        # time cost per layer updating
        self.layer_second_cost = [8, 5, 5, 3, 1] # in seconds
        self.layer_output_bit = [self.x, self.x / 1e2, self.x / 1e5, self.x / 1e9, 8]

        # precision
        self.step_per_slot = 320 # steps per slot
        
        # step variables
        self.t_step = self.t_slot / self.step_per_slot # seconds per step
        self.isl_bit_per_step = self.isl_bit_per_slot / self.step_per_slot
        self.layer_process_step_cost = [math.ceil(i / self.t_step) for i in self.layer_second_cost]

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
                layer_process=0,
                link_process=0,
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

        total_reward = 0.0
        total_energy_cost = 0.0
        total_delay_cost = 0.0
        
        reward = TIME_PENALTY
        energy_cost = 0.0
        
        # ========== (1) 更新能量 ==========
        for k, node in self.nodes.items():
            node.energy -= 0.01  # 静态损耗
            if int(node.gamma) == 1:  # 光照下充电
                node.energy = min(node.energy + 0.2, 100.0)

        # ========== (2) 统计所有传输请求 ==========
        transfer_reqs = []  # [(task, src, dst, data_size_bits)]
        for task, act in zip(self.tasks, actions):
            if task.t_start > self.global_time_counter or task.is_done:
                continue
            
            p, o = task.plane_at, task.order_at
            node = self.nodes.get((p, o))
            if node is None:
                continue

            # move actions 1-4 会产生传输需求
            if act in [1, 2, 3, 4]:
                
                task.layer_process = 0  # 传输完清空计算进度
                
                if act == 1:
                    dst = ((p + 1) % self.num_planes, o)
                elif act == 2:
                    dst = ((p - 1) % self.num_planes, o)
                elif act == 3:
                    dst = (p, (o + 1) % self.sats_per_plane)
                else:
                    dst = (p, (o - 1) % self.sats_per_plane)
                
                if ((p, o), dst) in self.edges:
                    
                    # get the data size of the current layer output
                    data_bits = self.layer_output_bit[task.layer_id]
                    
                    # append transfer request
                    transfer_reqs.append((task, (p, o), dst, data_bits))
        
        # ========== (3) 执行传输：按任务大小分配链路带宽 ==========
        edge_transfers: Dict[Tuple[Tuple[int,int],Tuple[int,int]], List[Tuple[Task, float]]] = {}

        for task, src, dst, data_bits in transfer_reqs:
            edge_key = (src, dst)
            edge_transfers.setdefault(edge_key, []).append((task, data_bits))

        # 按比例分配带宽
        for edge_key, task_list in edge_transfers.items():
            edge = self.edges.get(edge_key)
            if not edge or edge.rate is None:
                continue

            total_data = sum(bits for _, bits in task_list)
            for task, data_bits in task_list:
                r = data_bits / total_data
                # 分配得到的带宽比例
                c = self.isl_bit_per_step * r
                task.link_process += c
                # 若带宽足够，传输完成
                if task.link_process >= data_bits:
                    task.plane_at, task.order_at = edge_key[1]
                    task.link_process = 0  # 传输完清空链路进度
                    reward = DATA_TRANSFER_PENALTY
                else:
                    # 传不完则下次继续
                    remain_ratio = 1 - c / data_bits
                    reward = DATA_TRANSFER_PENALTY * remain_ratio
                energy_cost = 0.2

        # ========== (4) 执行动作 ==========
        for task, act in zip(self.tasks, actions):
            if task.t_start > self.global_time_counter or task.is_done:
                continue

            p, o = task.plane_at, task.order_at
            node = self.nodes.get((p, o))
            if node is None:
                continue

            # act == 0: 不动
            if act == 0:
                reward = NO_ACTION_PENALTY

            # act == 5: 执行推理
            elif act == 5:
                energy_cost = 1
                task.layer_process += 1
                reward = DATA_COMPUTE_PENALTY

                # 达到层处理要求则层数+1
                if task.layer_process >= self.layer_process_step_cost[task.layer_id]:
                    task.layer_id += 1
                    task.layer_process = 0
                    reward = LAYER_COMPLETION_REWARD

                # 全部完成
                if task.layer_id >= self.num_layers:
                    task.t_end = self.global_time_counter
                    task.is_done = True
                    reward = TASK_COMPLETION_REWARD
            else:
                # 传输动作已在上面处理
                pass

            node.energy -= energy_cost
            total_energy_cost += energy_cost
            total_delay_cost += 1
            total_reward += reward

        # ========== (5) 计算归一化Reward ==========
        num_done_task = sum(t.is_done for t in self.tasks)
        
        avg_delay = self.step_per_slot
        if num_done_task > 0:
            avg_delay = sum(t.t_end - t.t_start for t in self.tasks if t.is_done) / num_done_task
        
        delay_penalty = (self.step_per_slot - avg_delay) / self.step_per_slot
        
        avg_energy = 100
        if num_done_task > 0:
            avg_energy = sum(n.energy for _, n in self.nodes.items()) / len(self.nodes)

        w1, w2 = 0.5, 0.5
        energy_penalty = avg_energy / 100
        
        total_reward += (w1 * delay_penalty + w2 * energy_penalty)

        # ========== (6) 输出 ==========
        is_all_done = all(t.is_done for t in self.tasks)
        
        # 终止条件
        done = False
        success = False
        fail_reason = None

        # 1. step 超过 step_per_slot
        if self.global_time_counter >= self.step_per_slot:
            done = True
            fail_reason = "step_limit"
            
        # 2. 任意卫星电量为0
        elif any(n.energy <= 0 for n in self.nodes.values()):
            done = True
            fail_reason = "energy_depleted"
            
        # 3. 任务全部完成
        elif is_all_done:
            done = True
            success = True

        obs = self._get_obs()
        info = {
            'global_time': self.global_time_counter,
            'delay': delay_penalty,
            'energy': energy_penalty,
            'success': success,
            'fail_reason': fail_reason
        }
        self.global_time_counter += 1
        return obs, total_reward, done, success, info


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
        ax.set_title(f'Satellite Network at {self.global_time_counter * self.t_step:.2f} seconds')
        # 图例只显示一次
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.pause(0.05)
        plt.show(block=False)
        