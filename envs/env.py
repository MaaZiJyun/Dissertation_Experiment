import gym
from gym import spaces
import numpy as np
import json
import random

class SatelliteEnv(gym.Env):
    def __init__(self, json_path="topology.json", n_planes=5, n_sats_per_plane=10):
        super(SatelliteEnv, self).__init__()

        # === 加载拓扑结构 ===
        with open(json_path, 'r') as f:
            topo = json.load(f)
        self.edges = topo["edges"]     # list of edges [[p1, s1, p2, s2, rate], ...]
        self.sunlight = topo["sunlight"]  # 光照状态: [ [bool,...], ... ]

        # === 参数 ===
        self.n_planes = n_planes
        self.n_sats = n_sats_per_plane
        self.total_sats = n_planes * n_sats_per_plane
        self.global_time = 0

        # === 动作空间 ===
        # 动作编号: 0 idle, 1 cross+, 2 cross-, 3 along+, 4 along-, 5 process
        self.action_space = spaces.Discrete(6)

        # === 状态空间 ===
        # 电量(每卫星) + 当前时间 + 所有任务信息(4任务 * [m,n,x,y,latency])
        obs_len = self.total_sats + 1 + 4*5
        self.observation_space = spaces.Box(
            low=0, high=200, shape=(obs_len,), dtype=np.float32
        )

        self.reset()


    # -----------------------------
    def reset(self):
        # 电量初始化
        self.energy = np.random.randint(50, 100, size=(self.n_planes, self.n_sats)).astype(float)
        
        # 初始化任务（4个任务）
        self.tasks = []
        for m in range(4):
            x = np.random.randint(0, self.n_planes)
            y = np.random.randint(0, self.n_sats)
            task = {
                "m": m,         # 任务id
                "n": 0,         # 当前layer
                "x": x,
                "y": y,
                "latency": self.global_time
            }
            self.tasks.append(task)
        
        self.global_time = 0
        self.done = False
        return self._get_obs()


    # -----------------------------
    def _get_obs(self):
        obs = []
        obs.extend(self.energy.flatten())  # 电量信息
        obs.append(self.global_time)       # 当前时间
        for task in self.tasks:
            obs.extend([task["m"], task["n"], task["x"], task["y"], task["latency"]])
        return np.array(obs, dtype=np.float32)


    # -----------------------------
    def step(self, actions):
        """
        actions: list of length 4, 每个任务一个动作
        """
        assert len(actions) == len(self.tasks)

        total_reward = 0
        self.global_time += 1

        # === 光照充电 ===
        for p in range(self.n_planes):
            for s in range(self.n_sats):
                if self.sunlight[p][s]:
                    self.energy[p][s] = min(100, self.energy[p][s] + 10)

        # === 对每个任务执行动作 ===
        for task, action in zip(self.tasks, actions):
            px, sy = task["x"], task["y"]
            rate = self._get_rate(px, sy)

            # 动作逻辑
            if action == 0:  # idle
                reward = -1
            elif action == 1:  # cross orbit +
                nx = (px + 1) % self.n_planes
                if self._connected(px, sy, nx, sy):
                    task["x"] = nx
                    reward = -rate
                    self.energy[px][sy] -= 2
                else:
                    reward = -5  # 不可行惩罚
            elif action == 2:  # cross orbit -
                nx = (px - 1) % self.n_planes
                if self._connected(px, sy, nx, sy):
                    task["x"] = nx
                    reward = -rate
                    self.energy[px][sy] -= 2
                else:
                    reward = -5
            elif action == 3:  # along orbit +
                ny = (sy + 1) % self.n_sats
                if self._connected(px, sy, px, ny):
                    task["y"] = ny
                    reward = -rate
                    self.energy[px][sy] -= 1
                else:
                    reward = -5
            elif action == 4:  # along orbit -
                ny = (sy - 1) % self.n_sats
                if self._connected(px, sy, px, ny):
                    task["y"] = ny
                    reward = -rate
                    self.energy[px][sy] -= 1
                else:
                    reward = -5
            elif action == 5:  # process layer
                self.energy[px][sy] -= 5 / (task["n"] + 1)
                reward = 10 / (task["n"] + 1)
                task["n"] += 1
                if task["n"] >= 6:
                    reward += 100  # 完成任务
                    task["done"] = True
            else:
                reward = -10

            task["latency"] += 1
            total_reward += reward

        # === 检查终止条件 ===
        all_done = all(t.get("done", False) for t in self.tasks)
        if all_done:
            self.done = True

        obs = self._get_obs()
        info = {"time": self.global_time}

        return obs, total_reward, self.done, info


    # -----------------------------
    def _connected(self, p1, s1, p2, s2):
        for e in self.edges:
            if e[:4] == [p1, s1, p2, s2] or e[:4] == [p2, s2, p1, s1]:
                return True
        return False

    def _get_rate(self, p, s):
        # 查找所有出边的平均rate（简化）
        rates = [e[4] for e in self.edges if e[0] == p and e[1] == s]
        return np.mean(rates) if rates else 1.0
