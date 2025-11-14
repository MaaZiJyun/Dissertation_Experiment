
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from envs.snapshot.node import Node
from envs.snapshot.edge import Edge

class JsonManager:
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.nodes: Dict[Tuple[int, int], Node] = {}
        self.edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Edge] = {}
        self.N_SAT = 0
        self.N_PLANE = 0
        self._load_json_from_file()
        
    def reset(self):
        self.nodes.clear()
        self.edges.clear()
        self.N_SAT = 0
        self.N_PLANE = 0
        self._load_json_from_file()
        
    def _load_json_from_file(self) -> dict:
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        self._convert(data)

    def _convert(self, data: dict):
        index_map = {}
        for node in data.get('nodes', []):
            pid = int(node['plane_id'])
            sid = int(node['order_id'])
            node_obj = Node.model_validate({
                'id': node.get('index', -1),
                'plane_id': pid,
                'order_id': sid,
                'energy': float(np.random.uniform(80, 100)),
                'gamma': bool(int(node.get('gamma', 0))),
                'x': node.get('x', 0.0),
                'y': node.get('y', 0.0),
                'z': node.get('z', 0.0)
            })
            self.nodes[(pid, sid)] = node_obj
            if 'index' in node:
                index_map[node['index']] = (pid, sid)
                
        for edge in data.get('edges', []):
            uid = edge['u']
            vid = edge['v']
            rate = edge.get('rate', None)
            if uid not in index_map or vid not in index_map:
                continue
            u = self.nodes[index_map[uid]]
            v = self.nodes[index_map[vid]]
            edge_obj = Edge.model_validate({
                'id': edge.get('index', -1),
                'u': u,
                'v': v,
                'rate': rate
            })
            self.edges[(index_map[uid], index_map[vid])] = edge_obj
            self.edges[(index_map[vid], index_map[uid])] = edge_obj
        self.N_SAT = max(s for (p, s) in self.nodes.keys()) + 1 if self.nodes else 0
        self.N_PLANE = max(p for (p, s) in self.nodes.keys()) + 1 if self.nodes else 0
        
    def get_nodes(self) -> List[Node]:
        return list(self.nodes.values())
    
    def get_edges(self) -> List[Edge]:
        return list(self.edges.values())
    
    def get_node_keys(self) -> List[Tuple[int, int]]:
        return list(self.nodes.keys())
    
    def get_edge_keys(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        return list(self.edges.keys())

    def connected(self, u, v):
        return (u, v) in self.edges
