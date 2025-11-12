
import json
from pathlib import Path
from typing import Dict, Tuple, List
from envs.snapshot.node import Node
from envs.snapshot.edge import Edge

class TopologyManager:
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.nodes: Dict[Tuple[int, int], Node] = {}
        self.edges: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Edge] = {}
        self.planes = []
        self.sats_per_plane = 0
        self.num_planes = 0
        self._load_topology()

    def _load_topology(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        index_map = {}
        for node in data.get('nodes', []):
            pid = int(node['plane_id'])
            sid = int(node['order_id'])
            node_obj = Node.model_validate({
                'id': node.get('index', -1),
                'plane_id': pid,
                'order_id': sid,
                'energy': node.get('energy', 100.0),
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
        self.planes = sorted({p for (p, s) in self.nodes.keys()})
        self.sats_per_plane = max(s for (p, s) in self.nodes.keys()) + 1 if self.nodes else 0
        self.num_planes = max(self.planes) + 1 if self.planes else 0

    def connected(self, u, v):
        return (u, v) in self.edges
