#!/usr/bin/env python3
"""
Compact parser that reads `data.json` and writes `parsed_orbital_data.json`.

Output shape (requested):
- nodes: satellites only, each with keys:
  - index, plane_id, order_id, gamma (0/1), energy (int), x, y, z
- edges: ISL list with keys: index, u, v, rate (float|null)

nodes: list of satellites only, each node is a dict with
  - index: int
  - plane_id: int
  - order_id: int
  - gamma: 0/1 (sunlit flag)
  - energy: int (remaining energy, default 100)
  - x: float
  - y: float
  - z: float

edges: list of ISL links between satellites, each edge is a dict with
  - index: int
  - u: index of u satellite
  - v: index of v satellite
  - rate: numeric transmission rate (float) or null

Run:
  ./parse.py

Assumptions:
- Input: raw_constellation_data/data.json
- If plane/order not found, set to -1
- energy default 100
- ISL detection uses geometric fields (linkPos, links, downlinkPos) and
  falls back to top-level links with rates assigned in order when counts match.
"""

from pathlib import Path
import json
import math
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / 'data.json'
OUT_PATH = HERE / 'parsed_orbital_data.json'


def _dist(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.sqrt((a.get('x', 0.0) - b.get('x', 0.0)) ** 2 + (a.get('y', 0.0) - b.get('y', 0.0)) ** 2 + (a.get('z', 0.0) - b.get('z', 0.0)) ** 2)


def load_raw(p: Path = DATA_PATH) -> Dict[str, Any]:
    with open(p, 'r') as f:
        return json.load(f)


def extract_satellites(raw: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    sats = raw.get('satellites', []) or []
    nodes: List[Dict[str, Any]] = []
    id_map: Dict[str, int] = {}
    for i, s in enumerate(sats):
        sid = s.get('id') or f'sat_{i}'
        # preserve numeric 0 values; check explicitly for None
        plane = s.get('plane') if 'plane' in s else s.get('planeId') if 'planeId' in s else None
        order = s.get('order') if 'order' in s else s.get('orderId') if 'orderId' in s else None
        try:
            plane_id = int(plane) if plane is not None else -1
        except Exception:
            plane_id = -1
        try:
            order_id = int(order) if order is not None else -1
        except Exception:
            order_id = -1
        pos = s.get('pos') or {}
        node = {
            'index': i,
            'plane_id': plane_id,
            'order_id': order_id,
            'gamma': 1 if s.get('onSun') else 0,
            'energy': 100,
            'x': float(pos.get('x', 0.0)),
            'y': float(pos.get('y', 0.0)),
            'z': float(pos.get('z', 0.0)),
            '_pos': {'x': pos.get('x', 0.0), 'y': pos.get('y', 0.0), 'z': pos.get('z', 0.0)}
        }
        nodes.append(node)
        id_map[sid] = i
    return nodes, id_map


def _find_nearest(nodes: List[Dict[str, Any]], coord: Dict[str, float]) -> int:
    return min(range(len(nodes)), key=lambda i: _dist(nodes[i]['_pos'], coord))


def detect_isl_pairs(raw: Dict[str, Any], nodes: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    seen = set()
    for section in (raw.get('satellites', []) or [], raw.get('stations', []) or []):
        for item in section:
            for key in ('linkPos', 'links', 'downlinkPos'):
                if key not in item:
                    continue
                val = item[key]
                if not isinstance(val, list) or not val:
                    continue
                if all(isinstance(v, dict) and 'x' in v for v in val):
                    if len(val) >= 2:
                        if len(val) == 2:
                            na = _find_nearest(nodes, val[0])
                            nb = _find_nearest(nodes, val[1])
                            if na != nb:
                                k = tuple(sorted((na, nb)))
                                if k not in seen:
                                    seen.add(k)
                                    pairs.append((na, nb))
                        else:
                            for i in range(0, len(val) - 1, 2):
                                acoord = val[i]
                                bcoord = val[i+1]
                                na = _find_nearest(nodes, acoord)
                                nb = _find_nearest(nodes, bcoord)
                                if na != nb:
                                    k = tuple(sorted((na, nb)))
                                    if k not in seen:
                                        seen.add(k)
                                        pairs.append((na, nb))
                    continue
                for element in val:
                    if isinstance(element, list) and len(element) >= 2 and isinstance(element[0], dict):
                        acoord = element[0]
                        bcoord = element[1]
                        na = _find_nearest(nodes, acoord)
                        nb = _find_nearest(nodes, bcoord)
                        if na != nb:
                            k = tuple(sorted((na, nb)))
                            if k not in seen:
                                seen.add(k)
                                pairs.append((na, nb))
    # fallback: top-level links with linkPos
    for link in raw.get('links', []) or []:
        if isinstance(link, dict) and 'linkPos' in link and isinstance(link['linkPos'], list) and len(link['linkPos']) >= 2:
            acoord = link['linkPos'][0]
            bcoord = link['linkPos'][1]
            na = _find_nearest(nodes, acoord)
            nb = _find_nearest(nodes, bcoord)
            if na != nb:
                k = tuple(sorted((na, nb)))
                if k not in seen:
                    seen.add(k)
                    pairs.append((na, nb))
    return pairs


def extract_rates(raw: Dict[str, Any], edges: List[Tuple[int, int]]) -> List[Optional[float]]:
    top_links = raw.get('links', []) or []
    numeric_rates = [l.get('rate') for l in top_links if isinstance(l, dict) and l.get('rate') is not None]
    if len(numeric_rates) >= len(edges) and len(edges) > 0:
        return numeric_rates[:len(edges)]
    return [None] * len(edges)


def build_parsed(path: Path = DATA_PATH, out: Path = OUT_PATH) -> Dict[str, Any]:
    raw = load_raw(path)
    nodes_raw, id_map = extract_satellites(raw)
    nodes = []
    for n in nodes_raw:
        nodes.append({
            'index': n['index'],
            'plane_id': n['plane_id'],
            'order_id': n['order_id'],
            'gamma': int(n['gamma']),
            'energy': int(n['energy']),
            'x': float(n['x']),
            'y': float(n['y']),
            'z': float(n['z'])
        })
    edges_pairs = detect_isl_pairs(raw, nodes_raw)
    rates = extract_rates(raw, edges_pairs)
    edges = []
    for i, ((u, v), rate) in enumerate(zip(edges_pairs, rates)):
        edges.append({'index': i, 'u': int(u), 'v': int(v), 'rate': rate})
    parsed = {'nodes': nodes, 'edges': edges}
    with open(out, 'w') as f:
        json.dump(parsed, f, indent=2)
    return parsed


def main():
    parsed = build_parsed()
    print(f'Wrote {OUT_PATH} with {len(parsed["nodes"])} nodes and {len(parsed["edges"])} edges')


if __name__ == '__main__':
    main()
