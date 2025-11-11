from pydantic import BaseModel
from envs.object.node import Node

"""
edges: list of ISL links between satellites, each edge is a dict with
- index: int
- u: index of u satellite
- v: index of v satellite
- rate: numeric transmission rate (float) or null
"""

class Edge(BaseModel):
    id: int
    u: Node
    v: Node
    rate: float