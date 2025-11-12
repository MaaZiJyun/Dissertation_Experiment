from pydantic import BaseModel

"""
nodes: list of satellites only, each node is a dict with
- id: int
- plane_id: int
- order_id: int
- gamma: 0/1 (sunlit flag)
- energy: int (remaining energy, default 100)
- x: float
- y: float
- z: float
"""

class Node(BaseModel):
    id: int
    plane_id: int
    order_id: int
    gamma: bool
    energy: float
    x: float
    y: float
    z: float