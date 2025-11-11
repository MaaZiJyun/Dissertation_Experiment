这是一个中心化智能体系统

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

Action space (per satellite per step): 
1. Idle (do nothing)
2. Cross-orbit forward propagation (+x) 
3. Cross-orbit backward propagation (-x) 
4. Along-orbit forward propagation (+y) 
5. Along-orbit backward propagation (-y) 
6. Process task (execute one model layer)

Each task record h = {m, n, x, y, latency} 
- m: task index
- n: current layer index
- x: plane index
- y: satellite index 
- latency: accumulated delay

Initialization

读取json信息，得到所有nodes和edges信息（并不是所有卫星之间都有链接，1234动作传播只能沿edges里面的，不能再不存在的边传播）；所以edges的通信速度，所有卫星光照条件和电量； 1. 全局随机生成4个任务位于不同的4个卫星上（m=0,1,2,3; n=0; latency= current global time）。 2. 每个卫星赋予随机电量energy(50, 100] 对于step，每一个step： global time都会+1； 有光照的卫星energy+10；