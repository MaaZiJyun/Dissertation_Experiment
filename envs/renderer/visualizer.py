from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np

from envs.core.json_manager import JsonManager
from envs.param import STEP_PER_SECOND, T_STEP
from envs.snapshot.edge import Edge
from envs.snapshot.node import Node
from envs.snapshot.task import Task

_fig = None
_ax = None

def render_satellite_network(
    nodes: List[Node],
    edges: List[Edge],
    tasks: List[Task], 
    step_counter: int, 
):
    global _fig, _ax
    if _fig is None or _ax is None:
        _fig = plt.figure(figsize=(8, 6))
        _ax = _fig.add_subplot(111, projection='3d')
    else:
        _ax.cla()
    ax = _ax
    xs, ys, zs, colors = [], [], [], []
    for node in nodes:
        xs.append(node.x)
        ys.append(node.y)
        zs.append(node.z)
        colors.append('green' if node.gamma else 'gray')
        # ax.text(node.x, node.y, node.z-2,
        #     f'{node.energy:.1f}:[{p},{o}]',
        #     color='black', fontsize=8, ha='center', va='top', alpha=0.7)
    ax.scatter(xs, ys, zs, c=colors, s=40, label='Satellites')
    for edge in edges:
        u = edge.u
        v = edge.v
        ax.plot([u.x, v.x], [u.y, v.y], [u.z, v.z], color='gray', alpha=0.5)
    task_colors = cm.rainbow(np.linspace(0, 1, len(tasks)))
    for i, task in enumerate(tasks):
        node = next((n for n in nodes if n.plane_id == task.plane_at and n.order_id == task.order_at), None)
        if node is not None:
            ax.scatter([node.x], [node.y], [node.z], color=task_colors[i], s=120, marker='o', label=f'Task {task.id}')
            ax.text(node.x, node.y, node.z+2, f'm={task.id}, n={task.layer_id}, ts={task.t_start / STEP_PER_SECOND}, td={task.t_end / STEP_PER_SECOND}: [{task.act}] ({task.workload_done}|{task.data_sent})', color=task_colors[i], fontsize=9)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Satellite Network at {step_counter / STEP_PER_SECOND:.2f} seconds')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.pause(0.01)
    plt.show(block=False)
