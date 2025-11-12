import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np

def render_satellite_network(topology, tasks, global_time_counter, t_step):
    if not hasattr(render_satellite_network, '_fig') or not hasattr(render_satellite_network, '_ax'):
        render_satellite_network._fig = plt.figure(figsize=(8, 6))
        render_satellite_network._ax = render_satellite_network._fig.add_subplot(111, projection='3d')
    else:
        render_satellite_network._ax.cla()
    ax = render_satellite_network._ax
    xs, ys, zs, colors = [], [], [], []
    for (p, s), node in topology.nodes.items():
        xs.append(node.x)
        ys.append(node.y)
        zs.append(node.z)
        colors.append('green' if node.gamma else 'gray')
        ax.text(node.x, node.y, node.z-2,
            f'{node.energy:.1f}:[{node.plane_id},{node.order_id}]',
            color='black', fontsize=8, ha='center', va='top', alpha=0.7)
    ax.scatter(xs, ys, zs, c=colors, s=40, label='Satellites')
    for (ua, va), edge in topology.edges.items():
        u = edge.u
        v = edge.v
        ax.plot([u.x, v.x], [u.y, v.y], [u.z, v.z], color='gray', alpha=0.5)
    task_colors = cm.rainbow(np.linspace(0, 1, len(tasks)))
    for i, task in enumerate(tasks):
        node = topology.nodes.get((task.plane_at, task.order_at))
        if node is not None:
            ax.scatter([node.x], [node.y], [node.z], color=task_colors[i], s=120, marker='o', label=f'Task {task.id}')
            ax.text(node.x, node.y, node.z+2, f'T{task.id} L{task.layer_id} {"Done" if task.is_done else ""}', color=task_colors[i], fontsize=9)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Satellite Network at {global_time_counter * t_step:.2f} seconds')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.pause(0.05)
    plt.show(block=False)
