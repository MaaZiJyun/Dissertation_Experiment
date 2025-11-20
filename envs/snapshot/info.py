from typing import Dict
import numpy as np

"""
edges: list of ISL links between satellites, each edge is a dict with
- index: int
- u: index of u satellite
- v: index of v satellite
- rate: numeric transmission rate (float) or null
"""

class Info():
    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        num_tasks: int,
        step: int,
        alpha: Dict[str, np.ndarray],
        beta: Dict[str, np.ndarray],
        reward: float = 0.0,
        is_truncated: bool = False,
        truncated_reason: str = "None",
        is_terminated: bool = False,
        terminated_reason: str = "None",
        **kwargs,
    ):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_tasks = num_tasks
        self.step = step
        self.alpha = alpha
        self.beta = beta
    # allow initializing these fields directly if provided
        self.reward = float(reward)
        self.is_truncated = bool(is_truncated)
        self.truncated_reason = str(truncated_reason)
        self.is_terminated = bool(is_terminated)
        self.terminated_reason = str(terminated_reason)

    def to_serializable(self) -> Dict[str, object]:
        """Return a JSON-serializable dict representation of this Info.

        Numpy arrays in alpha/beta are converted to Python lists.
        """
        def _conv(obj):
            try:
                if isinstance(obj, dict):
                    return {k: _conv(v) for k, v in obj.items()}
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                # fallback for scalars
                return obj
            except Exception:
                return str(obj)

        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_tasks": self.num_tasks,
            "step": self.step,
            "alpha": _conv(self.alpha),
            "beta": _conv(self.beta),
            "reward": float(self.reward),
            "is_truncated": bool(self.is_truncated),
            "truncated_reason": str(self.truncated_reason),
            "is_terminated": bool(self.is_terminated),
            "terminated_reason": str(self.terminated_reason),
        }

    def pretty(self, max_items: int = 6) -> str:
        """Return a short human-friendly string summarising the Info.

        For large arrays, only the first `max_items` entries are shown.
        """
        s = [f"Step={self.step}, nodes={self.num_nodes}, edges={self.num_edges}, tasks={self.num_tasks}"]
        s.append(f"reward={self.reward:.4f}")
        s.append(f"truncated={self.is_truncated}({self.truncated_reason}), terminated={self.is_terminated}({self.terminated_reason})")

        # alpha/beta summary
        try:
            a_keys = list(self.alpha.keys()) if isinstance(self.alpha, dict) else []
            b_keys = list(self.beta.keys()) if isinstance(self.beta, dict) else []
            s.append(f"alpha_keys={a_keys}")
            s.append(f"beta_keys={b_keys}")
        except Exception:
            s.append("alpha/beta summary unavailable")

        return " | ".join(s)

    def __repr__(self) -> str:
        return f"Info(step={self.step}, reward={self.reward:.4f}, tasks={self.num_tasks})"