# backend/rl_agent.py
from typing import Tuple
from .config import TEMPLATES
from .memory_manager import read_rl_memory, write_rl_memory
import random

class RLAgent:
    """
    Very simple template-based policy with epsilon-greedy selection.
    Stores template statistics in rl_memory.json.
    """

    def __init__(self):
        mem = read_rl_memory()
        self.epsilon = mem.get("epsilon", 0.25)
        # ensure stats exist
        stats = mem.get("template_stats", {})
        for i in range(len(TEMPLATES)):
            stats.setdefault(str(i), {"count":0, "sum_reward":0.0})
        mem["template_stats"] = stats
        write_rl_memory(mem)

    def select(self) -> Tuple[int, str]:
        mem = read_rl_memory()
        stats = mem["template_stats"]
        # epsilon-greedy
        import random
        if random.random() < self.epsilon:
            idx = random.randrange(len(TEMPLATES))
            return idx, TEMPLATES[idx]
        # pick best avg reward
        best_idx = 0
        best_avg = float("-inf")
        for i in range(len(TEMPLATES)):
            s = stats.get(str(i), {"count":0,"sum_reward":0.0})
            avg = s["sum_reward"] / s["count"] if s["count"]>0 else 0.0
            if avg > best_avg:
                best_avg = avg
                best_idx = i
        return best_idx, TEMPLATES[best_idx]

    def update(self, template_idx: int, reward: float):
        mem = read_rl_memory()
        stats = mem["template_stats"]
        key = str(template_idx)
        if key not in stats:
            stats[key] = {"count":1, "sum_reward": float(reward)}
        else:
            stats[key]["count"] += 1
            stats[key]["sum_reward"] += float(reward)
        mem["template_stats"] = stats
        write_rl_memory(mem)
