"""PPO rollout buffer with GAE."""
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Transition:
    grid: torch.Tensor
    aux: torch.Tensor
    available_actions: torch.Tensor
    action_type: int
    action_x: int
    action_y: int
    reward: float
    done: bool
    log_prob: float
    value: float


class ReplayBuffer:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.transitions: list[Transition] = []
        self.episode_boundaries: list[int] = [0]
        self._adv: Optional[torch.Tensor] = None
        self._ret: Optional[torch.Tensor] = None

    def add(self, t: Transition):
        self.transitions.append(t)
        if len(self.transitions) > self.max_size:
            self.transitions.pop(0)
            self.episode_boundaries = [max(0, b - 1) for b in self.episode_boundaries]
        self._adv = self._ret = None

    def mark_episode_end(self):
        self.episode_boundaries.append(len(self.transitions))

    def compute_gae(self, gamma=0.99, lam=0.95):
        n = len(self.transitions)
        if n == 0:
            return torch.tensor([]), torch.tensor([])
        adv, ret = torch.zeros(n), torch.zeros(n)
        bounds = sorted(set(self.episode_boundaries))
        if bounds[-1] != n:
            bounds.append(n)
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if s >= e:
                continue
            gae = 0.0
            for t in reversed(range(s, e)):
                tr = self.transitions[t]
                nv = 0.0 if t == e - 1 or tr.done else self.transitions[t + 1].value
                delta = tr.reward + gamma * nv * (1 - float(tr.done)) - tr.value
                gae = delta + gamma * lam * (1 - float(tr.done)) * gae
                adv[t], ret[t] = gae, gae + tr.value
        self._adv, self._ret = adv, ret
        return adv, ret

    def sample_minibatch(self, batch_size: int):
        n = len(self.transitions)
        assert n > 0
        if self._adv is None:
            self.compute_gae()
        idx = np.random.choice(n, min(batch_size, n), replace=False)
        return {
            'grids': torch.stack([self.transitions[i].grid for i in idx]),
            'aux': torch.stack([self.transitions[i].aux for i in idx]),
            'available_actions': torch.stack([self.transitions[i].available_actions for i in idx]),
            'action_types': torch.tensor([self.transitions[i].action_type for i in idx], dtype=torch.long),
            'action_x': torch.tensor([self.transitions[i].action_x for i in idx], dtype=torch.long),
            'action_y': torch.tensor([self.transitions[i].action_y for i in idx], dtype=torch.long),
            'old_log_probs': torch.tensor([self.transitions[i].log_prob for i in idx]),
            'old_values': torch.tensor([self.transitions[i].value for i in idx]),
            'advantages': self._adv[idx],
            'returns': self._ret[idx],
        }

    def clear(self):
        self.transitions.clear()
        self.episode_boundaries = [0]
        self._adv = self._ret = None

    def __len__(self):
        return len(self.transitions)

    def episode_stats(self):
        if not self.transitions:
            return {"num_episodes": 0}
        eps, bounds = [], sorted(set(self.episode_boundaries))
        if bounds[-1] != len(self.transitions):
            bounds.append(len(self.transitions))
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if s >= e:
                continue
            rews = [self.transitions[t].reward for t in range(s, e)]
            eps.append({"length": e - s, "total_reward": sum(rews)})
        if not eps:
            return {"num_episodes": 0}
        return {
            "num_episodes": len(eps),
            "mean_length": np.mean([e["length"] for e in eps]),
            "mean_total_reward": np.mean([e["total_reward"] for e in eps]),
        }
