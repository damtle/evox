from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch


@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor
    future_improvement: torch.Tensor
    td_priority: float
    source: int  # 0 = EA, 1 = RL
    traj_id: int
    step_id: int


class PrioritizedReplayBuffer:
    """Replay buffer specialized for optimization traces.

    Priorities combine TD-style priority and future-improvement signal:
        p_i ∝ alpha * td_priority + beta * future_improvement
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.7,
        beta: float = 0.3,
        device: Optional[torch.device] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.device = device if device is not None else torch.device('cpu')
        self.data: List[Transition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, transition: Transition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition
            self.pos = (self.pos + 1) % self.capacity

    def _priority_array(self) -> np.ndarray:
        if not self.data:
            return np.array([], dtype=np.float64)
        priorities = []
        for t in self.data:
            td_term = max(float(t.td_priority), 1e-6)
            future_term = max(float(t.future_improvement.item()), 0.0)
            p = self.alpha * td_term + self.beta * future_term + 1e-6
            priorities.append(p)
        arr = np.asarray(priorities, dtype=np.float64)
        arr /= arr.sum()
        return arr

    def sample(self, batch_size: int, rl_ratio: Optional[float] = None) -> Dict[str, torch.Tensor]:
        if len(self.data) < batch_size:
            raise ValueError(f"Not enough data in replay buffer: {len(self.data)} < {batch_size}")

        if rl_ratio is None:
            indices = np.random.choice(len(self.data), size=batch_size, replace=False, p=self._priority_array())
        else:
            rl_ratio = float(rl_ratio)
            rl_idx = [i for i, t in enumerate(self.data) if t.source == 1]
            ea_idx = [i for i, t in enumerate(self.data) if t.source == 0]
            n_rl = min(len(rl_idx), max(0, int(round(batch_size * rl_ratio))))
            n_ea = batch_size - n_rl
            if len(ea_idx) < n_ea:
                n_ea = len(ea_idx)
                n_rl = batch_size - n_ea
            if len(rl_idx) < n_rl:
                n_rl = len(rl_idx)
                n_ea = batch_size - n_rl
            p = self._priority_array()
            chosen = []
            if n_rl > 0:
                p_rl = p[rl_idx]
                p_rl = p_rl / p_rl.sum()
                chosen.extend(np.random.choice(rl_idx, size=n_rl, replace=False, p=p_rl).tolist())
            if n_ea > 0:
                p_ea = p[ea_idx]
                p_ea = p_ea / p_ea.sum()
                chosen.extend(np.random.choice(ea_idx, size=n_ea, replace=False, p=p_ea).tolist())
            indices = np.asarray(chosen, dtype=np.int64)
            np.random.shuffle(indices)

        batch = [self.data[i] for i in indices]
        return {
            'states': torch.stack([t.state for t in batch]).to(self.device),
            'actions': torch.stack([t.action for t in batch]).to(self.device),
            'rewards': torch.stack([t.reward for t in batch]).to(self.device),
            'next_states': torch.stack([t.next_state for t in batch]).to(self.device),
            'dones': torch.stack([t.done for t in batch]).to(self.device),
            'future_improvements': torch.stack([t.future_improvement for t in batch]).to(self.device),
            # 【新增】：把 source (0为EA, 1为RL) 传给 agent 做区分训练
            'sources': torch.as_tensor([t.source for t in batch], device=self.device, dtype=torch.long),
            'indices': torch.as_tensor(indices, device=self.device, dtype=torch.long),
        }

    def update_td_priorities(self, indices: Sequence[int], td_errors: torch.Tensor) -> None:
        td = td_errors.detach().abs().flatten().cpu().numpy()
        for i, err in zip(indices, td):
            if 0 <= int(i) < len(self.data):
                t = self.data[int(i)]
                t.td_priority = float(err) + 1e-6
