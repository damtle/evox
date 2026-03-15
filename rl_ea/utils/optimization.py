from __future__ import annotations

import torch


def relative_improvement_reward(curr_fit: torch.Tensor, next_fit: torch.Tensor, global_best: torch.Tensor,
                                eps: float = 1e-8, best_bonus: float = 0.1) -> torch.Tensor:
    """Relative-improvement reward for minimization."""
    reward = (curr_fit - next_fit) / (curr_fit.abs() + eps)

    # 【关键修复】：限制单步奖励的范围，防止极端负奖励摧毁 Critic 网络
    reward = torch.clamp(reward, min=-1.0, max=1.0)

    reward = reward.clone()
    improved = next_fit < global_best
    reward[improved] += best_bonus
    return reward


def compute_future_improvement(fits: torch.Tensor) -> torch.Tensor:
    """For a trajectory fitness sequence [f0, ..., fT], return future improvements
    fi - min_{k>=i} fk for each i.
    """
    best_suffix = torch.minimum(torch.flip(torch.cummin(torch.flip(fits, dims=[0]), dim=0).values, dims=[0]), fits)
    return fits - best_suffix
