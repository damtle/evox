from __future__ import annotations

import torch
from evox.algorithms import PSO

from rl_ea.rl.agent import TD3Agent, TD3Config
from rl_ea.replay.buffer import PrioritizedReplayBuffer
from rl_ea.algorithms.rl_wrapper import RLEnhancedAlgorithm


def make_rl_pso(
    pop_size: int,
    lb: torch.Tensor,
    ub: torch.Tensor,
    device: torch.device | None = None,
    w: float = 0.6,
    phi_p: float = 2.5,
    phi_g: float = 0.8,
    replay_capacity: int = 200000,
    rl_candidate_ratio: float = 0.3,
    train_after: int = 512,
    train_steps_per_gen: int = 1,
    batch_size: int = 256,
    exploration_noise: float = 0.1,
    enable_rl: bool = True
):
    device = torch.get_default_device() if device is None else device
    base = PSO(pop_size=pop_size, lb=lb, ub=ub, w=w, phi_p=phi_p, phi_g=phi_g, device=device)

    if not enable_rl:
        return base

    dim = lb.shape[0]
    state_dim = 2 * dim + 2
    action_dim = dim
    max_action = (ub - lb).to(device) * 0.1
    agent = TD3Agent(state_dim, action_dim, max_action=max_action, device=device, cfg=TD3Config())
    replay = PrioritizedReplayBuffer(capacity=replay_capacity, device=device)
    algo = RLEnhancedAlgorithm(
        base_algo=base,
        rl_agent=agent,
        replay_buffer=replay,
        rl_candidate_ratio=rl_candidate_ratio,
        train_after=train_after,
        train_steps_per_gen=train_steps_per_gen,
        batch_size=batch_size,
        action_scale=max_action,
        exploration_noise=exploration_noise,
        device=device,
    )
    return algo
