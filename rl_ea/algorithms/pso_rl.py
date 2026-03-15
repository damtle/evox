from __future__ import annotations

import torch
from evox.algorithms import PSO, SaDE

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
    rl_candidate_ratio: float = 0.01,
    train_after: int = 512,
    train_steps_per_gen: int = 5,
    batch_size: int = 256,
    exploration_noise: float = 0.05,
    enable_rl: bool = True,
    warmup_gens: int = 200
):
    device = torch.get_default_device() if device is None else device
    # base = PSO(pop_size=pop_size, lb=lb, ub=ub, w=w, phi_p=phi_p, phi_g=phi_g, device=device)
    base = SaDE(pop_size, lb, ub, device=device)
    if not enable_rl:
        return base

    dim = lb.shape[0]
    state_dim = 2 * dim + 2
    action_dim = dim
    max_action = (ub - lb).to(device) * 0.01
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
        warmup_gens=warmup_gens,
    )
    return algo
