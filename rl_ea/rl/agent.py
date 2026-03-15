from __future__ import annotations

from dataclasses import dataclass
import copy
import torch
import torch.nn.functional as F
from .networks import Actor, Critic


@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    hidden_dim: int = 256


class TD3Agent:
    def __init__(self, state_dim: int, action_dim: int, max_action: torch.Tensor, device: torch.device, cfg: TD3Config | None = None):
        self.cfg = cfg or TD3Config()
        self.device = device
        self.max_action = max_action.to(device)

        self.actor = Actor(state_dim, action_dim, self.cfg.hidden_dim, self.max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)

        self.critic = Critic(state_dim, action_dim, self.cfg.hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        self.total_updates = 0
        self.last_actor_loss = 0.0  # 【新增】

    def act(self, state: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state.to(self.device))
            if noise_std > 0:
                action = action + noise_std * self.max_action * torch.randn_like(action)
            action = torch.max(torch.min(action, self.max_action), -self.max_action)
        self.actor.train()
        return action

    def update(self, batch: dict) -> dict:
        s = batch['states']
        a = batch['actions']
        r = batch['rewards'].unsqueeze(-1)
        s2 = batch['next_states']
        d = batch['dones'].unsqueeze(-1)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.cfg.policy_noise * self.max_action).clamp(
                -self.cfg.noise_clip * self.max_action,
                self.cfg.noise_clip * self.max_action,
            )
            next_a = self.actor_target(s2) + noise
            next_a = torch.max(torch.min(next_a, self.max_action), -self.max_action)
            target_q1, target_q2 = self.critic_target(s2, next_a)
            target_q = torch.min(target_q1, target_q2)
            y = r + (1.0 - d) * self.cfg.gamma * target_q

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        td_error = (q1.detach() - y).abs().squeeze(-1)
        info = {
            'critic_loss': float(critic_loss.item()),
            'td_error': td_error,
        }

        self.total_updates += 1
        if self.total_updates % self.cfg.policy_freq == 0:
            actor_loss = -self.critic.q1_only(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            # info['actor_loss'] = float(actor_loss.item())

            self.last_actor_loss = float(actor_loss.item())  # 【新增】记录最新 loss

        info['actor_loss'] = self.last_actor_loss  # 【新增】保证每次字典里都有这行

        return info

    def _soft_update(self, net, target_net):
        tau = self.cfg.tau
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
