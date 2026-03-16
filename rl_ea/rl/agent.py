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
        sources = batch['sources']

        # 区分数据来源
        rl_mask = (sources == 1)
        ea_mask = (sources == 0)

        td_error = torch.zeros(s.shape[0], device=self.device)
        critic_loss_val = 0.0

        # =====================================================================
        # 1. Critic Update (只用 RL 数据，彻底斩断外推误差！)
        # =====================================================================
        if rl_mask.any():
            s_rl, a_rl, r_rl, s2_rl, d_rl = s[rl_mask], a[rl_mask], r[rl_mask], s2[rl_mask], d[rl_mask]

            with torch.no_grad():
                noise = (torch.randn_like(a_rl) * self.cfg.policy_noise * self.max_action).clamp(
                    -self.cfg.noise_clip * self.max_action,
                    self.cfg.noise_clip * self.max_action,
                )
                next_a = self.actor_target(s2_rl) + noise
                next_a = torch.max(torch.min(next_a, self.max_action), -self.max_action)
                target_q1, target_q2 = self.critic_target(s2_rl, next_a)
                target_q = torch.min(target_q1, target_q2)
                y = r_rl + (1.0 - d_rl) * self.cfg.gamma * target_q
                y = torch.clamp(y, -10.0, 10.0)

            q1, q2 = self.critic(s_rl, a_rl)
            critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_opt.step()

            td_error[rl_mask] = (q1.detach() - y).abs().squeeze(-1)
            critic_loss_val = float(critic_loss.item())

            # 记录 Q 的绝对均值，用于 Actor BC Loss 的自适应权重
            q_abs_mean = q1.detach().abs().mean().item()
        else:
            q_abs_mean = 1.0

        info = {
            'critic_loss': critic_loss_val,
            'td_error': td_error,
        }

        # =====================================================================
        # 2. Actor Update (混合数据：RL 数据优化 Q，EA 数据做 BC 正则化)
        # =====================================================================
        self.total_updates += 1
        if self.total_updates % self.cfg.policy_freq == 0:
            actor_loss = torch.tensor(0.0, device=self.device)

            # 2.1 对于 RL 数据：最大化 Q 值 (寻找超越 EA 的方向)
            if rl_mask.any():
                actor_loss_rl = -self.critic.q1_only(s[rl_mask], self.actor(s[rl_mask])).mean()
                actor_loss = actor_loss + actor_loss_rl

            # 2.2 对于 EA 数据：Behavior Cloning
            if ea_mask.any():
                pi_ea = self.actor(s[ea_mask])
                bc_loss = F.mse_loss(pi_ea, a[ea_mask])

                # 【接住补丁】：获取来自 Wrapper 的时间退火因子 (默认 1.0)
                bc_decay = batch.get('bc_decay', torch.tensor(1.0, device=self.device)).item()

                # TD3+BC 自适应退火：双重保障 (Q值自信度退火 + 时间硬性退火)
                alpha = 2.5
                lmbda = (alpha / (q_abs_mean + 1e-5)) * bc_decay

                actor_loss = actor_loss + lmbda * bc_loss

            if actor_loss.requires_grad:
                self.actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_opt.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            self.last_actor_loss = float(actor_loss.item())

        info['actor_loss'] = self.last_actor_loss
        return info

    def _soft_update(self, net, target_net):
        tau = self.cfg.tau
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
