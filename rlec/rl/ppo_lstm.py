import torch
import torch.nn as nn
import numpy as np
from .networks import ActorMLP, CriticMLP
from .rollout_buffer import RolloutBuffer


class PPO:
    def __init__(self, state_dim: int, action_dim: int, device: torch.device,
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, clip_ratio=0.2,
                 entropy_coef=0.01, k_epochs=4):
        self.device = device
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.k_epochs = k_epochs

        self.actor = ActorMLP(state_dim, action_dim).to(device)
        self.critic = CriticMLP(state_dim).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])
        self.buffer = RolloutBuffer(device)

    def select_action(self, state: torch.Tensor, deterministic=False):
        with torch.no_grad():
            dist = self.actor(state)
            value = self.critic(state)
            action = dist.mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            # 【建议修 4】：限制高斯采样的极端值，防止 sigmoid 梯度消失
            raw_action = torch.clamp(action, -4.0, 4.0)
            intent_action = torch.sigmoid(raw_action)

        return (action.cpu().numpy(), intent_action.cpu().numpy(),
                log_prob.item(), value.item())

    def update(self):
        if len(self.buffer.states) == 0: return {}

        states = torch.stack(self.buffer.states).to(self.device)
        actions = torch.stack(self.buffer.actions).to(self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, device=self.device)

        with torch.no_grad():
            next_value = self.critic(states[-1:]).item()

        returns, advantages = self.buffer.compute_gae(next_value, self.gamma)
        actor_loss_list, critic_loss_list = [], []

        for _ in range(self.k_epochs):
            dist = self.actor(states)
            values = self.critic(states).squeeze(-1)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages

            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            critic_loss = nn.MSELoss()(values, returns)

            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer.step()

            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())

        self.buffer.clear()
        return {"actor_loss": np.mean(actor_loss_list), "critic_loss": np.mean(critic_loss_list)}