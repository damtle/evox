import torch

class RolloutBuffer:
    """PPO 专用的 On-policy 轨迹缓冲池"""
    def __init__(self, device: torch.device):
        self.device = device
        self.clear()

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.log_probs, self.dones = [], [], []

    def add(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_gae(self, next_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.values + [next_value], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)

        advantages = torch.zeros(len(rewards), dtype=torch.float32, device=self.device)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * mask * last_gae

        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages