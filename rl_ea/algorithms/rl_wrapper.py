from __future__ import annotations

from typing import Optional
import torch
from evox.core import Algorithm, Mutable
from evox.utils import clamp

from rl_ea.features.state_builder import StateBuilder
from rl_ea.replay.buffer import PrioritizedReplayBuffer, Transition
from rl_ea.utils.optimization import relative_improvement_reward, compute_future_improvement


class RLEnhancedAlgorithm(Algorithm):
    """Generic wrapper that augments any EvoX algorithm with RL-generated candidates.

    Requirements on base algorithm:
    - has attributes: pop, fit, lb, ub
    - supports init_step() and step()
    - self.evaluate(...) available through EvoX workflow
    """

    def __init__(
        self,
        base_algo: Algorithm,
        rl_agent,
        replay_buffer: PrioritizedReplayBuffer,
        rl_candidate_ratio: float = 0.3,
        train_after: int = 512,
        train_steps_per_gen: int = 1,
        batch_size: int = 256,
        action_scale: Optional[torch.Tensor] = None,
        exploration_noise: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.base = base_algo
        self.rl_agent = rl_agent
        self.replay = replay_buffer
        self.state_builder = StateBuilder()
        self.rl_candidate_ratio = float(rl_candidate_ratio)
        self.train_after = int(train_after)
        self.train_steps_per_gen = int(train_steps_per_gen)
        self.batch_size = int(batch_size)
        self.exploration_noise = float(exploration_noise)
        self.device = device if device is not None else base_algo.fit.device

        pop_shape = base_algo.pop.shape
        self.pop_size, self.dim = pop_shape[0], pop_shape[1]
        if action_scale is None:
            action_scale = (base_algo.ub - base_algo.lb).squeeze(0) * 0.1
        self.action_scale = action_scale.to(self.device)

        # history required to form state features
        self.prev_pop = Mutable(base_algo.pop.clone())
        self.prev_fit = Mutable(torch.full_like(base_algo.fit, torch.inf))
        self.traj_counter = Mutable(torch.tensor(0, device=self.device))
        self.step_counter = Mutable(torch.tensor(0, device=self.device))

    # expose population and fitness for workflow compatibility
    @property
    def pop(self):
        return self.base.pop

    @pop.setter
    def pop(self, value):
        self.base.pop = value

    @property
    def fit(self):
        return self.base.fit

    @fit.setter
    def fit(self, value):
        self.base.fit = value

    @property
    def lb(self):
        return self.base.lb

    @property
    def ub(self):
        return self.base.ub

    def init_step(self):
        # 【关键修复】：将工作流注入到 Wrapper 层的 evaluate 函数，同步传递给底层算法
        if hasattr(self, 'evaluate'):
            self.base.evaluate = self.evaluate

        # 然后再正常调用底层的 init_step
        self.base.init_step()

        self.prev_pop = self.base.pop.clone()
        self.prev_fit = self.base.fit.clone()

    def step(self):
        # 1) Let the base EA advance first.
        self.base.step()
        pop = self.base.pop
        fit = self.base.fit

        # 2) Build current state from real EA trace.
        prev_fit = torch.where(torch.isfinite(self.prev_fit), self.prev_fit, fit)
        states = self.state_builder.build(pop, self.prev_pop, fit, prev_fit)

        # 3) RL proposes additional candidate steps for a subset of population.
        n_rl = max(1, int(round(self.pop_size * self.rl_candidate_ratio)))
        rl_indices = torch.randperm(self.pop_size, device=self.device)[:n_rl]
        rl_states = states[rl_indices]
        raw_action = self.rl_agent.act(rl_states, noise_std=self.exploration_noise)
        scaled_action = raw_action * self.action_scale

        candidate_pop = pop[rl_indices] + scaled_action
        candidate_pop = clamp(candidate_pop, self.lb, self.ub)
        candidate_fit = self.evaluate(candidate_pop)

        # 4) Store RL-generated transitions with true evaluations.
        next_states = self.state_builder.build(candidate_pop, pop[rl_indices], candidate_fit, fit[rl_indices])
        reward = relative_improvement_reward(fit[rl_indices], candidate_fit, self.base.global_best_fit.expand_as(candidate_fit))
        done = torch.zeros_like(reward)
        # future-improvement placeholder; updated from immediate improvement for online usage
        future_improvement = torch.clamp(fit[rl_indices] - candidate_fit, min=0.0)

        traj_id = int(self.traj_counter.item())
        step_id = int(self.step_counter.item())
        for i in range(n_rl):
            t = Transition(
                state=rl_states[i].detach().cpu(),
                action=scaled_action[i].detach().cpu(),
                reward=reward[i].detach().cpu(),
                next_state=next_states[i].detach().cpu(),
                done=done[i].detach().cpu(),
                future_improvement=future_improvement[i].detach().cpu(),
                td_priority=float(abs(reward[i].item()) + 1e-6),
                source=1,
                traj_id=traj_id,
                step_id=step_id,
            )
            self.replay.add(t)

        # 5) Selection: replace EA individuals if RL candidate is better.
        better = candidate_fit < fit[rl_indices]
        if better.any():
            selected_idx = rl_indices[better]
            pop[selected_idx] = candidate_pop[better]
            fit[selected_idx] = candidate_fit[better]
            self.base.pop = pop
            self.base.fit = fit
            # update best records if base algorithm keeps them as Mutable state
            if hasattr(self.base, 'local_best_fit'):
                compare = self.base.local_best_fit[selected_idx] > candidate_fit[better]
                lidx = selected_idx[compare]
                if lidx.numel() > 0:
                    self.base.local_best_location[lidx] = candidate_pop[better][compare]
                    self.base.local_best_fit[lidx] = candidate_fit[better][compare]
            if hasattr(self.base, 'global_best_fit'):
                current_best_val, current_best_idx = torch.min(self.base.fit, dim=0)
                if current_best_val < self.base.global_best_fit:
                    self.base.global_best_fit = current_best_val
                    self.base.global_best_location = self.base.pop[current_best_idx]

        # 6) Also store a small amount of EA transitions so replay covers EA behavior.
        ea_reward = torch.clamp(self.prev_fit - fit, min=-1.0, max=1.0)
        ea_future = torch.clamp(self.prev_fit - fit, min=0.0)
        ea_next_states = states
        prev_states = self.state_builder.build(self.prev_pop, self.prev_pop, prev_fit, prev_fit)
        for i in range(min(self.pop_size, n_rl)):
            idx = int(i)
            t = Transition(
                state=prev_states[idx].detach().cpu(),
                action=(pop[idx] - self.prev_pop[idx]).detach().cpu(),
                reward=ea_reward[idx].detach().cpu(),
                next_state=ea_next_states[idx].detach().cpu(),
                done=torch.tensor(0.0),
                future_improvement=ea_future[idx].detach().cpu(),
                td_priority=float(abs(ea_reward[idx].item()) + 1e-6),
                source=0,
                traj_id=traj_id,
                step_id=step_id,
            )
            self.replay.add(t)

        # 7) Train RL from replay.
        if len(self.replay) >= self.train_after:
            for _ in range(self.train_steps_per_gen):
                batch = self.replay.sample(self.batch_size, rl_ratio=0.5)
                info = self.rl_agent.update(batch)
                self.replay.update_td_priorities(batch['indices'].tolist(), info['td_error'])

        self.prev_pop = self.base.pop.clone()
        self.prev_fit = self.base.fit.clone()
        self.step_counter = self.step_counter + 1
