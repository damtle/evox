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
        warmup_gens: int = 50,
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
        self.warmup_gens = int(warmup_gens)

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

        # 【新增】：用于日志统计的计数器
        self.log_rl_success = 0  # RL 生成更好解的次数
        self.log_rl_total = 0  # RL 生成解的总次数
        self.log_rl_improve = 0.0  # RL 带来的适应度总提升量

        # 【核心修复 1】：Wrapper 独立维护全局最优，彻底与底层算法解耦
        self.global_best_fit = Mutable(torch.tensor(torch.inf, device=self.device))
        self.global_best_location = Mutable(torch.zeros(self.dim, device=self.device))


        self.ema_success_rate = 1.0  # 初始保持乐观，设为 100%
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
        if hasattr(self, 'evaluate'):
            self.base.evaluate = self.evaluate

        self.base.init_step()
        self.prev_pop = self.base.pop.clone()
        self.prev_fit = self.base.fit.clone()

        # 【核心修复 2】：初始化时，从种群中找出当前的最优值存入 Wrapper
        best_val, best_idx = torch.min(self.base.fit, dim=0)
        self.global_best_fit = best_val.clone()
        self.global_best_location = self.base.pop[best_idx].clone()

    def step(self):
        # 1) Let the base EA advance first.
        self.base.step()
        pop = self.base.pop
        fit = self.base.fit

        # 2) Build current state from real EA trace.
        prev_fit = torch.where(torch.isfinite(self.prev_fit), self.prev_fit, fit)
        states = self.state_builder.build(pop, self.prev_pop, fit, prev_fit)

        # 计算当前种群在各个维度上的标准差，作为 "收敛程度感知器"
        pop_std = torch.std(self.base.pop, dim=0)
        diversity = pop_std.mean().item()

        current_step = int(self.step_counter.item())
        is_warmup_over = current_step >= self.warmup_gens

        # =========================================================================
        # 🧠 高级智能门控机制 (RL Usefulness Estimator)
        # 条件 1: 多样性 > 1e-2 (EA 还在活跃探索期，地形复杂，需要 RL 帮忙)
        # 条件 2: EMA 成功率 > 1% (说明 RL 最近的指导非常有效，值得继续信任)
        # 只要满足其一，RL 就保持激活状态。否则冻结，防止 Extrapolation Error 导致崩盘。
        # =========================================================================
        is_rl_useful = (diversity > 1e-2) or (self.ema_success_rate > 0.01)

        n_rl = max(1, int(round(self.pop_size * self.rl_candidate_ratio)))
        traj_id = int(self.traj_counter.item())
        step_id = current_step

        num_success = 0
        improvement_sum = 0.0

        # ★ 只有预热结束 且 RL 被评估为“有用”时，才进行动作提出与评估 (极大节省 FEs)
        if is_warmup_over and is_rl_useful:
            rl_indices = torch.randperm(self.pop_size, device=self.device)[:n_rl]
            rl_states = states[rl_indices]

            # 获取 RL 网络输出的动作 (范围被限制在 [-action_scale, action_scale])
            raw_action = self.rl_agent.act(rl_states, noise_std=self.exploration_noise)

            # 【关键修复】：将动作还原为 [-1, 1] 的相对比例
            normalized_action = raw_action / self.action_scale

            # 计算当前种群在各个维度上的标准差，作为 "收敛程度感知器"
            pop_std = torch.std(self.base.pop, dim=0)

            # 加上一个极小的 epsilon 防止后期完全变成 0，导致停滞
            dynamic_step_size = pop_std + 1e-6

            # 实际执行的动作 = 相对方向 * 种群标准差 * 探索系数(比如让它在 1.5 倍标准差内探索)
            actual_scaled_action = normalized_action * dynamic_step_size * 0.3

            # 叠加到原位置上
            candidate_pop = pop[rl_indices] + actual_scaled_action
            candidate_pop = clamp(candidate_pop, self.lb, self.ub)
            candidate_fit = self.evaluate(candidate_pop)

            # 4) Store RL-generated transitions with true evaluations.
            next_states = self.state_builder.build(candidate_pop, pop[rl_indices], candidate_fit, fit[rl_indices])
            # 【核心修复 3】：使用 Wrapper 自己的 global_best_fit 计算奖励
            reward = relative_improvement_reward(fit[rl_indices], candidate_fit,
                                                 self.global_best_fit.expand_as(candidate_fit))
            done = torch.zeros_like(reward)
            # future-improvement placeholder; updated from immediate improvement for online usage
            future_improvement = torch.clamp(fit[rl_indices] - candidate_fit, min=0.0)

            traj_id = int(self.traj_counter.item())
            step_id = int(self.step_counter.item())
            for i in range(n_rl):
                t = Transition(
                    state=rl_states[i].detach().cpu(),
                    action=raw_action[i].detach().cpu(),
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

            # =========================================================================
            # 5) Selection: 1-to-1 Niche Replacement (通用生态位原位替换)
            # 核心思想: 无论是 PSO 还是 DE，RL 只在个体的当前位置提供“局部梯度下降”
            # 绝不跨越生态位替换其他粒子，从而最大程度保护 EA 的种群多样性和探索能力
            # =========================================================================
            better = candidate_fit < fit[rl_indices]

            num_success = better.sum().item()
            improvement_sum = 0.0

            if better.any():
                improvement_sum = (fit[rl_indices][better] - candidate_fit[better]).sum().item()

                # 获取获得了适应度提升的个体原始索引
                selected_idx = rl_indices[better]

                # 1. 物理位置与适应度的原位替换 (Universal)
                pop[selected_idx] = candidate_pop[better]
                fit[selected_idx] = candidate_fit[better]

                # 2. 动态状态重置 (向下兼容 PSO 等含有复杂物理动量的算法)
                # 如果是 SaDE，因为没有 velocity 属性，会自动安全跳过
                if hasattr(self.base, 'velocity'):
                    self.base.velocity[selected_idx] = torch.zeros_like(self.base.velocity[selected_idx])

                # 如果是 PSO，需要同步更新个体的历史最优记录
                if hasattr(self.base, 'local_best_fit'):
                    compare = self.base.local_best_fit[selected_idx] > candidate_fit[better]
                    lidx = selected_idx[compare]
                    if lidx.numel() > 0:
                        self.base.local_best_location[lidx] = candidate_pop[better][compare].clone()
                        self.base.local_best_fit[lidx] = candidate_fit[better][compare]

            # 记录日志统计
            self.log_rl_success += num_success
            self.log_rl_total += n_rl
            self.log_rl_improve += improvement_sum

            # 【新增】：平滑更新 EMA 成功率 (Alpha = 0.1)
            current_success_rate = num_success / max(1, n_rl)
            self.ema_success_rate = 0.9 * self.ema_success_rate + 0.1 * current_success_rate

        # 同步回底层 Algorithm
        self.base.pop = pop
        self.base.fit = fit

        # =========================================================================
        # 3. 更新全局最优 (Global Best) - 同样采用向下兼容设计
        # =========================================================================
        current_best_val, current_best_idx = torch.min(self.base.fit, dim=0)
        if current_best_val < self.global_best_fit:
            self.global_best_fit = current_best_val.clone()
            self.global_best_location = self.base.pop[current_best_idx].clone()

        # 将 Wrapper 维护的全局最优，反向同步给需要的底层 EA
        if hasattr(self.base, 'global_best_fit'):  # 兼容 PSO
            self.base.global_best_fit = self.global_best_fit.clone()
            self.base.global_best_location = self.global_best_location.clone()
        if hasattr(self.base, 'best_index'):  # 兼容 SaDE
            self.base.best_index = current_best_idx

        # =========================================================================
        # 6) Store EA transitions (保持不变)
        # =========================================================================
        ea_reward = torch.clamp(self.prev_fit - fit, min=-1.0, max=1.0)
        ea_future = torch.clamp(self.prev_fit - fit, min=0.0)
        ea_next_states = states
        prev_states = self.state_builder.build(self.prev_pop, self.prev_pop, prev_fit, prev_fit)

        pop_std = torch.std(self.base.pop, dim=0)
        dynamic_step_size = pop_std + 1e-6

        for i in range(min(self.pop_size, n_rl)):
            idx = int(i)
            dx = pop[idx] - self.prev_pop[idx]
            ea_raw_action = (dx / (dynamic_step_size * 1.5)) * self.action_scale
            ea_raw_action = torch.clamp(ea_raw_action, -self.action_scale, self.action_scale)

            t = Transition(
                state=prev_states[idx].detach().cpu(),
                action=ea_raw_action.detach().cpu(),
                reward=ea_reward[idx].detach().cpu(),
                next_state=ea_next_states[idx].detach().cpu(),
                done=torch.tensor(0.0, device='cpu'),
                future_improvement=ea_future[idx].detach().cpu(),
                td_priority=float(abs(ea_reward[idx].item()) + 1e-6),
                source=0,
                traj_id=traj_id,
                step_id=step_id,
            )
            self.replay.add(t)

        # =========================================================================
        # 7) Train RL from replay
        # ★ 如果 RL 失去作用（成功率低下且种群停滞），立刻停止训练，阻断 Extrapolation Error！
        # =========================================================================
        info = {}
        if len(self.replay) >= self.train_after and is_warmup_over and is_rl_useful:
            for _ in range(self.train_steps_per_gen):
                batch = self.replay.sample(self.batch_size, rl_ratio=0.5)
                info = self.rl_agent.update(batch)
                self.replay.update_td_priorities(batch['indices'].tolist(), info['td_error'])
        # =========================================================================
        # 8) Logging with Population Diversity Monitor
        # =========================================================================
        current_step = int(self.step_counter.item())
        if current_step > 0 and current_step % 50 == 0:
            success_rate = self.log_rl_success / max(1, self.log_rl_total)
            avg_imp = self.log_rl_improve / max(1, self.log_rl_success)

            # 【重要指标】：计算种群平均标准差，监控多样性
            diversity = pop_std.mean().item()

            print(
                f"   [RL Log Gen {current_step}] 成功率: {success_rate:.2%} ({self.log_rl_success}/{self.log_rl_total}) | 平均提升: {avg_imp:.2e} | 种群多样性: {diversity:.4e}")
            if 'critic_loss' in info:
                print(
                    f"                  Critic Loss: {info.get('critic_loss', 0):.4f} | Actor Loss: {info.get('actor_loss', 0):.4f}")

            self.log_rl_success = 0
            self.log_rl_total = 0
            self.log_rl_improve = 0.0

        self.prev_pop = self.base.pop.clone()
        self.prev_fit = self.base.fit.clone()
        self.step_counter = self.step_counter + 1
