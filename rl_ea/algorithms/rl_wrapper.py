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
        # 【新增】：追踪每个粒子的停滞代数
        self.stagnation = Mutable(torch.zeros(self.pop_size, device=self.device))
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
        self.base.step()
        pop = self.base.pop
        fit = self.base.fit

        # 【新增】：更新个体的停滞代数
        improved = fit < self.prev_fit
        self.stagnation = torch.where(improved, torch.zeros_like(self.stagnation), self.stagnation + 1)

        pop_mean = torch.mean(pop, dim=0)
        pop_std = torch.std(pop, dim=0)
        diversity = pop_std.mean().item()

        # Build state (传入新增的 pop_mean, pop_std, stagnation)
        prev_fit_safe = torch.where(torch.isfinite(self.prev_fit), self.prev_fit, fit)
        states = self.state_builder.build(pop, self.prev_pop, fit, prev_fit_safe, pop_mean, pop_std, self.stagnation)

        current_step = int(self.step_counter.item())
        is_warmup_over = current_step >= self.warmup_gens
        # =========================================================================
        # 【修复问题 7】：极其严谨的分段智能门控逻辑
        # =========================================================================
        div_threshold = 1e-2
        sr_threshold_low = 0.005  # 0.5%
        sr_threshold_high = 0.05  # 5%

        # 免死金牌：如果 Replay Buffer 还没攒够数据开始训练，无条件给 RL 放行！
        is_training_started = len(self.replay) >= self.train_after

        # 强制试探：每 50 代强行拉升一次信心，防止彻底死锁
        if current_step > 0 and current_step % 50 == 0:
            self.ema_success_rate = max(self.ema_success_rate, 0.5)

        is_rl_useful = (not is_training_started) or (
                (diversity > div_threshold and self.ema_success_rate > sr_threshold_low)
                or (self.ema_success_rate > sr_threshold_high)
        )

        n_rl = max(1, int(round(self.pop_size * self.rl_candidate_ratio)))

        traj_id = int(self.traj_counter.item())
        step_id = current_step

        improvement_sum = 0.0

        if is_warmup_over and is_rl_useful:
            # =========================================================================
            # 【核心修改 A】：优先抽样！不再随机选，长期停滞的粒子被选中的概率更高
            # =========================================================================
            sample_probs = (self.stagnation + 1.0)
            sample_probs = sample_probs / sample_probs.sum()
            rl_indices = torch.multinomial(sample_probs, n_rl, replacement=False)
            rl_states = states[rl_indices]

            raw_action = self.rl_agent.act(rl_states, noise_std=self.exploration_noise)

            # =========================================================================
            # 【核心修改 B】：解析复合动作空间 (Direction + Step Scale + Explore Gate)
            # =========================================================================
            # 1. 探索方向 [-1, 1]
            direction = raw_action[:, :self.dim]

            # 【修复】：强制将方向向量归一化为单位向量 (长度为1)，防止网络通过缩小方向来偷懒
            direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-8)

            # 2. 步长缩放 [0, 1] (将 -1~1 映射到 0~1)
            step_scale = (raw_action[:, self.dim:self.dim + 1] + 1.0) / 2.0
            # 3. 探索门控 [0, 1] (0: 纯局部修调 pop_std, 1: 纯全局大跳跃 ub-lb)
            explore_gate = (raw_action[:, self.dim + 1:self.dim + 2] + 1.0) / 2.0

            # 强制试探：5% 概率做极限大跳跃，5% 概率做种群内部大跨越
            rand_val = torch.rand(1).item()
            if rand_val < 0.05:
                explore_gate = torch.ones_like(explore_gate)
            elif rand_val < 0.10:
                explore_gate = torch.zeros_like(explore_gate)

            # =========================================================================
            # 【稳健 Patch 1】：温和放大物理尺度
            # 局部：放大到 1.25 倍 pop_std，给一点探索空间但不破坏 Critic 的认知
            # 全局：设为搜索空间的 10%，既能跳出当前 basin，又不算瞎跳
            # =========================================================================
            macro_scale = (self.ub - self.lb) * 0.1
            effective_scale = (1.0 - explore_gate) * (pop_std * 1.25) + explore_gate * macro_scale

            dx = direction * effective_scale * step_scale

            candidate_pop = pop[rl_indices] + dx
            candidate_pop = clamp(candidate_pop, self.lb, self.ub)
            candidate_fit = self.evaluate(candidate_pop)

            # =========================================================================
            # 【核心修改 C & 修复问题 8】：组合奖励 (Fitness + Novelty + Escape Bonus)
            # =========================================================================
            improved_rl = candidate_fit < fit[rl_indices]
            next_stag = torch.where(improved_rl, torch.zeros_like(self.stagnation[rl_indices]),
                                    self.stagnation[rl_indices] + 1)
            next_states = self.state_builder.build(candidate_pop, pop[rl_indices], candidate_fit, fit[rl_indices],
                                                   pop_mean, pop_std, next_stag)

            from rl_ea.utils.optimization import compute_novelty_reward
            fit_reward = relative_improvement_reward(fit[rl_indices], candidate_fit,
                                                     self.global_best_fit.expand_as(candidate_fit))
            novelty_reward = compute_novelty_reward(candidate_pop, pop)

            # 【新增：Escape Bonus (逃逸奖励)】
            # 定义：如果一个粒子卡住超过了 10 代 (T=10)，并且被 RL 一脚踢出坑获得了提升，给予 0.5 的巨大奖励！
            is_stagnant = self.stagnation[rl_indices] >= 10.0
            escape_bonus = 0.5 * (is_stagnant.float() * improved_rl.float())

            # 最终的战略级奖励：Fitness (主目标) + Novelty (鼓励走新路) + Escape (鼓励拯救烂摊子)
            reward = fit_reward + 0.08 * novelty_reward + escape_bonus

            done = torch.zeros_like(reward)
            future_improvement = torch.clamp(fit[rl_indices] - candidate_fit, min=0.0)

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
            # 【稳健 Patch 3】：坚决拒绝毫无意义的微调，逼迫 RL 找真正有价值的解
            rel_improvement = (fit[rl_indices] - candidate_fit) / (torch.abs(fit[rl_indices]) + 1e-8)

            # 条件1：创造了新的全局最优 (无条件接受)
            is_new_global = candidate_fit < self.global_best_fit
            # 条件2：获得了实质性的局部提升 (大于万分之五 5e-4，门槛温和但足以拦住刷分)
            is_significant = rel_improvement > 5e-4

            # 必须严格更好，且满足实质性提升或全局最优才允许替换！
            better = (candidate_fit < fit[rl_indices]) & (is_new_global | is_significant)

            num_success = better.sum().item()

            if better.any():
                improvement_sum = (fit[rl_indices][better] - candidate_fit[better]).sum().item()

                # 获取获得了适应度提升的个体原始索引
                selected_idx = rl_indices[better]

                # 1. 物理位置与适应度的原位替换 (Universal)
                pop[selected_idx] = candidate_pop[better]
                fit[selected_idx] = candidate_fit[better]

                # 既然获得了提升，重置停滞状态
                self.stagnation[selected_idx] = 0.0

                if hasattr(self.base, 'velocity'):
                    self.base.velocity[selected_idx] = torch.zeros_like(self.base.velocity[selected_idx])
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
            self.ema_success_rate = 0.95 * self.ema_success_rate + 0.05 * current_success_rate

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

            self.ema_success_rate = 1.0

        # 将 Wrapper 维护的全局最优，反向同步给需要的底层 EA
        if hasattr(self.base, 'global_best_fit'):  # 兼容 PSO
            self.base.global_best_fit = self.global_best_fit.clone()
            self.base.global_best_location = self.global_best_location.clone()
        if hasattr(self.base, 'best_index'):  # 兼容 SaDE
            self.base.best_index = current_best_idx

        # # =========================================================================
        # # 【核心修改 D】：精准逆向映射 EA 经验到新的复合动作空间
        # # =========================================================================
        # ea_reward = torch.clamp(self.prev_fit - fit, min=-1.0, max=1.0)
        # ea_future = torch.clamp(self.prev_fit - fit, min=0.0)
        # ea_next_states = states
        # 构建一个安全的 prev_fit
        # prev_fit_safe_for_prev = torch.where(torch.isfinite(self.prev_fit), self.prev_fit, fit)
        # prev_states = self.state_builder.build(self.prev_pop, self.prev_pop, prev_fit_safe_for_prev,
        #                                        prev_fit_safe_for_prev, pop_mean, pop_std, self.stagnation)
        # for i in range(min(self.pop_size, n_rl)):
        #     idx = int(i)
        #     dx_ea = pop[idx] - self.prev_pop[idx]
        #
        #     # 将 EA 的自然步长逆推为网络输出格式：
        #     # 假设 explore_gate = 0.0 -> 映射到网络输出为 -1.0
        #     # 那么 effective_scale = pop_std
        #     rel_dx = dx_ea / (pop_std + 1e-6)
        #
        #     # 提取最大相对尺度作为 step_scale_ea (范围 0~1)
        #     step_scale_ea = torch.clamp(torch.max(torch.abs(rel_dx)), max=1.0)
        #
        #     if step_scale_ea > 1e-6:
        #         direction_ea = torch.clamp(rel_dx / step_scale_ea, -1.0, 1.0)
        #     else:
        #         direction_ea = torch.zeros_like(dx_ea)
        #
        #     # 将 step_scale_ea (0~1) 映射回网络的原始范围 (-1~1)
        #     raw_step_scale = step_scale_ea * 2.0 - 1.0
        #     raw_explore_gate = torch.tensor([-1.0], device=self.device)
        #
        #     ea_raw_action = torch.cat([
        #         direction_ea,
        #         raw_step_scale.unsqueeze(0),
        #         raw_explore_gate
        #     ])
        #
        #     t = Transition(
        #         state=prev_states[idx].detach().cpu(),
        #         action=ea_raw_action.detach().cpu(),
        #         reward=ea_reward[idx].detach().cpu(),
        #         next_state=ea_next_states[idx].detach().cpu(),
        #         done=torch.tensor(0.0, device='cpu'),
        #         future_improvement=ea_future[idx].detach().cpu(),
        #         td_priority=float(abs(ea_reward[idx].item()) + 1e-6),
        #         source=0,
        #         traj_id=traj_id,
        #         step_id=step_id,
        #     )
        #     self.replay.add(t)

        # =========================================================================
        # 7) Train RL from replay
        # ★ 如果 RL 失去作用（成功率低下且种群停滞），立刻停止训练，阻断 Extrapolation Error！
        # =========================================================================
        info = {}
        if len(self.replay) >= self.train_after and is_warmup_over and is_rl_useful:
            for _ in range(self.train_steps_per_gen):
                # 【修复】：去掉 rl_ratio 参数，只使用 RL 经验
                batch = self.replay.sample(self.batch_size)
                info = self.rl_agent.update(batch)
                self.replay.update_td_priorities(batch['indices'].tolist(), info['td_error'])

        # =========================================================================
        # 8) Logging with Population Diversity Monitor
        # =========================================================================
        current_step = int(self.step_counter.item())
        if current_step > 0 and current_step % 50 == 0:
            success_rate = self.log_rl_success / max(1, self.log_rl_total)
            avg_imp = self.log_rl_improve / max(1, self.log_rl_success)
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

        self.traj_counter = self.traj_counter + 1