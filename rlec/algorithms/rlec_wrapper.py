from typing import Optional
import torch
import numpy as np

from evox.core import Algorithm, Mutable
from evox.utils import clamp

from rlec.features.macro_state import MacroStateBuilder
from rlec.features.stage_reward import StageRewardCalculator
from rlec.control.intent_vector import IntentVector
from rlec.control.interpreter import ControlInterpreter
from rlec.rl.ppo_lstm import PPO
from rlec.utils.population_metrics import compute_diversity


class RLECWrapper(Algorithm):
    """
    RLEC 强化学习进化阶段控制器
    将底层 EA 视为战术执行层，RL 视为战略调度层。
    """

    def __init__(
            self,
            base_algo: Algorithm,
            stage_length: int = 20,  # K 代为一个决策周期
            update_stages: int = 5,  # 收集多少个阶段的数据更新一次 PPO
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.base = base_algo
        self.K = stage_length
        self.update_stages = update_stages
        self.device = device if device is not None else base_algo.fit.device

        pop_shape = base_algo.pop.shape
        self.pop_size, self.dim = pop_shape[0], pop_shape[1]

        # 初始化三大件：雷达、评分、翻译官
        self.state_builder = MacroStateBuilder(self.dim, self.pop_size)
        self.reward_calc = StageRewardCalculator()
        self.interpreter = ControlInterpreter(self.pop_size)

        # 初始化 PPO 大脑 (状态维度 12，动作维度 6)
        self.rl_agent = PPO(
            state_dim=self.state_builder.state_dim,
            action_dim=6,
            device=self.device
        )
        self.initial_div = Mutable(torch.tensor(0.0, device=self.device))
        self.current_cmd_alpha_margin = Mutable(torch.tensor(0.0, device=self.device))
        self.current_cmd_beta_margin = Mutable(torch.tensor(0.0, device=self.device))

        self.current_cmd_scales = Mutable(torch.ones(self.pop_size, device=self.device))

        # ================= 时序追踪状态 =================
        self.step_counter = Mutable(torch.tensor(0, device=self.device))
        self.stagnation = Mutable(torch.zeros(self.pop_size, device=self.device))

        # 记录上一代的种群 (用于算提升和更新 stagnation)
        self.prev_pop = Mutable(base_algo.pop.clone())
        self.prev_fit = Mutable(torch.full_like(base_algo.fit, torch.inf))

        # 记录“阶段初”的种群，用于在 K 代结束后算总账
        self.stage_start_pop = Mutable(base_algo.pop.clone())
        self.stage_start_fit = Mutable(torch.full_like(base_algo.fit, torch.inf))
        self.stage_start_div = Mutable(torch.tensor(0.0, device=self.device))
        self.stage_start_stagnation = Mutable(torch.zeros(self.pop_size, device=self.device))

        # ================= RL 记忆追踪 =================
        self.hidden_a = Mutable(torch.zeros(1, 1, 64, device=self.device))
        self.hidden_c = Mutable(torch.zeros(1, 1, 64, device=self.device))
        self.last_state = Mutable(torch.zeros(1, self.state_builder.state_dim, device=self.device))
        self.last_action_raw = Mutable(torch.zeros(6, device=self.device))
        self.last_intent = Mutable(torch.zeros(6, device=self.device))
        self.last_logprob = Mutable(torch.tensor(0.0, device=self.device))
        self.last_value = Mutable(torch.tensor(0.0, device=self.device))
        self.last_roi = Mutable(torch.tensor(0.0, device=self.device))

        # ================= 战术指令缓存 =================
        # 用于存放当前 K 代中，底层 EA 必须服从的指令
        self.current_cmd_protected = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_cmd_rescue = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_cmd_target_div = Mutable(torch.tensor(0.0, device=self.device))

        # 包装器自己维护的最优记录
        self.global_best_fit = Mutable(torch.tensor(torch.inf, device=self.device))
        self.global_best_location = Mutable(torch.zeros(self.dim, device=self.device))

    # [此处暴露 pop, fit 等属性，保持与 EVOX 的兼容性]
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
        self.stage_start_pop = self.base.pop.clone()
        self.stage_start_fit = self.base.fit.clone()
        self.stage_start_div = compute_diversity(self.base.pop)

        best_val, best_idx = torch.min(self.base.fit, dim=0)
        self.global_best_fit = best_val.clone()
        self.global_best_location = self.base.pop[best_idx].clone()

        # 初始化时触发第 0 次调度
        self.initial_div = compute_diversity(self.base.pop)
        self._dispatch_new_stage()

    def _dispatch_new_stage(self):
        """司令部下达新阶段指令"""
        current_pop = self.base.pop
        current_fit = self.base.fit
        current_div = compute_diversity(current_pop)

        # 1. 提取 12 维宏观态势雷达图
        state = self.state_builder.build(
            pop_t=current_pop, fit_t=current_fit,
            pop_t_minus_k=self.stage_start_pop, fit_t_minus_k=self.stage_start_fit,
            stagnation=self.stagnation,
            last_action=self.last_intent, last_roi=self.last_roi.item()
        )

        # 2. PPO-LSTM 大脑思考，生成新指令
        # state: [1, 12]
        action_raw, intent_np, log_prob, value = self.rl_agent.select_action(
            state, deterministic=False
        )

        self.last_state = state.clone()

        # 3. 翻译官翻译指令
        intent_tensor = torch.tensor(intent_np, device=self.device, dtype=torch.float32)
        intent_obj = IntentVector.from_tensor(intent_tensor)
        # 传入 initial_div
        cmd = self.interpreter.interpret(intent_obj, current_fit, self.stagnation, self.initial_div.item())

        self.last_action_raw = torch.tensor(action_raw, device=self.device)
        self.last_intent = intent_tensor.clone()
        self.last_logprob = torch.tensor(log_prob, device=self.device)
        self.last_value = torch.tensor(value, device=self.device)

        self.current_cmd_protected = cmd.protected_mask
        self.current_cmd_rescue = cmd.rescue_mask
        self.current_cmd_scales = cmd.step_scales.clone()
        # 记录双容忍度
        self.current_cmd_alpha_margin = torch.tensor(cmd.alpha_margin, device=self.device)
        self.current_cmd_beta_margin = torch.tensor(cmd.beta_margin, device=self.device)
        self.current_cmd_target_div = torch.tensor(cmd.target_diversity, device=self.device)

        # # 5. 【硬核战术动作：破局救援】
        # # 在阶段初，对被标记为 rescue 的粒子施加无视物理规律的随机传送！
        # rescue_idx = torch.where(cmd.rescue_mask)[0]
        # if len(rescue_idx) > 0:
        #     new_pos = self.lb + torch.rand(len(rescue_idx), self.dim, device=self.device) * (self.ub - self.lb)
        #     self.base.pop[rescue_idx] = new_pos
        #     self.base.fit[rescue_idx] = self.evaluate(new_pos)
        #     # 重新计算救援后的真实多样性
        #     current_div = compute_diversity(self.base.pop)

        # 6. 更新新阶段的起点锚点 (此时种群已经是救援后的干净状态了)
        self.stage_start_pop = self.base.pop.clone()
        self.stage_start_fit = self.base.fit.clone()
        self.stage_start_div = current_div
        self.stage_start_stagnation = self.stagnation.clone()

    def step(self):
        step_val = int(self.step_counter.item())

        # =====================================================================
        # 第一阶段：清算上一阶段奖励，并下达新阶段指令 (每 K 代触发一次)
        # =====================================================================
        if step_val > 0 and step_val % self.K == 0:
            current_div = compute_diversity(self.base.pop)

            # 1. 结算这个周期的总账 (K代奖励)
            stag_ratio = (self.stage_start_stagnation >= 10).sum().float() / self.pop_size
            reward, roi = self.reward_calc.calculate(
                fit_start=self.stage_start_fit, fit_end=self.base.fit,
                div_start=self.stage_start_div.item(), div_end=current_div.item(),
                target_div=self.current_cmd_target_div.item(),
                initial_div=self.initial_div.item(),
                stagnation_start=self.stage_start_stagnation, stag_ratio=stag_ratio.item(),
                action=self.last_intent
            )
            self.last_roi = torch.tensor(roi, device=self.device)

            self.rl_agent.buffer.add(
                state=self.last_state, action=self.last_action_raw,
                reward=reward, value=self.last_value.item(),
                log_prob=self.last_logprob.item(), done=0.0
            )

            # 3. 如果攒够了阶段数，通知 PPO 进行反向传播更新大脑
            if len(self.rl_agent.buffer.states) >= self.update_stages:
                self.rl_agent.update()

            # 4. 发布新一阶段的控制意图
            self._dispatch_new_stage()

        # =====================================================================
        # 第二阶段：底层战术执行 (EA 按自己规则跑一步)
        # =====================================================================
        prev_fit_gen = self.base.fit.clone()
        prev_pop_gen = self.base.pop.clone()

        # 让底层的 CoDE 自己去跑一轮，生成候选解
        self.base.step()

        curr_pop = self.base.pop
        curr_fit = self.base.fit

        # 【必须改 1 应用】：计算出 EA 原本打算走的位移 dx
        dx = curr_pop - prev_pop_gen

        # 对受干预的个体施加缩放 (未受干预个体的 scale 为 1.0)
        dx = dx * self.current_cmd_scales.unsqueeze(1)

        # =====================================================================
        # 【必须改 2 应用】：分级脱困逃逸 (定向排斥力)
        # =====================================================================
        rescue_mask = self.current_cmd_rescue
        if rescue_mask.any():
            # 找到种群中心
            center = prev_pop_gen.mean(dim=0)
            # 算出远离中心的方向向量
            dir_escape = prev_pop_gen[rescue_mask] - center
            dir_escape = dir_escape / (torch.norm(dir_escape, dim=1, keepdim=True) + 1e-8)

            # 【建议修 3】：加入 30% 的纯随机扰动方向，防止集体朝同一个方向崩出去
            rand_dir = torch.randn_like(dir_escape)
            rand_dir = rand_dir / (torch.norm(rand_dir, dim=1, keepdim=True) + 1e-8)
            escape_dir = 0.7 * dir_escape + 0.3 * rand_dir
            escape_dir = escape_dir / (torch.norm(escape_dir, dim=1, keepdim=True) + 1e-8)

            # 沿混合逃逸方向施加大扰动
            dx[rescue_mask] = escape_dir * self.current_cmd_scales[rescue_mask].unsqueeze(1) * \
                              torch.rand(1, device=self.device) * 0.5 * (self.ub - self.lb)

        # 合成最终的候选位置
        trial_pop = prev_pop_gen + dx
        trial_pop = clamp(trial_pop, self.lb, self.ub)

        # 【极其关键】：因为我们修改了物理位置，必须对被修改过位移的个体进行重新评估！
        # 凡是 scale 不等于 1，或者被 rescue 的个体，都算 modified
        modified_mask = (self.current_cmd_scales != 1.0) | rescue_mask
        if modified_mask.any():
            trial_fit = curr_fit.clone()
            # 扣除额外的评估开销，只评被修改过的，完美契合 Intervention Budget!
            trial_fit[modified_mask] = self.evaluate(trial_pop[modified_mask])
            curr_pop = trial_pop
            curr_fit = trial_fit

        # =====================================================================
        # 第三阶段：司令部审查规则
        # =====================================================================
        margin = self.current_cmd_alpha_margin * torch.abs(prev_fit_gen) + self.current_cmd_beta_margin
        accept_mask = curr_fit < (prev_fit_gen + margin)

        elite_mask = self.current_cmd_protected
        elite_accept = curr_fit < prev_fit_gen
        accept_mask[elite_mask] = elite_accept[elite_mask]

        reject_mask = ~accept_mask
        curr_pop[reject_mask] = prev_pop_gen[reject_mask]
        curr_fit[reject_mask] = prev_fit_gen[reject_mask]

        self.base.pop = curr_pop
        self.base.fit = curr_fit

        # =====================================================================
        # 收尾状态更新
        # =====================================================================
        improved = curr_fit < self.prev_fit
        self.stagnation = torch.where(improved, torch.zeros_like(self.stagnation), self.stagnation + 1)

        self.prev_pop = curr_pop.clone()
        self.prev_fit = curr_fit.clone()

        current_best_val, current_best_idx = torch.min(self.base.fit, dim=0)
        if current_best_val < self.global_best_fit:
            self.global_best_fit = current_best_val.clone()
            self.global_best_location = self.base.pop[current_best_idx].clone()

        # 兼容性同步
        if hasattr(self.base, 'global_best_fit'):
            self.base.global_best_fit = self.global_best_fit.clone()
            self.base.global_best_location = self.global_best_location.clone()

        self.step_counter = self.step_counter + 1