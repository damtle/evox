import torch
from dataclasses import dataclass
from .intent_vector import IntentVector


@dataclass
class EACommand:
    protected_mask: torch.Tensor
    rescue_mask: torch.Tensor
    middle_mask: torch.Tensor  # 新增：中间组标识
    step_scales: torch.Tensor
    alpha_margin: float  # 新增：相对容忍度
    beta_margin: float  # 新增：绝对容忍度 (防止后期 f 极小时门槛过死)
    target_diversity: float


class ControlInterpreter:
    def __init__(self, pop_size: int, max_stagnation: int = 10):
        self.pop_size = pop_size
        self.max_stag = max_stagnation

    def interpret(self, intent: IntentVector, fit: torch.Tensor,
                  stagnation: torch.Tensor, initial_div: float) -> EACommand:
        device = fit.device
        N = self.pop_size

        # --- 1. 优先度与干预预算 (b_t) ---
        _, rank_idx = torch.sort(fit, descending=True)
        rank_percentile = torch.empty(N, device=device)
        rank_percentile[rank_idx] = torch.arange(N, dtype=torch.float32, device=device) / (N - 1)

        # 【建议修 2】：让分数偏向“中间层”(mid_pref)，而不是单纯的头部
        mid_pref = 1.0 - torch.abs(rank_percentile - 0.5) * 2.0
        score = 0.6 * (stagnation / self.max_stag) + 0.4 * mid_pref

        n_intervene = int(N * intent.b_t)

        intervene_mask = torch.zeros(N, dtype=torch.bool, device=device)
        if n_intervene > 0:
            _, intervene_idx = torch.topk(score, n_intervene)
            intervene_mask[intervene_idx] = True

        # --- 2. 精英保护指令 (p_t) ---
        k_elite = max(1, int(N * 0.10 * intent.p_t))
        _, sorted_indices = torch.sort(fit)
        protected_mask = torch.zeros(N, dtype=torch.bool, device=device)
        if intent.p_t > 0.05:
            protected_mask[sorted_indices[:k_elite]] = True

        # 【建议修 2】：让精英绝对免于被算作“常规干预预算 (b_t)”
        intervene_mask = intervene_mask & (~protected_mask)
        # --- 3. 救援预算指令 (r_t) ---
        num_rescue = int(n_intervene * intent.r_t)
        rescue_mask = torch.zeros(N, dtype=torch.bool, device=device)
        if num_rescue > 0:
            intervene_stag = stagnation.clone()
            intervene_stag[~intervene_mask] = -1.0
            _, rescue_idx = torch.topk(intervene_stag, num_rescue)
            rescue_mask[rescue_idx] = True
        rescue_mask = rescue_mask & (~protected_mask)

        # --- 4. 中间组定位 (Middle Group) ---
        # 既不是被保护的精英，也不是被抢救的停滞个体，承担定向扩散主力！
        middle_mask = intervene_mask & (~rescue_mask) & (~protected_mask)

        # --- 5. 物理步长缩放分配 (Scales) ---
        global_scale = 1.0 + 2.0 * intent.e_t
        local_scale = 1.0 - 0.9 * intent.x_t

        step_scales = torch.ones(N, device=device)
        # 中间组：受探索和开发的混合牵引 (e_t主导扩散，x_t主导收敛)
        step_scales[middle_mask] = 1.0 + intent.e_t - 0.5 * intent.x_t
        step_scales[rescue_mask] = global_scale
        # step_scales[intervene_mask & protected_mask] = local_scale

        # --- 6. 阶段接受门槛 (Relative + Absolute) ---
        # 相对比例 alpha_margin
        alpha_margin = 0.02 * intent.e_t - 0.02 * intent.x_t
        # 绝对底线 beta_margin (防止 |f| 接近 0 时，margin 退化为 0)
        beta_margin = 1e-5 * intent.e_t - 1e-5 * intent.x_t

        # --- 7. 稳健的多样性目标 ---
        # 彻底摆脱“当前状态的线性缩放”，严格锚定最初代的原始分布！
        # d_t=0.5时维持初始一半，d_t=1.0时扩到初始规模
        target_diversity = initial_div * (0.1 + 0.9 * intent.d_t)

        return EACommand(
            protected_mask=protected_mask, rescue_mask=rescue_mask, middle_mask=middle_mask,
            step_scales=step_scales, alpha_margin=alpha_margin, beta_margin=beta_margin,
            target_diversity=target_diversity
        )