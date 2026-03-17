import torch


class StageRewardCalculator:
    def __init__(self):
        # 简化权重配置
        pass

    def calculate(self,
                  fit_start: torch.Tensor, fit_end: torch.Tensor,
                  div_start: float, div_end: float, target_div: float, initial_div: float,
                  stagnation_start: torch.Tensor, stag_ratio_start: float, stag_ratio_end: float,
                  action: torch.Tensor,
                  feedbacks: torch.Tensor) -> float:
        # 解析反馈参数
        intervene_ratio, accept_ratio, rescue_success, elite_damage = feedbacks

        # A. 阶段性能项 (Performance)
        best_start, best_end = torch.min(fit_start), torch.min(fit_end)
        median_start, median_end = torch.median(fit_start), torch.median(fit_end)

        # 新增 20% 分位改善
        q20_start = torch.quantile(fit_start, 0.2)
        q20_end = torch.quantile(fit_end, 0.2)

        r_best = torch.log1p(torch.relu(best_start - best_end)) / (torch.log1p(torch.abs(best_start)) + 1e-8)
        r_median = torch.log1p(torch.relu(median_start - median_end)) / (torch.log1p(torch.abs(median_start)) + 1e-8)
        r_q20 = torch.log1p(torch.relu(q20_start - q20_end)) / (torch.log1p(torch.abs(q20_start)) + 1e-8)

        r_perf = 2.0 * r_best.item() + 0.5 * r_median.item() + 1.0 * r_q20.item()

        # B. 控制效果项 (Control Credit Assignment)
        # e/x/b 的约束：如果干预了很多，但没被接受，则惩罚；被接受了就奖励
        r_efficiency = accept_ratio.item() - 0.5 * intervene_ratio.item()
        # r 的约束：抢救成功率
        r_rescue = rescue_success.item()
        # p 的约束：惩罚精英破坏
        r_elite_stab = -1.0 * elite_damage.item()
        # d 的约束：多样性偏离惩罚
        div_error = abs(div_end - target_div) / (initial_div + 1e-8)
        r_div_overshoot = -0.5 * div_error

        # stagnation reduction credit: encourage fewer stuck particles at stage end.
        r_stag = stag_ratio_start - stag_ratio_end

        # mild action regularization: avoid extreme exploration-exploitation imbalance.
        action = torch.clamp(action, 0.0, 1.0)
        e_t, x_t, _, b_t, _, _ = action.tolist()
        r_balance = -0.1 * abs(e_t - x_t)
        r_budget_safety = -0.05 * max(0.0, b_t - 0.8)

        r_ctrl = r_efficiency + r_rescue + r_elite_stab + r_div_overshoot + r_stag + r_balance + r_budget_safety

        total_reward = r_perf + r_ctrl
        return total_reward
