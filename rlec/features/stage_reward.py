import torch


class StageRewardCalculator:
    def __init__(self, w_best=2.0, w_median=0.5, w_div=0.2, w_rescue=1.0, w_cost=0.5, w_collapse=1.0):
        self.w_best = w_best
        self.w_median = w_median
        self.w_div = w_div
        self.w_rescue = w_rescue
        self.w_cost = w_cost
        self.w_collapse = w_collapse  # 新增：塌缩惩罚权重

    def calculate(self,
                  fit_start: torch.Tensor, fit_end: torch.Tensor,
                  div_start: float, div_end: float, target_div: float, initial_div: float,
                  stagnation_start: torch.Tensor, stag_ratio: float,
                  action: torch.Tensor) -> float:

        # 1. 最优值收益
        best_start, best_end = torch.min(fit_start), torch.min(fit_end)
        r_best = torch.log1p(torch.relu(best_start - best_end)) / (torch.log1p(torch.abs(best_start)) + 1e-8)

        # 2. 中位数收益
        median_start, median_end = torch.median(fit_start), torch.median(fit_end)
        r_median = torch.log1p(torch.relu(median_start - median_end)) / (torch.log1p(torch.abs(median_start)) + 1e-8)

        # 3. 多样性控制收益 (向目标靠拢)
        err_start = abs(div_start - target_div)
        err_end = abs(div_end - target_div)
        r_div = (err_start - err_end) / (initial_div + 1e-8)

        # 4. 停滞脱困收益
        is_stagnant_start = (stagnation_start >= 10)
        num_stagnant = is_stagnant_start.sum().item()
        if num_stagnant > 0:
            rel_improve = (fit_start - fit_end) / (torch.abs(fit_start) + 1e-8)
            r_rescue = (is_stagnant_start & (rel_improve > 1e-4)).sum().item() / num_stagnant
        else:
            r_rescue = 0.0

        # 5. 干预成本
        c_intervene = (action[3].item() + action[4].item()) / 2.0

        # 6. 【必须改 4 应用】：过早塌缩惩罚 (Collapse Penalty)
        # 如果多样性跌破初始值的 1% (极度塌缩)，且全场有大量粒子处于停滞期，严惩！
        tau_div = initial_div * 0.01
        collapse_penalty = 0.0
        if div_end < tau_div and stag_ratio > 0.5:
            collapse_penalty = (tau_div - div_end) / tau_div * stag_ratio

        total_reward = (
                self.w_best * r_best.item() +
                self.w_median * r_median.item() +
                self.w_div * r_div +
                self.w_rescue * r_rescue -
                self.w_cost * c_intervene -
                self.w_collapse * collapse_penalty
        )

        roi = total_reward / (c_intervene + 1e-4)
        return total_reward, roi