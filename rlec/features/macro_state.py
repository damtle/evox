import torch
from rlec.utils.population_metrics import (
    compute_diversity,
    compute_fitness_skewness,
    compute_rank_entropy,
    compute_stagnation_ratio,
    compute_elite_separation  # 引入新特征
)


class MacroStateBuilder:
    def __init__(self, dim: int, pop_size: int):
        self.dim = dim
        self.pop_size = pop_size
        self.state_dim = 18  # 8(基础) + 6(动作) + 4(反馈)

    def build(self,
              pop_t: torch.Tensor, fit_t: torch.Tensor,
              pop_t_minus_k: torch.Tensor, fit_t_minus_k: torch.Tensor,
              stagnation: torch.Tensor,
              last_action: torch.Tensor,
              feedback_stats: torch.Tensor) -> torch.Tensor:  # 新增 feedback 参数

        device = pop_t.device

        # 1. 进展与结构特征 (计算同原版)
        best_t, best_t_k = torch.min(fit_t), torch.min(fit_t_minus_k)
        median_t, median_t_k = torch.median(fit_t), torch.median(fit_t_minus_k)
        prog_best = (best_t_k - best_t) / (torch.abs(best_t_k) + 1e-8)
        prog_median = (median_t_k - median_t) / (torch.abs(median_t_k) + 1e-8)

        div_t = compute_diversity(pop_t)
        div_t_k = compute_diversity(pop_t_minus_k)
        div_change = (div_t - div_t_k) / (div_t_k + 1e-8)
        stag_ratio = compute_stagnation_ratio(stagnation, threshold=10)
        separation = compute_elite_separation(pop_t, fit_t)
        skewness = compute_fitness_skewness(fit_t)
        entropy = compute_rank_entropy(fit_t)

        # 拼接 18 维状态 (去除 ROI，全量保留 action)
        state = torch.cat([
            torch.clamp(prog_best.unsqueeze(0), -5.0, 5.0),
            torch.clamp(prog_median.unsqueeze(0), -5.0, 5.0),
            torch.clamp(div_t.unsqueeze(0) / 100.0, 0.0, 5.0),
            torch.clamp(div_change.unsqueeze(0), -1.0, 1.0),
            stag_ratio.unsqueeze(0),
            skewness.unsqueeze(0) / 5.0,
            entropy.unsqueeze(0),
            torch.clamp(separation.unsqueeze(0) / 10.0, 0.0, 5.0),
            last_action.flatten(),  # 6维直接展平加入
            feedback_stats  # 4维：[intervene_ratio, accept_ratio, rescue_success, elite_damage]
        ], dim=0)

        return state.unsqueeze(0)  # [1, 18]