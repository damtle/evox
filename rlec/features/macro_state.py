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
        self.state_dim = 13  # 【修改】：状态维度提升为 13

    def build(self,
              pop_t: torch.Tensor, fit_t: torch.Tensor,
              pop_t_minus_k: torch.Tensor, fit_t_minus_k: torch.Tensor,
              stagnation: torch.Tensor,
              last_action: torch.Tensor, last_roi: float) -> torch.Tensor:
        device = pop_t.device

        # 1. 进展特征
        best_t = torch.min(fit_t)
        best_t_k = torch.min(fit_t_minus_k)
        median_t = torch.median(fit_t)
        median_t_k = torch.median(fit_t_minus_k)

        prog_best = (best_t_k - best_t) / (torch.abs(best_t_k) + 1e-8)
        prog_median = (median_t_k - median_t) / (torch.abs(median_t_k) + 1e-8)

        # 2. 结构特征
        div_t = compute_diversity(pop_t)
        div_t_k = compute_diversity(pop_t_minus_k)
        div_change = (div_t - div_t_k) / (div_t_k + 1e-8)
        stag_ratio = compute_stagnation_ratio(stagnation, threshold=10)

        # 【必须改 4 应用】：计算分离度
        separation = compute_elite_separation(pop_t, fit_t)

        # 3. 分布特征
        skewness = compute_fitness_skewness(fit_t)
        entropy = compute_rank_entropy(fit_t)

        # 4. 控制历史
        roi_tensor = torch.tensor([last_roi], device=device)

        state = torch.cat([
            torch.clamp(prog_best.unsqueeze(0), -5.0, 5.0),
            torch.clamp(prog_median.unsqueeze(0), -5.0, 5.0),
            torch.clamp(div_t.unsqueeze(0) / 100.0, 0.0, 5.0),
            torch.clamp(div_change.unsqueeze(0), -1.0, 1.0),
            stag_ratio.unsqueeze(0),
            skewness.unsqueeze(0) / 5.0,
            entropy.unsqueeze(0),
            torch.clamp(separation.unsqueeze(0) / 10.0, 0.0, 5.0),  # 新增：聚类分离度
            roi_tensor,
            last_action[0].unsqueeze(0),
            last_action[1].unsqueeze(0),
            last_action[3].unsqueeze(0),
            last_action[4].unsqueeze(0),
        ], dim=0)

        return state.unsqueeze(0)  # [1, 13]