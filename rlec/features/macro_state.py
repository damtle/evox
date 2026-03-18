import torch
from rlec.utils.population_metrics import (
    compute_diversity,
    compute_fitness_skewness,
    compute_rank_entropy,
    compute_stagnation_ratio,
    compute_elite_separation,
)


class MacroStateBuilder:
    def __init__(self, dim: int, pop_size: int):
        self.dim = dim
        self.pop_size = pop_size
        self.state_dim = 30  # 8 base + 6 action + 4 feedback + 12 niche summary

    def build(
        self,
        pop_t: torch.Tensor,
        fit_t: torch.Tensor,
        pop_t_minus_k: torch.Tensor,
        fit_t_minus_k: torch.Tensor,
        stagnation: torch.Tensor,
        last_action: torch.Tensor,
        feedback_stats: torch.Tensor,
        niche_summary: torch.Tensor,
    ) -> torch.Tensor:
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

        state = torch.cat(
            [
                torch.clamp(prog_best.unsqueeze(0), -5.0, 5.0),
                torch.clamp(prog_median.unsqueeze(0), -5.0, 5.0),
                torch.clamp(div_t.unsqueeze(0) / 100.0, 0.0, 5.0),
                torch.clamp(div_change.unsqueeze(0), -1.0, 1.0),
                stag_ratio.unsqueeze(0),
                skewness.unsqueeze(0) / 5.0,
                entropy.unsqueeze(0),
                torch.clamp(separation.unsqueeze(0) / 10.0, 0.0, 5.0),
                torch.clamp(last_action.flatten(), 0.0, 1.0),
                feedback_stats,
                niche_summary,
            ],
            dim=0,
        )
        return state.unsqueeze(0)
