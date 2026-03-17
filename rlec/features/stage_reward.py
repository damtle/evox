import torch


class StageRewardCalculator:
    def __init__(self):
        pass

    def calculate(
        self,
        fit_start: torch.Tensor,
        fit_end: torch.Tensor,
        div_start: float,
        div_end: float,
        target_div: float,
        initial_div: float,
        stagnation_start: torch.Tensor,
        stag_ratio_start: float,
        stag_ratio_end: float,
        action: torch.Tensor,
        feedbacks: torch.Tensor,
        niche_stats: torch.Tensor,
    ) -> float:
        intervene_ratio, accept_ratio, rescue_success, elite_damage = feedbacks

        best_start, best_end = torch.min(fit_start), torch.min(fit_end)
        median_start, median_end = torch.median(fit_start), torch.median(fit_end)
        q20_start = torch.quantile(fit_start, 0.2)
        q20_end = torch.quantile(fit_end, 0.2)

        r_best = torch.log1p(torch.relu(best_start - best_end)) / (torch.log1p(torch.abs(best_start)) + 1e-8)
        r_median = torch.log1p(torch.relu(median_start - median_end)) / (torch.log1p(torch.abs(median_start)) + 1e-8)
        r_q20 = torch.log1p(torch.relu(q20_start - q20_end)) / (torch.log1p(torch.abs(q20_start)) + 1e-8)
        r_perf = 2.0 * r_best.item() + 0.5 * r_median.item() + 1.0 * r_q20.item()

        best_end_safe = torch.clamp(best_end, min=0.0)
        r_precision = 0.02 * (-torch.log(best_end_safe + 1e-12)).item()
        r_precision = min(r_precision, 0.8)

        r_efficiency = accept_ratio.item() - 0.5 * intervene_ratio.item()
        r_rescue = rescue_success.item()
        r_elite_stab = -1.0 * elite_damage.item()
        div_error = abs(div_end - target_div) / (initial_div + 1e-8)
        r_div_overshoot = -0.5 * div_error
        r_stag = stag_ratio_start - stag_ratio_end

        action = torch.clamp(action, 0.0, 1.0)
        e_t, x_t, _, b_t, _, _ = action.tolist()
        r_balance = -0.02 * abs(e_t - x_t)
        r_budget_safety = -0.05 * max(0.0, b_t - 0.8)
        compact_basin = (div_end / (initial_div + 1e-8)) < 0.12
        r_late_settle = (-0.2 * e_t - 0.15 * b_t) if compact_basin else 0.0

        # Niche-aware credit terms.
        # niche_stats = [new_niche_norm, archive_unique_norm, repeat_ratio, guardian_stability, release_efficiency]
        new_niche_norm, archive_unique_norm, repeat_ratio, guardian_stability, release_efficiency = niche_stats.tolist()
        r_new_niche = 0.20 * new_niche_norm
        r_archive_unique = 0.20 * archive_unique_norm
        r_repeat_penalty = -0.25 * repeat_ratio
        r_guardian_stability = 0.20 * guardian_stability
        r_release_eff = 0.20 * release_efficiency
        r_niche = r_new_niche + r_archive_unique + r_repeat_penalty + r_guardian_stability + r_release_eff

        r_ctrl = (
            r_efficiency
            + r_rescue
            + r_elite_stab
            + r_div_overshoot
            + r_stag
            + r_balance
            + r_budget_safety
            + r_late_settle
        )

        total_reward = r_perf + r_ctrl + r_precision + r_niche
        return total_reward
