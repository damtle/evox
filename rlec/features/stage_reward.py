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
        subpop_stats: torch.Tensor,
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

        # subpop_stats = [
        #   exploit_stability,
        #   explore_birth_efficiency,
        #   bridge_transfer_efficiency,
        #   migration_efficiency,
        #   subpop_balance_penalty,
        # ]
        (
            exploit_stability,
            explore_birth_efficiency,
            bridge_transfer_efficiency,
            migration_efficiency,
            subpop_balance_penalty,
        ) = subpop_stats.tolist()

        r_subpop = (
            0.25 * exploit_stability
            + 0.20 * explore_birth_efficiency
            + 0.15 * bridge_transfer_efficiency
            + 0.15 * migration_efficiency
            - 0.20 * subpop_balance_penalty
        )

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

        total_reward = r_perf + r_ctrl + r_precision + r_subpop
        return total_reward
