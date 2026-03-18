import torch
from dataclasses import dataclass
from typing import Dict, Optional

from .intent_vector import IntentVector


@dataclass
class EACommand:
    protected_mask: torch.Tensor
    rescue_mask: torch.Tensor
    middle_mask: torch.Tensor
    quiet_mask: torch.Tensor
    step_scales: torch.Tensor
    alpha_margin: float
    beta_margin: float
    target_diversity: float


class ControlInterpreter:
    def __init__(self, pop_size: int, max_stagnation: int = 10):
        self.pop_size = pop_size
        self.max_stag = max_stagnation

    def interpret(
        self,
        intent: IntentVector,
        fit: torch.Tensor,
        stagnation: torch.Tensor,
        initial_div: float,
        subpop_roles: Optional[Dict[str, torch.Tensor]] = None,
    ) -> EACommand:
        device = fit.device
        n = self.pop_size

        exploit_mask = torch.zeros(n, dtype=torch.bool, device=device)
        bridge_mask = torch.zeros(n, dtype=torch.bool, device=device)
        explore_mask = torch.zeros(n, dtype=torch.bool, device=device)
        exploit_guardian_mask = torch.zeros(n, dtype=torch.bool, device=device)

        if subpop_roles is not None:
            exploit_mask = subpop_roles.get("exploit_mask", exploit_mask)
            bridge_mask = subpop_roles.get("bridge_mask", bridge_mask)
            explore_mask = subpop_roles.get("explore_mask", explore_mask)
            exploit_guardian_mask = subpop_roles.get("exploit_guardian_mask", exploit_guardian_mask)

        _, rank_idx = torch.sort(fit, descending=True)
        rank_percentile = torch.empty(n, device=device)
        rank_percentile[rank_idx] = torch.arange(n, dtype=torch.float32, device=device) / max(1, n - 1)

        fit_std = torch.std(fit) + 1e-8
        best_fit = torch.min(fit)
        norm_gap_to_best = (fit - best_fit) / fit_std
        near_best_mask = norm_gap_to_best < 0.5
        good_stagnation_mask = (stagnation >= self.max_stag) & near_best_mask

        mid_pref = 1.0 - torch.abs(rank_percentile - 0.5) * 2.0
        bad_stagnation = (stagnation / self.max_stag) * rank_percentile
        score = 0.55 * bad_stagnation + 0.30 * mid_pref + 0.15 * rank_percentile

        # Prioritize interventions by role intent: Explore > Bridge > Exploit
        score = score + 0.20 * explore_mask.float() + 0.08 * bridge_mask.float() - 0.10 * exploit_mask.float()

        n_intervene = int(n * intent.b_t)
        intervene_mask = torch.zeros(n, dtype=torch.bool, device=device)
        if n_intervene > 0:
            _, intervene_idx = torch.topk(score, n_intervene)
            intervene_mask[intervene_idx] = True

        k_elite = max(1, int(n * 0.10 * intent.p_t))
        _, sorted_indices = torch.sort(fit)
        protected_mask = torch.zeros(n, dtype=torch.bool, device=device)
        if intent.p_t > 0.05:
            protected_mask[sorted_indices[:k_elite]] = True
        protected_mask = protected_mask | exploit_guardian_mask

        intervene_mask = intervene_mask & (~protected_mask)
        intervene_mask = intervene_mask & (~good_stagnation_mask)

        num_rescue = int(n_intervene * intent.r_t)
        rescue_mask = torch.zeros(n, dtype=torch.bool, device=device)
        if num_rescue > 0:
            rescue_priority = stagnation.clone()
            rescue_priority[~intervene_mask] = -1.0
            rescue_priority[exploit_mask] = -1.0
            rescue_priority = rescue_priority + 1.5 * explore_mask.float() + 0.6 * bridge_mask.float()
            _, rescue_idx = torch.topk(rescue_priority, num_rescue)
            rescue_mask[rescue_idx] = True

        rescue_mask = rescue_mask & (~protected_mask)
        rescue_mask = rescue_mask & (~exploit_mask)

        middle_mask = intervene_mask & (~rescue_mask) & (~protected_mask)
        middle_mask = middle_mask & (~exploit_guardian_mask)

        global_scale = 1.0 + 2.0 * intent.e_t
        step_scales = torch.ones(n, device=device)
        step_scales[middle_mask] = 1.0 + intent.e_t - 0.5 * intent.x_t
        step_scales[rescue_mask] = global_scale

        # Role-specific scaling
        step_scales[exploit_mask] = step_scales[exploit_mask] * (0.20 + 0.30 * (1.0 - intent.e_t))
        step_scales[bridge_mask] = step_scales[bridge_mask] * 0.80
        step_scales[explore_mask] = step_scales[explore_mask] * (1.20 + 0.80 * intent.e_t)

        alpha_margin = 0.02 * intent.e_t - 0.02 * intent.x_t
        beta_margin = 1e-5 * intent.e_t - 1e-5 * intent.x_t
        target_diversity = initial_div * intent.d_t

        quiet_mask = good_stagnation_mask & exploit_mask

        return EACommand(
            protected_mask=protected_mask,
            rescue_mask=rescue_mask,
            middle_mask=middle_mask,
            quiet_mask=quiet_mask,
            step_scales=step_scales,
            alpha_margin=alpha_margin,
            beta_margin=beta_margin,
            target_diversity=target_diversity,
        )
