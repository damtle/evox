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
        niche_roles: Optional[Dict[str, torch.Tensor]] = None,
    ) -> EACommand:
        device = fit.device
        n = self.pop_size

        anchor_mask = None
        scout_mask = None
        completed_mask = None
        guardian_mask = None
        if niche_roles is not None:
            anchor_mask = niche_roles.get("anchor_mask")
            scout_mask = niche_roles.get("scout_mask")
            completed_mask = niche_roles.get("completed_mask")
            guardian_mask = niche_roles.get("guardian_mask")

        # Lower fit is better. rank_percentile: 0 -> best, 1 -> worst
        _, rank_idx = torch.sort(fit, descending=True)
        rank_percentile = torch.empty(n, device=device)
        rank_percentile[rank_idx] = torch.arange(n, dtype=torch.float32, device=device) / (n - 1)

        fit_std = torch.std(fit) + 1e-8
        best_fit = torch.min(fit)
        norm_gap_to_best = (fit - best_fit) / fit_std
        near_best_mask = norm_gap_to_best < 0.5
        good_stagnation_mask = (stagnation >= self.max_stag) & (rank_percentile < 0.3) & near_best_mask

        mid_pref = 1.0 - torch.abs(rank_percentile - 0.5) * 2.0
        bad_stagnation = (stagnation / self.max_stag) * rank_percentile
        score = 0.55 * bad_stagnation + 0.30 * mid_pref + 0.15 * rank_percentile

        n_intervene = int(n * intent.b_t)
        intervene_mask = torch.zeros(n, dtype=torch.bool, device=device)
        if n_intervene > 0:
            _, intervene_idx = torch.topk(score, n_intervene)
            intervene_mask[intervene_idx] = True

        if scout_mask is not None and scout_mask.any() and n_intervene > 0:
            extra = max(1, int(0.15 * n_intervene))
            scout_score = score.clone()
            scout_score[~scout_mask] = -1e9
            _, scout_idx = torch.topk(scout_score, min(extra, int(scout_mask.sum().item())))
            intervene_mask[scout_idx] = True

        intervene_mask = intervene_mask & (~good_stagnation_mask)

        k_elite = max(1, int(n * 0.10 * intent.p_t))
        _, sorted_indices = torch.sort(fit)
        protected_mask = torch.zeros(n, dtype=torch.bool, device=device)
        if intent.p_t > 0.05:
            protected_mask[sorted_indices[:k_elite]] = True

        intervene_mask = intervene_mask & (~protected_mask)

        num_rescue = int(n_intervene * intent.r_t)
        rescue_mask = torch.zeros(n, dtype=torch.bool, device=device)
        if num_rescue > 0:
            rescue_priority = stagnation.clone()
            rescue_priority[~intervene_mask] = -1.0
            rescue_priority[good_stagnation_mask] = -1.0
            rescue_priority[rank_percentile < 0.3] = -1.0
            if scout_mask is not None:
                rescue_priority = rescue_priority + 1.0 * scout_mask.float()
            _, rescue_idx = torch.topk(rescue_priority, num_rescue)
            rescue_mask[rescue_idx] = True

        rescue_mask = rescue_mask & (~protected_mask)
        middle_mask = intervene_mask & (~rescue_mask) & (~protected_mask)

        if anchor_mask is not None:
            rescue_mask = rescue_mask & (~anchor_mask)
            if guardian_mask is not None:
                anchor_guardian = anchor_mask & guardian_mask
                protected_mask = protected_mask | anchor_guardian
                middle_mask = middle_mask & (~anchor_guardian)
            else:
                middle_mask = middle_mask & (~anchor_mask)

        if completed_mask is not None and guardian_mask is not None:
            completed_non_guard = completed_mask & (~guardian_mask)
            rescue_mask = rescue_mask & (~completed_non_guard)
            middle_mask = middle_mask & (~completed_non_guard)

        global_scale = 1.0 + 2.0 * intent.e_t
        step_scales = torch.ones(n, device=device)
        step_scales[middle_mask] = 1.0 + intent.e_t - 0.5 * intent.x_t
        step_scales[rescue_mask] = global_scale

        if anchor_mask is not None:
            step_scales[anchor_mask] = step_scales[anchor_mask] * 0.3
        if scout_mask is not None:
            step_scales[scout_mask & middle_mask] = step_scales[scout_mask & middle_mask] * 1.15
            step_scales[scout_mask & rescue_mask] = step_scales[scout_mask & rescue_mask] * 1.20

        alpha_margin = 0.02 * intent.e_t - 0.02 * intent.x_t
        beta_margin = 1e-5 * intent.e_t - 1e-5 * intent.x_t
        target_diversity = initial_div * intent.d_t

        return EACommand(
            protected_mask=protected_mask,
            rescue_mask=rescue_mask,
            middle_mask=middle_mask,
            quiet_mask=good_stagnation_mask,
            step_scales=step_scales,
            alpha_margin=alpha_margin,
            beta_margin=beta_margin,
            target_diversity=target_diversity,
        )
