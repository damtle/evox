from __future__ import annotations

import os
from typing import Optional

import torch

from evox.core import Algorithm, Mutable
from evox.utils import clamp

from rlec.control.intent_vector import IntentVector
from rlec.control.interpreter import ControlInterpreter
from rlec.control.niche_manager import NicheManager
from rlec.features.macro_state import MacroStateBuilder
from rlec.features.stage_reward import StageRewardCalculator
from rlec.rl.ppo import PPO
from rlec.utils.niche_logger import NicheLogger
from rlec.utils.population_metrics import compute_diversity


class RLECWrapper(Algorithm):
    """RLEC wrapper with global 6D intent + niche lifecycle control."""

    def __init__(
        self,
        base_algo: Algorithm,
        stage_length: int = 10,
        update_stages: int = 2,
        device: Optional[torch.device] = None,
        func_id: Optional[int] = None,
        run_id: Optional[int] = None,
        log_dir: Optional[str] = None,
    ):
        super().__init__()
        self.base = base_algo
        self.K = stage_length
        self.update_stages = update_stages
        self.device = device if device is not None else base_algo.fit.device

        self.func_id = func_id
        self.run_id = run_id
        self.log_dir = log_dir
        self.niche_logger = NicheLogger(log_dir) if log_dir is not None else None

        self.pop_size, self.dim = base_algo.pop.shape[0], base_algo.pop.shape[1]

        self.state_builder = MacroStateBuilder(self.dim, self.pop_size)
        self.reward_calc = StageRewardCalculator()
        self.interpreter = ControlInterpreter(self.pop_size)
        self.niche_manager = NicheManager(self.pop_size, self.dim, self.device)

        self.rl_agent = PPO(state_dim=self.state_builder.state_dim, action_dim=6, device=self.device)

        self.step_counter = Mutable(torch.tensor(0, device=self.device))
        self.stage_counter = Mutable(torch.tensor(0, device=self.device))

        self.initial_div = Mutable(torch.tensor(0.0, device=self.device))
        self.stagnation = Mutable(torch.zeros(self.pop_size, device=self.device))

        self.prev_pop = Mutable(base_algo.pop.clone())
        self.prev_fit = Mutable(torch.full_like(base_algo.fit, torch.inf))

        self.stage_start_pop = Mutable(base_algo.pop.clone())
        self.stage_start_fit = Mutable(torch.full_like(base_algo.fit, torch.inf))
        self.stage_start_div = Mutable(torch.tensor(0.0, device=self.device))
        self.stage_start_stagnation = Mutable(torch.zeros(self.pop_size, device=self.device))

        self.hidden_a = Mutable(torch.zeros(1, 1, 64, device=self.device))
        self.hidden_c = Mutable(torch.zeros(1, 1, 64, device=self.device))
        self.last_state = Mutable(torch.zeros(1, self.state_builder.state_dim, device=self.device))
        self.last_action_raw = Mutable(torch.zeros(6, device=self.device))
        self.last_intent = Mutable(torch.zeros(6, device=self.device))
        self.last_intent_eff = Mutable(torch.zeros(6, device=self.device))
        self.last_logprob = Mutable(torch.tensor(0.0, device=self.device))
        self.last_value = Mutable(torch.tensor(0.0, device=self.device))

        self.current_cmd_protected = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_cmd_rescue = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_cmd_middle = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_cmd_quiet = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_cmd_target_div = Mutable(torch.tensor(0.0, device=self.device))
        self.current_cmd_alpha_margin = Mutable(torch.tensor(0.0, device=self.device))
        self.current_cmd_beta_margin = Mutable(torch.tensor(0.0, device=self.device))
        self.current_cmd_scales = Mutable(torch.ones(self.pop_size, device=self.device))

        self.current_niche_guardian = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_niche_converged = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))

        self.current_gate_refine = Mutable(torch.tensor(0.0, device=self.device))
        self.current_gate_recover = Mutable(torch.tensor(0.0, device=self.device))

        self.global_best_fit = Mutable(torch.tensor(torch.inf, device=self.device))
        self.global_best_location = Mutable(torch.zeros(self.dim, device=self.device))

        self.last_intervene_ratio = Mutable(torch.tensor(0.0, device=self.device))
        self.last_accept_ratio = Mutable(torch.tensor(0.0, device=self.device))
        self.last_rescue_success = Mutable(torch.tensor(0.0, device=self.device))
        self.last_elite_damage = Mutable(torch.tensor(0.0, device=self.device))

        self.niche_summary_vec = Mutable(torch.zeros(8, device=self.device))
        self.last_niche_reward_stats = Mutable(torch.zeros(5, device=self.device))

        self.last_n_niches = Mutable(torch.tensor(0.0, device=self.device))
        self.last_n_exploring = Mutable(torch.tensor(0.0, device=self.device))
        self.last_n_refining = Mutable(torch.tensor(0.0, device=self.device))
        self.last_n_converged = Mutable(torch.tensor(0.0, device=self.device))
        self.last_n_guardians = Mutable(torch.tensor(0.0, device=self.device))
        self.last_n_release = Mutable(torch.tensor(0.0, device=self.device))
        self.last_n_rescue = Mutable(torch.tensor(0.0, device=self.device))

        self.last_release_mask = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.last_release_baseline_fit = Mutable(torch.zeros(self.pop_size, device=self.device))
        self.last_release_repeat_ratio = Mutable(torch.tensor(0.0, device=self.device))
        self.last_archive_before_release = Mutable(torch.empty((0, self.dim), device=self.device))
        self.last_release_birth_pop = Mutable(torch.zeros((self.pop_size, self.dim), device=self.device))
        self.stage_guardian_mask = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.stage_guardian_start_best = Mutable(torch.tensor(torch.inf, device=self.device))

    @property
    def pop(self):
        return self.base.pop

    @pop.setter
    def pop(self, value):
        self.base.pop = value

    @property
    def fit(self):
        return self.base.fit

    @fit.setter
    def fit(self, value):
        self.base.fit = value

    @property
    def lb(self):
        return self.base.lb

    @property
    def ub(self):
        return self.base.ub

    def init_step(self):
        if hasattr(self, "evaluate"):
            self.base.evaluate = self.evaluate
        self.base.init_step()

        self.prev_pop = self.base.pop.clone()
        self.prev_fit = self.base.fit.clone()
        self.stage_start_pop = self.base.pop.clone()
        self.stage_start_fit = self.base.fit.clone()
        self.stage_start_div = compute_diversity(self.base.pop)
        self.initial_div = compute_diversity(self.base.pop)

        best_val, best_idx = torch.min(self.base.fit, dim=0)
        self.global_best_fit = best_val.clone()
        self.global_best_location = self.base.pop[best_idx].clone()

        self._dispatch_new_stage()

    def _compute_repeat_ratio(self, release_idx: torch.Tensor, archive_centers: torch.Tensor) -> float:
        if release_idx.numel() == 0 or archive_centers.numel() == 0:
            return 0.0
        released_pos = self.base.pop[release_idx]
        d = torch.cdist(released_pos, archive_centers)
        min_d = torch.min(d, dim=1).values
        span = torch.norm((self.ub - self.lb).squeeze(0)).item()
        near_th = max(1e-6, 0.05 * span / (self.dim ** 0.5))
        return float((min_d < near_th).float().mean().item())

    def _build_niche_summary(self, niche_plan) -> torch.Tensor:
        n_niches = max(1, len(niche_plan.niche_rows))
        largest_ratio = 0.0
        mean_radius_rel = 0.0
        if niche_plan.niche_rows:
            largest_ratio = max(row["size"] for row in niche_plan.niche_rows) / self.pop_size
            mean_radius_rel = sum(row["radius_rel"] for row in niche_plan.niche_rows) / len(niche_plan.niche_rows)

        n_exploring = len([r for r in niche_plan.niche_rows if r["status"] == "exploring"])
        n_refining = len([r for r in niche_plan.niche_rows if r["status"] == "refining"])
        n_converged = len([r for r in niche_plan.niche_rows if r["status"] == "converged"])

        summary = torch.tensor(
            [
                len(niche_plan.niche_rows) / self.pop_size,
                largest_ratio,
                n_refining / n_niches,
                n_converged / n_niches,
                float(self.last_n_release.item()) / self.pop_size,
                float(self.last_n_guardians.item()) / self.pop_size,
                min(1.0, float(self.niche_manager.archive_centers.shape[0]) / self.pop_size),
                mean_radius_rel,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        self.last_n_niches = torch.tensor(float(len(niche_plan.niche_rows)), device=self.device)
        self.last_n_exploring = torch.tensor(float(n_exploring), device=self.device)
        self.last_n_refining = torch.tensor(float(n_refining), device=self.device)
        self.last_n_converged = torch.tensor(float(n_converged), device=self.device)

        return summary

    def _log_stage_and_niches(self, current_div: torch.Tensor, niche_plan, intent_eff: torch.Tensor):
        if self.niche_logger is None:
            return

        stage_id = int(self.stage_counter.item())
        gen = int(self.step_counter.item())
        best_now = float(torch.min(self.base.fit).item())

        stage_row = {
            "func_id": self.func_id,
            "run_id": self.run_id,
            "gen": gen,
            "stage": stage_id,
            "best_fit": best_now,
            "global_div": float(current_div.item()),
            "intent_e": float(intent_eff[0].item()),
            "intent_x": float(intent_eff[1].item()),
            "intent_d": float(intent_eff[2].item()),
            "intent_b": float(intent_eff[3].item()),
            "intent_r": float(intent_eff[4].item()),
            "intent_p": float(intent_eff[5].item()),
            "g_refine": float(self.current_gate_refine.item()),
            "g_recover": float(self.current_gate_recover.item()),
            "n_niches": int(self.last_n_niches.item()),
            "n_exploring": int(self.last_n_exploring.item()),
            "n_refining": int(self.last_n_refining.item()),
            "n_converged": int(self.last_n_converged.item()),
            "n_guardians": int(self.last_n_guardians.item()),
            "n_release": int(self.last_n_release.item()),
            "n_rescue": int(self.last_n_rescue.item()),
            "archive_size": int(self.niche_manager.archive_centers.shape[0]),
            "intervene_ratio": float(self.last_intervene_ratio.item()),
            "accept_ratio": float(self.last_accept_ratio.item()),
            "rescue_success": float(self.last_rescue_success.item()),
            "elite_damage": float(self.last_elite_damage.item()),
        }
        self.niche_logger.log_stage(stage_row)

        niche_rows = []
        for row in niche_plan.niche_rows:
            out = {
                "func_id": self.func_id,
                "run_id": self.run_id,
                "gen": gen,
                "stage": stage_id,
                "niche_id": row["niche_id"],
                "niche_age": row["niche_age"],
                "size": row["size"],
                "status": row["status"],
                "center_norm": row["center_norm"],
                "radius": row["radius"],
                "radius_rel": row["radius_rel"],
                "radius_shrink": row["radius_shrink"],
                "best_fit": row["best_fit"],
                "best_improve_rel": row["best_improve_rel"],
                "accept_ratio_stage": row["accept_ratio_stage"],
                "improve_ratio_stage": row.get("improve_ratio_stage", row["accept_ratio_stage"]),
                "stag_ratio": row["stag_ratio"],
                "stag_growth": row["stag_growth"],
                "guardian_count": row["guardian_count"],
                "release_count": row["release_count"],
                "rescue_count": row["rescue_count"],
                "dist_to_archive_min": row["dist_to_archive_min"],
            }
            niche_rows.append(out)
        self.niche_logger.log_niches(niche_rows)

    def _dispatch_new_stage(self):
        current_pop = self.base.pop
        current_fit = self.base.fit
        current_div = compute_diversity(current_pop)
        archive_before = self.niche_manager.archive_centers.clone()

        feedback_stats = torch.stack(
            [
                self.last_intervene_ratio,
                self.last_accept_ratio,
                self.last_rescue_success,
                self.last_elite_damage,
            ]
        ).to(self.device)

        state = self.state_builder.build(
            pop_t=current_pop,
            fit_t=current_fit,
            pop_t_minus_k=self.stage_start_pop,
            fit_t_minus_k=self.stage_start_fit,
            stagnation=self.stagnation,
            last_action=self.last_intent_eff,
            feedback_stats=feedback_stats,
            niche_summary=self.niche_summary_vec,
        )

        action_raw, intent_np, log_prob, value = self.rl_agent.select_action(state, deterministic=False)

        self.last_state = state.clone()

        intent_tensor = torch.tensor(intent_np, device=self.device, dtype=torch.float32)

        stage_best_start = torch.min(self.stage_start_fit)
        stage_best_now = torch.min(current_fit)
        stage_progress = (stage_best_start - stage_best_now) / (torch.abs(stage_best_start) + 1e-8)

        div_ratio = current_div / (self.initial_div + 1e-8)
        best_now = torch.min(current_fit)
        median_now = torch.median(current_fit)
        spread_ratio = (median_now - best_now) / (torch.abs(best_now) + 1e-8)
        stag_ratio = torch.mean((self.stagnation >= 10).float())

        refine_div = torch.clamp((0.15 - div_ratio) / 0.15, 0.0, 1.0)
        refine_spread = torch.clamp((0.20 - spread_ratio) / 0.20, 0.0, 1.0)
        refine_prog = torch.clamp((1e-4 - stage_progress) / 1e-4, 0.0, 1.0)
        refine_stag = torch.clamp((stag_ratio - 0.2) / 0.6, 0.0, 1.0)
        g_refine = torch.clamp(0.35 * refine_div + 0.25 * refine_spread + 0.20 * refine_prog + 0.20 * refine_stag, 0.0, 1.0)

        recover_div = torch.clamp((0.18 - div_ratio) / 0.18, 0.0, 1.0)
        recover_prog = torch.clamp((1e-4 - stage_progress) / 1e-4, 0.0, 1.0)
        recover_spread = torch.clamp((spread_ratio - 0.20) / 0.40, 0.0, 1.0)
        g_recover = torch.clamp(0.40 * recover_div + 0.30 * recover_prog + 0.30 * recover_spread, 0.0, 1.0) * (1.0 - g_refine)

        e_eff = torch.clamp(intent_tensor[0] * (1.0 - 0.85 * g_refine) + 0.50 * g_recover * (1.0 - intent_tensor[0]), 0.0, 1.0)
        x_eff = torch.clamp(intent_tensor[1] + 0.25 * g_refine * (1.0 - intent_tensor[1]), 0.0, 1.0)
        d_eff = torch.clamp(intent_tensor[2] * (1.0 - 0.50 * g_refine) + 0.30 * g_recover * (1.0 - intent_tensor[2]), 0.0, 1.0)
        b_eff = torch.clamp(intent_tensor[3] * (1.0 - 0.75 * g_refine) + 0.45 * g_recover * (1.0 - intent_tensor[3]), 0.0, 1.0)
        r_eff = torch.clamp(intent_tensor[4] * (1.0 - 0.90 * g_refine) + 0.35 * g_recover * (1.0 - intent_tensor[4]), 0.0, 1.0)
        p_eff = torch.clamp(intent_tensor[5] + 0.30 * g_refine * (1.0 - intent_tensor[5]), 0.0, 1.0)

        intent_eff_tensor = torch.stack([e_eff, x_eff, d_eff, b_eff, r_eff, p_eff]).to(self.device)
        intent_eff_obj = IntentVector.from_tensor(intent_eff_tensor)

        self.current_gate_refine = g_refine.detach()
        self.current_gate_recover = g_recover.detach()

        cmd = self.interpreter.interpret(intent_eff_obj, current_fit, self.stagnation, float(self.initial_div.item()))

        self.last_action_raw = torch.tensor(action_raw, device=self.device, dtype=torch.float32)
        self.last_intent = intent_tensor.clone()
        self.last_intent_eff = intent_eff_tensor.clone()
        self.last_logprob = torch.tensor(log_prob, device=self.device, dtype=torch.float32)
        self.last_value = torch.tensor(value, device=self.device, dtype=torch.float32)

        self.current_cmd_protected = cmd.protected_mask.clone()
        self.current_cmd_middle = cmd.middle_mask.clone()
        self.current_cmd_quiet = cmd.quiet_mask.clone()
        self.current_cmd_rescue = cmd.rescue_mask.clone()
        self.current_cmd_scales = cmd.step_scales.clone()
        self.current_cmd_alpha_margin = torch.tensor(cmd.alpha_margin, device=self.device)
        self.current_cmd_beta_margin = torch.tensor(cmd.beta_margin, device=self.device)
        self.current_cmd_target_div = torch.tensor(cmd.target_diversity, device=self.device)

        niche_plan = self.niche_manager.plan(
            pop=current_pop,
            fit=current_fit,
            stage_start_pop=self.stage_start_pop,
            stage_start_fit=self.stage_start_fit,
            stage_start_stagnation=self.stage_start_stagnation,
            stagnation=self.stagnation,
            intent_eff=intent_eff_tensor,
            initial_div=float(self.initial_div.item()),
        )

        self.current_niche_guardian = niche_plan.guardian_mask.clone()
        self.current_niche_converged = niche_plan.converged_mask.clone()

        self.current_cmd_protected = self.current_cmd_protected | niche_plan.guardian_mask
        self.current_cmd_quiet = self.current_cmd_quiet | niche_plan.guardian_mask

        role_active = niche_plan.exploring_mask | niche_plan.refining_mask
        self.current_cmd_middle = self.current_cmd_middle & role_active & (~niche_plan.guardian_mask)
        self.current_cmd_rescue = (
            self.current_cmd_rescue
            & niche_plan.rescue_mask
            & niche_plan.exploring_mask
            & (~niche_plan.guardian_mask)
        )

        self.current_cmd_scales[niche_plan.refining_mask] = self.current_cmd_scales[niche_plan.refining_mask] * 0.55

        converged_non_guardian = niche_plan.converged_mask & (~niche_plan.guardian_mask)
        self.current_cmd_middle[converged_non_guardian] = False
        self.current_cmd_rescue[converged_non_guardian] = False
        self.current_cmd_scales[converged_non_guardian] = 0.0
        self.current_cmd_quiet[converged_non_guardian] = False

        self.current_cmd_alpha_margin = self.current_cmd_alpha_margin * (1.0 - 0.4 * g_refine)
        self.current_cmd_beta_margin = self.current_cmd_beta_margin * (1.0 - 0.4 * g_refine)

        release_idx = torch.where(niche_plan.release_mask)[0]
        self.last_release_mask = torch.zeros(self.pop_size, dtype=torch.bool, device=self.device)
        self.last_release_baseline_fit = torch.zeros(self.pop_size, device=self.device)

        if release_idx.numel() > 0:
            new_pos = self.niche_manager.sample_release_positions(
                int(release_idx.numel()),
                self.lb,
                self.ub,
                avoid_centers=niche_plan.active_centers,
            )
            self.base.pop[release_idx] = new_pos
            self.base.fit[release_idx] = self.evaluate(new_pos)
            self.stagnation[release_idx] = 0.0

            self.prev_pop[release_idx] = self.base.pop[release_idx]
            self.prev_fit[release_idx] = self.base.fit[release_idx]

            self.current_niche_converged[release_idx] = False
            self.current_niche_guardian[release_idx] = False
            self.current_cmd_quiet[release_idx] = False
            self.current_cmd_protected[release_idx] = False
            self.current_cmd_rescue[release_idx] = False
            self.current_cmd_middle[release_idx] = False
            self.current_cmd_scales[release_idx] = 1.0

            self.last_release_mask[release_idx] = True
            self.last_release_baseline_fit[release_idx] = self.base.fit[release_idx]
            self.last_release_birth_pop[release_idx] = self.base.pop[release_idx]
            self.last_archive_before_release = archive_before
            self.last_release_repeat_ratio = torch.tensor(
                self._compute_repeat_ratio(release_idx, archive_before),
                device=self.device,
            )
            current_div = compute_diversity(self.base.pop)
        else:
            self.last_release_repeat_ratio = torch.tensor(0.0, device=self.device)
            self.last_archive_before_release = archive_before

        self.last_n_guardians = torch.tensor(float(niche_plan.guardian_mask.sum().item()), device=self.device)
        self.last_n_release = torch.tensor(float(niche_plan.release_mask.sum().item()), device=self.device)
        self.last_n_rescue = torch.tensor(float(self.current_cmd_rescue.sum().item()), device=self.device)

        self.niche_summary_vec = self._build_niche_summary(niche_plan)

        stage_new_niche_norm = float(niche_plan.n_new_niches) / max(1.0, float(self.pop_size))
        stage_archive_unique_norm = float(niche_plan.archive_added) / max(1.0, float(self.pop_size))
        stage_repeat_ratio = float(self.last_release_repeat_ratio.item())
        # stage-level placeholders; finalized at stage end before reward.
        self.last_niche_reward_stats = torch.tensor(
            [
                stage_new_niche_norm,
                stage_archive_unique_norm,
                stage_repeat_ratio,
                0.0,
                0.0,
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self.stage_start_pop = self.base.pop.clone()
        self.stage_start_fit = self.base.fit.clone()
        self.stage_start_div = current_div
        self.stage_start_stagnation = self.stagnation.clone()
        self.stage_guardian_mask = self.current_niche_guardian.clone()
        if self.stage_guardian_mask.any():
            self.stage_guardian_start_best = torch.min(self.base.fit[self.stage_guardian_mask]).clone()
        else:
            self.stage_guardian_start_best = torch.tensor(torch.inf, device=self.device)

        self._log_stage_and_niches(current_div, niche_plan, self.last_intent_eff)
        self.stage_counter = self.stage_counter + 1

    def step(self):
        step_val = int(self.step_counter.item())

        if step_val > 0 and step_val % self.K == 0:
            old_stage_start_fit = self.stage_start_fit.clone()
            old_stage_start_div = self.stage_start_div.clone()
            old_stage_start_stagnation = self.stage_start_stagnation.clone()
            current_div = compute_diversity(self.base.pop)

            stag_ratio_start = (old_stage_start_stagnation >= 10).sum().float() / self.pop_size
            stag_ratio_end = (self.stagnation >= 10).sum().float() / self.pop_size

            feedbacks = torch.stack(
                [
                    self.last_intervene_ratio,
                    self.last_accept_ratio,
                    self.last_rescue_success,
                    self.last_elite_damage,
                ]
            ).to(self.device)

            release_count = int(self.last_release_mask.sum().item())
            if release_count > 0:
                released_pos = self.base.pop[self.last_release_mask]
                released_fit = self.base.fit[self.last_release_mask]
                birth_pos = self.last_release_birth_pop[self.last_release_mask]
                baseline_fit = self.last_release_baseline_fit[self.last_release_mask]
                improved = released_fit < baseline_fit

                span = torch.norm((self.ub - self.lb).squeeze(0)).item()
                far_th = max(1e-6, 0.05 * span / (self.dim ** 0.5))
                move_th = max(1e-6, 0.02 * span / (self.dim ** 0.5))

                if self.last_archive_before_release.numel() > 0:
                    d_arch = torch.cdist(released_pos, self.last_archive_before_release)
                    far_from_archive = torch.min(d_arch, dim=1).values > far_th
                else:
                    far_from_archive = torch.ones(released_pos.shape[0], dtype=torch.bool, device=self.device)

                d_pop = torch.cdist(released_pos, self.base.pop)
                neighbor_count = (d_pop <= far_th).sum(dim=1)
                joined_active = neighbor_count > 1
                moved = torch.norm(released_pos - birth_pos, dim=1) > move_th
                success = far_from_archive & (joined_active | improved | moved)
                release_efficiency = float(success.float().mean().item())
            else:
                release_efficiency = 0.0

            guardian_mask = self.stage_guardian_mask
            if guardian_mask.any():
                curr_guard_best = torch.min(self.base.fit[guardian_mask])
                start_guard_best = self.stage_guardian_start_best
                degrade = torch.relu(curr_guard_best - start_guard_best) / (torch.abs(start_guard_best) + 1e-8)
                guardian_stability = float(torch.clamp(1.0 - degrade, 0.0, 1.0).item())
            else:
                guardian_stability = 0.0

            niche_stats = self.last_niche_reward_stats.clone()
            niche_stats[3] = guardian_stability
            niche_stats[4] = release_efficiency

            reward = self.reward_calc.calculate(
                fit_start=old_stage_start_fit,
                fit_end=self.base.fit,
                div_start=float(old_stage_start_div.item()),
                div_end=float(current_div.item()),
                target_div=float(self.current_cmd_target_div.item()),
                initial_div=float(self.initial_div.item()),
                stagnation_start=old_stage_start_stagnation,
                stag_ratio_start=float(stag_ratio_start.item()),
                stag_ratio_end=float(stag_ratio_end.item()),
                action=self.last_intent_eff,
                feedbacks=feedbacks,
                niche_stats=niche_stats,
            )

            self.rl_agent.buffer.add(
                state=self.last_state,
                action=self.last_action_raw,
                reward=reward,
                value=float(self.last_value.item()),
                log_prob=float(self.last_logprob.item()),
                done=0.0,
            )

            if len(self.rl_agent.buffer.states) >= self.update_stages:
                self.rl_agent.update()

            self._dispatch_new_stage()

        self.base.step()

        curr_pop = self.base.pop.clone()
        curr_fit = self.base.fit.clone()

        guardian_mask = self.current_niche_guardian
        if guardian_mask.any():
            curr_pop[guardian_mask] = self.prev_pop[guardian_mask]
            curr_fit[guardian_mask] = self.prev_fit[guardian_mask]

        pop_std = torch.std(curr_pop, dim=0) + 1e-8
        trial_pop = curr_pop.clone()

        intent = IntentVector.from_tensor(self.last_intent_eff)
        middle_mask = self.current_cmd_middle
        rescue_mask = self.current_cmd_rescue
        quiet_mask = self.current_cmd_quiet
        refine_gate = float(self.current_gate_refine.item())

        if refine_gate > 0.5:
            middle_mask = middle_mask & (~quiet_mask)
            rescue_mask = rescue_mask & (~quiet_mask)

        if middle_mask.any() and intent.e_t > 0.01:
            mid_scales = self.current_cmd_scales[middle_mask].unsqueeze(1)
            noise = torch.randn_like(trial_pop[middle_mask]) * pop_std * intent.e_t * mid_scales
            small_noise = torch.norm(noise, dim=1, keepdim=True) < 1e-6
            noise = torch.where(small_noise, torch.zeros_like(noise), noise)
            trial_pop[middle_mask] += noise

        if middle_mask.any() and intent.x_t > 0.01:
            dir_to_best = self.global_best_location - trial_pop[middle_mask]
            mid_scales = self.current_cmd_scales[middle_mask].unsqueeze(1)
            pull = torch.rand(middle_mask.sum().item(), 1, device=self.device) * intent.x_t * mid_scales
            trial_pop[middle_mask] += dir_to_best * pull

        if rescue_mask.any():
            rescue_scales = self.current_cmd_scales[rescue_mask].unsqueeze(1)
            rescue_strength = 1.0 - 0.8 * refine_gate
            base_pos = curr_pop[rescue_mask]
            to_best = self.global_best_location - base_pos
            noise = torch.randn_like(base_pos) * pop_std * 0.5 * rescue_strength * rescue_scales
            trial_pop[rescue_mask] = base_pos + 0.3 * to_best + noise

        trial_pop = clamp(trial_pop, self.lb, self.ub)

        modified_mask = middle_mask | rescue_mask
        n_mod = modified_mask.sum().float()

        if n_mod > 0:
            trial_fit = curr_fit.clone()
            trial_fit[modified_mask] = self.evaluate(trial_pop[modified_mask])

            strict_accept = trial_fit < curr_fit
            margin_fit = curr_fit + torch.abs(curr_fit) * self.current_cmd_alpha_margin + self.current_cmd_beta_margin
            soft_accept = trial_fit < margin_fit
            accept_mask = strict_accept if refine_gate > 0.65 else (strict_accept | (rescue_mask & soft_accept))

            curr_pop[accept_mask] = trial_pop[accept_mask]
            curr_fit[accept_mask] = trial_fit[accept_mask]

            self.last_intervene_ratio = n_mod / self.pop_size
            self.last_accept_ratio = accept_mask.sum().float() / n_mod
            if rescue_mask.any():
                self.last_rescue_success = (accept_mask & rescue_mask).sum().float() / rescue_mask.sum().float()
            else:
                self.last_rescue_success = torch.tensor(0.0, device=self.device)
        else:
            self.last_intervene_ratio = torch.tensor(0.0, device=self.device)
            self.last_accept_ratio = torch.tensor(0.0, device=self.device)
            self.last_rescue_success = torch.tensor(0.0, device=self.device)

        protected_mask = self.current_cmd_protected
        if protected_mask.any():
            worsened = (curr_fit > self.prev_fit) & protected_mask
            self.last_elite_damage = worsened.sum().float() / protected_mask.sum().float()
            curr_pop[worsened] = self.prev_pop[worsened]
            curr_fit[worsened] = self.prev_fit[worsened]
        else:
            self.last_elite_damage = torch.tensor(0.0, device=self.device)

        self.base.pop = curr_pop
        self.base.fit = curr_fit

        improved = curr_fit < self.prev_fit
        self.stagnation = torch.where(improved, torch.zeros_like(self.stagnation), self.stagnation + 1)

        self.prev_pop = curr_pop.clone()
        self.prev_fit = curr_fit.clone()

        current_best_val, current_best_idx = torch.min(self.base.fit, dim=0)
        if current_best_val < self.global_best_fit:
            self.global_best_fit = current_best_val.clone()
            self.global_best_location = self.base.pop[current_best_idx].clone()

        if hasattr(self.base, "global_best_fit"):
            self.base.global_best_fit = self.global_best_fit.clone()
            self.base.global_best_location = self.global_best_location.clone()

        self.step_counter = self.step_counter + 1

    def flush_logs(self):
        if self.niche_logger is None or self.log_dir is None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        fid = self.func_id if self.func_id is not None else "NA"
        rid = self.run_id if self.run_id is not None else "NA"
        stage_csv = os.path.join(self.log_dir, f"F{fid}_run{rid}_stage_log.csv")
        niche_csv = os.path.join(self.log_dir, f"F{fid}_run{rid}_niche_log.csv")
        self.niche_logger.flush(stage_csv, niche_csv)
