from __future__ import annotations

import os
from typing import Optional

import torch

from evox.core import Algorithm, Mutable
from evox.utils import clamp

from rlec.control.intent_vector import IntentVector
from rlec.control.interpreter import ControlInterpreter
from rlec.control.subpopulation_manager import SubpopulationManager
from rlec.features.macro_state import MacroStateBuilder
from rlec.features.stage_reward import StageRewardCalculator
from rlec.rl.ppo import PPO
from rlec.utils.niche_logger import NicheLogger
from rlec.utils.population_metrics import compute_diversity


class RLECWrapper(Algorithm):
    """RLEC wrapper with global 6D intent + three-role subpopulation control."""

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
        self.subpop_manager = SubpopulationManager(
            pop_size=self.pop_size,
            dim=self.dim,
            device=self.device,
            exploit_ratio=0.20,
            bridge_ratio=0.30,
            explore_ratio=0.50,
            migration_period=4,
        )

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

        self.current_subpop_exploit = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_subpop_bridge = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_subpop_explore = Mutable(torch.ones(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_exploit_guardian = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))

        self.current_migrate_to_exploit = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_migrate_to_bridge = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.current_migrate_to_explore = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))

        self.current_gate_refine = Mutable(torch.tensor(0.0, device=self.device))
        self.current_gate_recover = Mutable(torch.tensor(0.0, device=self.device))

        self.global_best_fit = Mutable(torch.tensor(torch.inf, device=self.device))
        self.global_best_location = Mutable(torch.zeros(self.dim, device=self.device))

        self.last_intervene_ratio = Mutable(torch.tensor(0.0, device=self.device))
        self.last_accept_ratio = Mutable(torch.tensor(0.0, device=self.device))
        self.last_rescue_success = Mutable(torch.tensor(0.0, device=self.device))
        self.last_elite_damage = Mutable(torch.tensor(0.0, device=self.device))

        self.subpop_summary_vec = Mutable(torch.zeros(12, device=self.device))
        self.last_subpop_reward_stats = Mutable(torch.zeros(5, device=self.device))

        self.last_n_exploit = Mutable(torch.tensor(0.0, device=self.device))
        self.last_n_bridge = Mutable(torch.tensor(0.0, device=self.device))
        self.last_n_explore = Mutable(torch.tensor(0.0, device=self.device))

        self.last_migrate_counts = Mutable(torch.zeros(4, device=self.device))  # e2b, b2x, x2e, resampled

        self.last_resample_mask = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.last_resample_baseline_fit = Mutable(torch.zeros(self.pop_size, device=self.device))

        self.stage_exploit_guardian_mask = Mutable(torch.zeros(self.pop_size, dtype=torch.bool, device=self.device))
        self.stage_exploit_guardian_start_best = Mutable(torch.tensor(torch.inf, device=self.device))

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

        self.subpop_manager.init_subpops()

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

    def _build_subpop_summary(self, subpop_rows: list[dict], fit: torch.Tensor, migration_counts: dict) -> torch.Tensor:
        rows = {row["subpop"]: row for row in subpop_rows}
        exploit_row = rows.get("exploit", {"size": 0, "best_fit": float("inf"), "div": 0.0, "stagnation_ratio": 0.0})
        bridge_row = rows.get("bridge", {"size": 0, "best_fit": float("inf"), "div": 0.0, "stagnation_ratio": 0.0})
        explore_row = rows.get("explore", {"size": 0, "best_fit": float("inf"), "div": 0.0, "stagnation_ratio": 0.0})

        n_exploit = int(exploit_row["size"])
        n_bridge = int(bridge_row["size"])
        n_explore = int(explore_row["size"])
        total = max(1, self.pop_size)

        best_global = float(torch.min(fit).item())
        denom = max(abs(best_global), 1e-8)

        def _gap(v: float) -> float:
            if not torch.isfinite(torch.tensor(v)):
                return 1.0
            return min(1.0, max(0.0, (float(v) - best_global) / denom))

        exploit_ratio = n_exploit / total
        bridge_ratio = n_bridge / total
        explore_ratio = n_explore / total

        migration_total = migration_counts.get("e2b", 0) + migration_counts.get("b2x", 0) + migration_counts.get("x2e", 0)
        migration_rate = migration_total / total

        balance_penalty = float(torch.std(torch.tensor([exploit_ratio, bridge_ratio, explore_ratio], device=self.device)).item())

        summary = torch.tensor(
            [
                exploit_ratio,
                bridge_ratio,
                explore_ratio,
                _gap(exploit_row.get("best_fit", float("inf"))),
                _gap(bridge_row.get("best_fit", float("inf"))),
                _gap(explore_row.get("best_fit", float("inf"))),
                float(exploit_row.get("div", 0.0)) / (float(self.initial_div.item()) + 1e-8),
                float(bridge_row.get("div", 0.0)) / (float(self.initial_div.item()) + 1e-8),
                float(explore_row.get("div", 0.0)) / (float(self.initial_div.item()) + 1e-8),
                migration_rate,
                float(explore_row.get("stagnation_ratio", 0.0)),
                balance_penalty,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        self.last_n_exploit = torch.tensor(float(n_exploit), device=self.device)
        self.last_n_bridge = torch.tensor(float(n_bridge), device=self.device)
        self.last_n_explore = torch.tensor(float(n_explore), device=self.device)

        return summary

    def _log_stage_and_subpops(self, current_div: torch.Tensor, subpop_rows: list[dict], intent_eff: torch.Tensor):
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
            "n_exploit": int(self.last_n_exploit.item()),
            "n_bridge": int(self.last_n_bridge.item()),
            "n_explore": int(self.last_n_explore.item()),
            "migrate_e2b": int(self.last_migrate_counts[0].item()),
            "migrate_b2x": int(self.last_migrate_counts[1].item()),
            "migrate_x2e": int(self.last_migrate_counts[2].item()),
            "resampled": int(self.last_migrate_counts[3].item()),
            "exploit_ratio": float(self.subpop_summary_vec[0].item()),
            "bridge_ratio": float(self.subpop_summary_vec[1].item()),
            "explore_ratio": float(self.subpop_summary_vec[2].item()),
            "intervene_ratio": float(self.last_intervene_ratio.item()),
            "accept_ratio": float(self.last_accept_ratio.item()),
            "rescue_success": float(self.last_rescue_success.item()),
            "elite_damage": float(self.last_elite_damage.item()),
            "exploit_stability": float(self.last_subpop_reward_stats[0].item()),
            "explore_birth_efficiency": float(self.last_subpop_reward_stats[1].item()),
            "bridge_transfer_efficiency": float(self.last_subpop_reward_stats[2].item()),
            "migration_efficiency": float(self.last_subpop_reward_stats[3].item()),
            "subpop_balance_penalty": float(self.last_subpop_reward_stats[4].item()),
        }
        self.niche_logger.log_stage(stage_row)

        rows_out = []
        for row in subpop_rows:
            rows_out.append(
                {
                    "func_id": self.func_id,
                    "run_id": self.run_id,
                    "gen": gen,
                    "stage": stage_id,
                    "subpop": row["subpop"],
                    "size": row["size"],
                    "best_fit": row["best_fit"],
                    "mean_fit": row["mean_fit"],
                    "div": row["div"],
                    "stagnation_ratio": row["stagnation_ratio"],
                    "accepted_ratio": row["accepted_ratio"],
                    "migrated_in": row["migrated_in"],
                    "migrated_out": row["migrated_out"],
                }
            )
        self.niche_logger.log_subpops(rows_out)

    def _dispatch_new_stage(self):
        current_pop = self.base.pop
        current_fit = self.base.fit
        current_div = compute_diversity(current_pop)

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
            subpop_summary=self.subpop_summary_vec,
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

        e_eff = torch.clamp(intent_tensor[0] * (1.0 - 0.70 * g_refine) + 0.50 * g_recover * (1.0 - intent_tensor[0]), 0.0, 1.0)
        x_eff = torch.clamp(intent_tensor[1] + 0.20 * g_refine * (1.0 - intent_tensor[1]), 0.0, 1.0)
        d_eff = torch.clamp(intent_tensor[2] * (1.0 - 0.40 * g_refine) + 0.30 * g_recover * (1.0 - intent_tensor[2]), 0.0, 1.0)
        b_eff = torch.clamp(intent_tensor[3] * (1.0 - 0.60 * g_refine) + 0.35 * g_recover * (1.0 - intent_tensor[3]), 0.0, 1.0)
        r_eff = torch.clamp(intent_tensor[4] * (1.0 - 0.70 * g_refine) + 0.25 * g_recover * (1.0 - intent_tensor[4]), 0.0, 1.0)
        p_eff = torch.clamp(intent_tensor[5] + 0.25 * g_refine * (1.0 - intent_tensor[5]), 0.0, 1.0)

        intent_eff_tensor = torch.stack([e_eff, x_eff, d_eff, b_eff, r_eff, p_eff]).to(self.device)
        intent_eff_obj = IntentVector.from_tensor(intent_eff_tensor)

        self.current_gate_refine = g_refine.detach()
        self.current_gate_recover = g_recover.detach()

        self.last_action_raw = torch.tensor(action_raw, device=self.device, dtype=torch.float32)
        self.last_intent = intent_tensor.clone()
        self.last_intent_eff = intent_eff_tensor.clone()
        self.last_logprob = torch.tensor(log_prob, device=self.device, dtype=torch.float32)
        self.last_value = torch.tensor(value, device=self.device, dtype=torch.float32)

        stage_id = int(self.stage_counter.item())
        subpop_plan = self.subpop_manager.plan_stage(
            pop=current_pop,
            fit=current_fit,
            prev_fit=self.prev_fit,
            stagnation=self.stagnation,
            stage_id=stage_id,
            intent_eff=intent_eff_tensor,
        )
        self.subpop_manager.apply_migration(subpop_plan, stage_id=stage_id)

        ids = self.subpop_manager.subpop_ids
        exploit_mask = ids == SubpopulationManager.EXPLOIT
        bridge_mask = ids == SubpopulationManager.BRIDGE
        explore_mask = ids == SubpopulationManager.EXPLORE
        exploit_guardian_mask = self.subpop_manager.select_exploit_guardians(
            fit=current_fit,
            intent_p=float(intent_eff_tensor[5].item()),
        )

        cmd = self.interpreter.interpret(
            intent_eff_obj,
            current_fit,
            self.stagnation,
            float(self.initial_div.item()),
            subpop_roles={
                "exploit_mask": exploit_mask,
                "bridge_mask": bridge_mask,
                "explore_mask": explore_mask,
                "exploit_guardian_mask": exploit_guardian_mask,
            },
        )

        self.current_cmd_protected = cmd.protected_mask.clone()
        self.current_cmd_middle = cmd.middle_mask.clone()
        self.current_cmd_quiet = cmd.quiet_mask.clone()
        self.current_cmd_rescue = cmd.rescue_mask.clone()
        self.current_cmd_scales = cmd.step_scales.clone()
        self.current_cmd_alpha_margin = torch.tensor(cmd.alpha_margin, device=self.device)
        self.current_cmd_beta_margin = torch.tensor(cmd.beta_margin, device=self.device)
        self.current_cmd_target_div = torch.tensor(cmd.target_diversity, device=self.device)

        self.current_subpop_exploit = exploit_mask.clone()
        self.current_subpop_bridge = bridge_mask.clone()
        self.current_subpop_explore = explore_mask.clone()
        self.current_exploit_guardian = exploit_guardian_mask.clone()

        self.current_migrate_to_exploit = subpop_plan.migrate_to_exploit.clone()
        self.current_migrate_to_bridge = subpop_plan.migrate_to_bridge.clone()
        self.current_migrate_to_explore = subpop_plan.migrate_to_explore.clone()

        # Role constraints
        self.current_cmd_rescue[exploit_mask] = False
        self.current_cmd_scales[exploit_mask] = self.current_cmd_scales[exploit_mask] * 0.35
        self.current_cmd_scales[bridge_mask] = self.current_cmd_scales[bridge_mask] * 0.85
        self.current_cmd_scales[explore_mask] = self.current_cmd_scales[explore_mask] * 1.25

        self.current_cmd_protected = self.current_cmd_protected | exploit_guardian_mask
        self.current_cmd_quiet = self.current_cmd_quiet & (~explore_mask)
        self.current_cmd_quiet[exploit_guardian_mask] = True

        self.current_cmd_alpha_margin = self.current_cmd_alpha_margin * (1.0 - 0.3 * g_refine)
        self.current_cmd_beta_margin = self.current_cmd_beta_margin * (1.0 - 0.3 * g_refine)

        # Explore resampling.
        resample_idx = torch.where(subpop_plan.resample_mask & explore_mask)[0]
        self.last_resample_mask = torch.zeros(self.pop_size, dtype=torch.bool, device=self.device)
        self.last_resample_baseline_fit = torch.zeros(self.pop_size, device=self.device)
        if resample_idx.numel() > 0:
            avoid_parts = []
            xb_mask = exploit_mask | bridge_mask
            if xb_mask.any():
                avoid_parts.append(current_pop[xb_mask])
            ex_idx = torch.where(explore_mask)[0]
            if ex_idx.numel() > 0:
                k_explore = max(1, int(0.10 * ex_idx.numel()))
                order_ex = torch.argsort(current_fit[ex_idx])
                elite_explore = ex_idx[order_ex[:k_explore]]
                avoid_parts.append(current_pop[elite_explore])
            avoid_parts.append(self.global_best_location.unsqueeze(0))
            avoid_points = torch.cat(avoid_parts, dim=0) if len(avoid_parts) > 0 else None
            new_pos = self.subpop_manager.sample_explore_positions(
                int(resample_idx.numel()),
                self.lb,
                self.ub,
                avoid_centers=avoid_points,
            )
            self.base.pop[resample_idx] = new_pos
            self.base.fit[resample_idx] = self.evaluate(new_pos)
            self.stagnation[resample_idx] = 0.0

            self.prev_pop[resample_idx] = self.base.pop[resample_idx]
            self.prev_fit[resample_idx] = self.base.fit[resample_idx]

            self.current_cmd_quiet[resample_idx] = False
            self.current_cmd_protected[resample_idx] = False
            self.current_cmd_rescue[resample_idx] = False
            self.current_cmd_middle[resample_idx] = False
            self.current_cmd_scales[resample_idx] = 1.0

            self.last_resample_mask[resample_idx] = True
            self.last_resample_baseline_fit[resample_idx] = self.base.fit[resample_idx]
            current_div = compute_diversity(self.base.pop)

        self.last_migrate_counts = torch.tensor(
            [
                float(subpop_plan.migration_counts.get("e2b", 0)),
                float(subpop_plan.migration_counts.get("b2x", 0)),
                float(subpop_plan.migration_counts.get("x2e", 0)),
                float(subpop_plan.migration_counts.get("resampled", 0)),
            ],
            device=self.device,
            dtype=torch.float32,
        )

        subpop_rows_post = self.subpop_manager.build_subpop_rows(
            pop=self.base.pop,
            fit=self.base.fit,
            prev_fit=self.prev_fit,
            stagnation=self.stagnation,
            migration_counts=subpop_plan.migration_counts,
        )
        self.subpop_summary_vec = self._build_subpop_summary(subpop_rows_post, self.base.fit, subpop_plan.migration_counts)

        # Stage-level placeholders for reward terms; finalized at stage end.
        self.last_subpop_reward_stats = torch.tensor(
            [
                0.0,
                0.0,
                float(self.last_migrate_counts[1].item()) / max(1.0, float(self.pop_size)),
                float(torch.sum(self.last_migrate_counts[:3]).item()) / max(1.0, float(self.pop_size)),
                float(self.subpop_summary_vec[11].item()),
            ],
            device=self.device,
            dtype=torch.float32,
        )

        self.stage_start_pop = self.base.pop.clone()
        self.stage_start_fit = self.base.fit.clone()
        self.stage_start_div = current_div
        self.stage_start_stagnation = self.stagnation.clone()

        self.stage_exploit_guardian_mask = self.current_exploit_guardian.clone()
        if self.stage_exploit_guardian_mask.any():
            self.stage_exploit_guardian_start_best = torch.min(self.base.fit[self.stage_exploit_guardian_mask]).clone()
        else:
            self.stage_exploit_guardian_start_best = torch.tensor(torch.inf, device=self.device)

        self._log_stage_and_subpops(current_div, subpop_rows_post, self.last_intent_eff)
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

            if self.stage_exploit_guardian_mask.any():
                curr_guard_best = torch.min(self.base.fit[self.stage_exploit_guardian_mask])
                start_guard_best = self.stage_exploit_guardian_start_best
                degrade = torch.relu(curr_guard_best - start_guard_best) / (torch.abs(start_guard_best) + 1e-8)
                exploit_stability = float(torch.clamp(1.0 - degrade, 0.0, 1.0).item())
            else:
                exploit_stability = 0.0

            resample_count = int(self.last_resample_mask.sum().item())
            if resample_count > 0:
                bridge_now = self.subpop_manager.subpop_ids == SubpopulationManager.BRIDGE
                explore_birth_efficiency = float(bridge_now[self.last_resample_mask].float().mean().item())
            else:
                explore_birth_efficiency = 0.0

            e2b = float(self.last_migrate_counts[0].item())
            b2x = float(self.last_migrate_counts[1].item())
            x2e = float(self.last_migrate_counts[2].item())
            moved = max(1.0, e2b + b2x + x2e)
            bridge_transfer_efficiency = b2x / moved
            migration_efficiency = (e2b + b2x) / moved
            subpop_balance_penalty = float(torch.clamp(self.subpop_summary_vec[11], 0.0, 1.0).item())

            subpop_stats = self.last_subpop_reward_stats.clone()
            subpop_stats[0] = exploit_stability
            subpop_stats[1] = explore_birth_efficiency
            subpop_stats[2] = bridge_transfer_efficiency
            subpop_stats[3] = migration_efficiency
            subpop_stats[4] = subpop_balance_penalty
            self.last_subpop_reward_stats = subpop_stats.clone()

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
                subpop_stats=subpop_stats,
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

        exploit_guardian = self.current_exploit_guardian
        if exploit_guardian.any():
            curr_pop[exploit_guardian] = self.prev_pop[exploit_guardian]
            curr_fit[exploit_guardian] = self.prev_fit[exploit_guardian]

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
            trial_pop[middle_mask] += noise

        if middle_mask.any() and intent.x_t > 0.01:
            dir_to_best = self.global_best_location - trial_pop[middle_mask]
            mid_scales = self.current_cmd_scales[middle_mask].unsqueeze(1)
            pull = torch.rand(middle_mask.sum().item(), 1, device=self.device) * intent.x_t * mid_scales
            trial_pop[middle_mask] += dir_to_best * pull

        if rescue_mask.any():
            rescue_scales = self.current_cmd_scales[rescue_mask].unsqueeze(1)
            rescue_strength = 1.0 - 0.6 * refine_gate
            base_pos = curr_pop[rescue_mask]
            to_best = self.global_best_location - base_pos
            noise = torch.randn_like(base_pos) * pop_std * 0.5 * rescue_strength * rescue_scales
            trial_pop[rescue_mask] = base_pos + 0.2 * to_best + noise

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
        subpop_csv = os.path.join(self.log_dir, f"F{fid}_run{rid}_subpop_log.csv")
        self.niche_logger.flush(stage_csv, subpop_csv)
