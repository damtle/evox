from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class SubpopPlan:
    exploit_mask: torch.Tensor
    bridge_mask: torch.Tensor
    explore_mask: torch.Tensor
    exploit_guardian_mask: torch.Tensor
    migrate_to_exploit: torch.Tensor
    migrate_to_bridge: torch.Tensor
    migrate_to_explore: torch.Tensor
    resample_mask: torch.Tensor
    migration_counts: Dict[str, int]
    subpop_rows: List[Dict]
    archive_added: int


class SubpopulationManager:
    """Three fixed-role subpopulation controller (Exploit/Bridge/Explore)."""

    EXPLOIT = 0
    BRIDGE = 1
    EXPLORE = 2

    def __init__(
        self,
        pop_size: int,
        dim: int,
        device: torch.device,
        exploit_ratio: float = 0.20,
        bridge_ratio: float = 0.30,
        explore_ratio: float = 0.50,
        migration_period: int = 4,
        archive_dist_factor: float = 0.25,
    ):
        self.pop_size = pop_size
        self.dim = dim
        self.device = device

        total = exploit_ratio + bridge_ratio + explore_ratio
        self.exploit_ratio = exploit_ratio / max(1e-8, total)
        self.bridge_ratio = bridge_ratio / max(1e-8, total)
        self.explore_ratio = explore_ratio / max(1e-8, total)

        self.migration_period = max(1, migration_period)
        self.archive_dist_factor = archive_dist_factor

        self.subpop_ids = torch.full((pop_size,), self.EXPLORE, dtype=torch.long, device=device)
        self.subpop_age = torch.zeros(pop_size, dtype=torch.long, device=device)
        self.last_migration_stage = -1

        self.archive_points = torch.empty((0, dim), device=device)
        self.archive_scores: List[float] = []

    def init_subpops(self) -> None:
        n = self.pop_size
        n_exploit = int(round(n * self.exploit_ratio))
        n_bridge = int(round(n * self.bridge_ratio))
        n_exploit = min(max(1, n_exploit), n - 2)
        n_bridge = min(max(1, n_bridge), n - n_exploit - 1)
        n_explore = n - n_exploit - n_bridge

        perm = torch.randperm(n, device=self.device)
        ids = torch.full((n,), self.EXPLORE, dtype=torch.long, device=self.device)
        ids[perm[:n_exploit]] = self.EXPLOIT
        ids[perm[n_exploit : n_exploit + n_bridge]] = self.BRIDGE
        ids[perm[n_exploit + n_bridge : n_exploit + n_bridge + n_explore]] = self.EXPLORE

        self.subpop_ids = ids
        self.subpop_age = torch.zeros(n, dtype=torch.long, device=self.device)
        self.last_migration_stage = -1

    def _mask_of(self, subpop_id: int) -> torch.Tensor:
        return self.subpop_ids == subpop_id

    def _select_exploit_guardians(self, fit: torch.Tensor, exploit_mask: torch.Tensor, intent_p: float) -> torch.Tensor:
        guardian = torch.zeros(self.pop_size, dtype=torch.bool, device=self.device)
        idx = torch.where(exploit_mask)[0]
        if idx.numel() == 0:
            return guardian
        k = max(1, int(idx.numel() * (0.08 + 0.20 * float(intent_p))))
        local_order = torch.argsort(fit[idx])
        guardian_idx = idx[local_order[:k]]
        guardian[guardian_idx] = True
        return guardian

    def select_exploit_guardians(self, fit: torch.Tensor, intent_p: float) -> torch.Tensor:
        exploit_mask = self._mask_of(self.EXPLOIT)
        return self._select_exploit_guardians(fit, exploit_mask, intent_p)

    def _adaptive_quantile(self, values: torch.Tensor, q: float) -> float:
        if values.numel() == 0:
            return float("inf")
        q = min(0.95, max(0.05, q))
        return float(torch.quantile(values, q).item())

    def _collect_group_row(
        self,
        name: str,
        mask: torch.Tensor,
        pop: torch.Tensor,
        fit: torch.Tensor,
        stagnation: torch.Tensor,
        improved: torch.Tensor,
        migrated_in: int,
        migrated_out: int,
    ) -> Dict:
        size = int(mask.sum().item())
        if size == 0:
            return {
                "subpop": name,
                "size": 0,
                "best_fit": float("inf"),
                "mean_fit": float("inf"),
                "div": 0.0,
                "stagnation_ratio": 0.0,
                "accepted_ratio": 0.0,
                "migrated_in": migrated_in,
                "migrated_out": migrated_out,
            }

        f = fit[mask]
        p = pop[mask]
        s = stagnation[mask]
        imp = improved[mask]
        div = float(torch.std(p, dim=0).mean().item()) if size > 1 else 0.0
        return {
            "subpop": name,
            "size": size,
            "best_fit": float(torch.min(f).item()),
            "mean_fit": float(torch.mean(f).item()),
            "div": div,
            "stagnation_ratio": float((s >= 10).float().mean().item()),
            "accepted_ratio": float(imp.float().mean().item()),
            "migrated_in": migrated_in,
            "migrated_out": migrated_out,
        }

    def _update_archive(self, points: torch.Tensor, scores: torch.Tensor, min_dist: float) -> int:
        if points.numel() == 0:
            return 0
        added = 0
        for i in range(points.shape[0]):
            c = points[i]
            score = float(scores[i].item())
            if self.archive_points.numel() == 0:
                self.archive_points = c.unsqueeze(0)
                self.archive_scores.append(score)
                added += 1
                continue
            d = torch.norm(self.archive_points - c.unsqueeze(0), dim=1)
            if float(torch.min(d).item()) > min_dist:
                self.archive_points = torch.cat([self.archive_points, c.unsqueeze(0)], dim=0)
                self.archive_scores.append(score)
                added += 1
        return added

    def plan_stage(
        self,
        pop: torch.Tensor,
        fit: torch.Tensor,
        prev_fit: torch.Tensor,
        stagnation: torch.Tensor,
        stage_id: int,
        intent_eff: torch.Tensor,
    ) -> SubpopPlan:
        exploit_mask = self._mask_of(self.EXPLOIT)
        bridge_mask = self._mask_of(self.BRIDGE)
        explore_mask = self._mask_of(self.EXPLORE)

        improved = fit < prev_fit
        exploit_guardian = self._select_exploit_guardians(fit, exploit_mask, float(intent_eff[5].item()))

        migrate_to_exploit = torch.zeros(self.pop_size, dtype=torch.bool, device=self.device)
        migrate_to_bridge = torch.zeros(self.pop_size, dtype=torch.bool, device=self.device)
        migrate_to_explore = torch.zeros(self.pop_size, dtype=torch.bool, device=self.device)

        do_migration = stage_id > 0 and (stage_id % self.migration_period == 0)
        if do_migration:
            max_migrate = max(1, int(self.pop_size * (0.05 + 0.20 * float(intent_eff[3].item()))))

            # Explore -> Bridge
            e_idx = torch.where(explore_mask)[0]
            e_good = torch.zeros_like(e_idx, dtype=torch.bool)
            if e_idx.numel() > 0:
                e_fit = fit[e_idx]
                e_q = self._adaptive_quantile(e_fit, 0.35)
                e_good = (e_fit <= e_q) & improved[e_idx] & (stagnation[e_idx] < 12)
            e_promote = e_idx[e_good]
            if e_promote.numel() > max_migrate:
                order = torch.argsort(fit[e_promote])
                e_promote = e_promote[order[:max_migrate]]
            migrate_to_bridge[e_promote] = True

            # Bridge -> Exploit
            b_idx = torch.where(bridge_mask)[0]
            b_good = torch.zeros_like(b_idx, dtype=torch.bool)
            if b_idx.numel() > 0:
                b_fit = fit[b_idx]
                b_q = self._adaptive_quantile(b_fit, 0.30)
                b_good = (b_fit <= b_q) & (stagnation[b_idx] < 10)
            b_promote = b_idx[b_good]
            if b_promote.numel() > max_migrate:
                order = torch.argsort(fit[b_promote])
                b_promote = b_promote[order[:max_migrate]]
            migrate_to_exploit[b_promote] = True

            # Exploit -> Explore (non-guardian stagnated)
            x_idx = torch.where(exploit_mask & (~exploit_guardian) & (stagnation >= 12))[0]
            if x_idx.numel() > max_migrate:
                order = torch.argsort(stagnation[x_idx], descending=True)
                x_idx = x_idx[order[:max_migrate]]
            migrate_to_explore[x_idx] = True

        # Explore resample candidates: migrated to Explore + long stagnation in Explore.
        explore_mask_now = explore_mask | migrate_to_explore
        bad_explore = explore_mask_now & (stagnation >= 20) & (~migrate_to_bridge)
        max_resample = max(1, int(self.pop_size * (0.05 + 0.20 * float(intent_eff[2].item()))))
        resample_mask = migrate_to_explore.clone()
        bad_idx = torch.where(bad_explore)[0]
        if bad_idx.numel() > 0:
            if bad_idx.numel() > max_resample:
                order = torch.argsort(stagnation[bad_idx], descending=True)
                bad_idx = bad_idx[order[:max_resample]]
            resample_mask[bad_idx] = True

        # Archive from high-quality explore/bridge points.
        explore_bridge_mask = (explore_mask | bridge_mask)
        eb_idx = torch.where(explore_bridge_mask)[0]
        archive_added = 0
        if eb_idx.numel() > 0:
            k = min(max(1, int(0.05 * eb_idx.numel())), eb_idx.numel())
            order = torch.argsort(fit[eb_idx])
            cand_idx = eb_idx[order[:k]]
            pop_scale = float(torch.std(pop, dim=0).mean().item())
            min_dist = max(1e-6, self.archive_dist_factor * pop_scale)
            archive_added = self._update_archive(pop[cand_idx], fit[cand_idx], min_dist=min_dist)

        migration_counts = {
            "e2b": int(migrate_to_bridge.sum().item()),
            "b2x": int(migrate_to_exploit.sum().item()),
            "x2e": int(migrate_to_explore.sum().item()),
            "resampled": int(resample_mask.sum().item()),
        }

        subpop_rows = [
            self._collect_group_row(
                "exploit",
                exploit_mask,
                pop,
                fit,
                stagnation,
                improved,
                migrated_in=migration_counts["b2x"],
                migrated_out=migration_counts["x2e"],
            ),
            self._collect_group_row(
                "bridge",
                bridge_mask,
                pop,
                fit,
                stagnation,
                improved,
                migrated_in=migration_counts["e2b"],
                migrated_out=migration_counts["b2x"],
            ),
            self._collect_group_row(
                "explore",
                explore_mask,
                pop,
                fit,
                stagnation,
                improved,
                migrated_in=migration_counts["x2e"],
                migrated_out=migration_counts["e2b"],
            ),
        ]

        return SubpopPlan(
            exploit_mask=exploit_mask,
            bridge_mask=bridge_mask,
            explore_mask=explore_mask,
            exploit_guardian_mask=exploit_guardian,
            migrate_to_exploit=migrate_to_exploit,
            migrate_to_bridge=migrate_to_bridge,
            migrate_to_explore=migrate_to_explore,
            resample_mask=resample_mask,
            migration_counts=migration_counts,
            subpop_rows=subpop_rows,
            archive_added=archive_added,
        )

    def build_subpop_rows(
        self,
        pop: torch.Tensor,
        fit: torch.Tensor,
        prev_fit: torch.Tensor,
        stagnation: torch.Tensor,
        migration_counts: Dict[str, int],
    ) -> List[Dict]:
        improved = fit < prev_fit
        exploit_mask = self._mask_of(self.EXPLOIT)
        bridge_mask = self._mask_of(self.BRIDGE)
        explore_mask = self._mask_of(self.EXPLORE)
        return [
            self._collect_group_row(
                "exploit",
                exploit_mask,
                pop,
                fit,
                stagnation,
                improved,
                migrated_in=int(migration_counts.get("b2x", 0)),
                migrated_out=int(migration_counts.get("x2e", 0)),
            ),
            self._collect_group_row(
                "bridge",
                bridge_mask,
                pop,
                fit,
                stagnation,
                improved,
                migrated_in=int(migration_counts.get("e2b", 0)),
                migrated_out=int(migration_counts.get("b2x", 0)),
            ),
            self._collect_group_row(
                "explore",
                explore_mask,
                pop,
                fit,
                stagnation,
                improved,
                migrated_in=int(migration_counts.get("x2e", 0)),
                migrated_out=int(migration_counts.get("e2b", 0)),
            ),
        ]

    def apply_migration(self, plan: SubpopPlan, stage_id: int) -> None:
        moved = torch.zeros(self.pop_size, dtype=torch.bool, device=self.device)

        if plan.migrate_to_exploit.any():
            self.subpop_ids[plan.migrate_to_exploit] = self.EXPLOIT
            moved |= plan.migrate_to_exploit
        if plan.migrate_to_bridge.any():
            self.subpop_ids[plan.migrate_to_bridge] = self.BRIDGE
            moved |= plan.migrate_to_bridge
        if plan.migrate_to_explore.any():
            self.subpop_ids[plan.migrate_to_explore] = self.EXPLORE
            moved |= plan.migrate_to_explore

        self.subpop_age = self.subpop_age + 1
        self.subpop_age[moved] = 0

        if moved.any():
            self.last_migration_stage = int(stage_id)

    def sample_explore_positions(
        self,
        n: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        avoid_centers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if n <= 0:
            return torch.empty((0, self.dim), device=self.device)

        avoid = self.archive_points
        if avoid_centers is not None and avoid_centers.numel() > 0:
            if avoid.numel() == 0:
                avoid = avoid_centers
            else:
                avoid = torch.cat([avoid, avoid_centers], dim=0)

        if avoid.numel() == 0:
            return lb + torch.rand((n, self.dim), device=self.device) * (ub - lb)

        low = lb
        high = ub
        span = torch.norm((ub - lb).squeeze(0)).item()
        min_dist = max(1e-6, 0.08 * span / (self.dim ** 0.5))

        out = []
        needed = n
        attempts = 0
        while needed > 0 and attempts < 20:
            cand = low + torch.rand((max(needed * 2, 8), self.dim), device=self.device) * (high - low)
            d = torch.cdist(cand, avoid)
            keep = torch.min(d, dim=1).values > min_dist
            kept = cand[keep]
            take = min(needed, kept.shape[0])
            if take > 0:
                out.append(kept[:take])
                needed -= take
            attempts += 1

        if needed > 0:
            out.append(low + torch.rand((needed, self.dim), device=self.device) * (high - low))
        return torch.cat(out, dim=0)
