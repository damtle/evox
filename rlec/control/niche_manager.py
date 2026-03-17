from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class NichePlan:
    exploring_mask: torch.Tensor
    refining_mask: torch.Tensor
    converged_mask: torch.Tensor
    guardian_mask: torch.Tensor
    release_mask: torch.Tensor
    rescue_mask: torch.Tensor
    cluster_ids: torch.Tensor
    active_centers: torch.Tensor
    niche_rows: List[Dict]
    n_new_niches: int
    archive_added: int


class NicheManager:
    """Niche-level lifecycle planner with stable niche IDs and archive."""

    def __init__(self, pop_size: int, dim: int, device: torch.device):
        self.pop_size = pop_size
        self.dim = dim
        self.device = device
        self.archive_centers = torch.empty((0, dim), device=device)

        self.prev_centers = torch.empty((0, dim), device=device)
        self.prev_ids: List[int] = []
        self.prev_ages: List[int] = []
        self.next_niche_id = 0

    def _cluster(self, pop: torch.Tensor, fit: torch.Tensor, eps: float) -> tuple[torch.Tensor, List[torch.Tensor]]:
        n = pop.shape[0]
        cluster_ids = torch.full((n,), -1, dtype=torch.long, device=pop.device)
        order = torch.argsort(fit)  # best-first
        cid = 0
        for idx in order:
            i = int(idx.item())
            if cluster_ids[i] != -1:
                continue
            d = torch.norm(pop - pop[i], dim=1)
            members = (d <= eps) & (cluster_ids == -1)
            cluster_ids[members] = cid
            cid += 1

        clusters: List[torch.Tensor] = []
        for k in range(cid):
            clusters.append(torch.where(cluster_ids == k)[0])
        return cluster_ids, clusters

    def _update_archive(self, centers: List[torch.Tensor], min_dist: float) -> int:
        added = 0
        for c in centers:
            if self.archive_centers.numel() == 0:
                self.archive_centers = c.unsqueeze(0)
                added += 1
                continue
            d = torch.norm(self.archive_centers - c.unsqueeze(0), dim=1)
            if torch.min(d).item() > min_dist:
                self.archive_centers = torch.cat([self.archive_centers, c.unsqueeze(0)], dim=0)
                added += 1
        return added

    def _assign_stable_ids(self, centers: List[torch.Tensor], eps: float) -> tuple[List[int], List[int], int]:
        if len(centers) == 0:
            self.prev_centers = torch.empty((0, self.dim), device=self.device)
            self.prev_ids = []
            self.prev_ages = []
            return [], [], 0

        if self.prev_centers.numel() == 0:
            ids = []
            ages = []
            for _ in centers:
                ids.append(self.next_niche_id)
                ages.append(1)
                self.next_niche_id += 1
            self.prev_centers = torch.stack(centers, dim=0)
            self.prev_ids = ids
            self.prev_ages = ages
            return ids, ages, len(ids)

        curr = torch.stack(centers, dim=0)
        d = torch.cdist(curr, self.prev_centers)
        match_dist = max(1e-6, 1.5 * eps)

        ids = [-1] * len(centers)
        ages = [1] * len(centers)
        used_prev = set()

        # Greedy nearest matching under threshold.
        flat = []
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                flat.append((float(d[i, j].item()), i, j))
        flat.sort(key=lambda x: x[0])
        for dist_ij, i, j in flat:
            if dist_ij > match_dist:
                break
            if ids[i] != -1 or j in used_prev:
                continue
            ids[i] = self.prev_ids[j]
            ages[i] = self.prev_ages[j] + 1
            used_prev.add(j)

        new_count = 0
        for i in range(len(ids)):
            if ids[i] == -1:
                ids[i] = self.next_niche_id
                ages[i] = 1
                self.next_niche_id += 1
                new_count += 1

        self.prev_centers = curr
        self.prev_ids = ids
        self.prev_ages = ages
        return ids, ages, new_count

    def plan(
        self,
        pop: torch.Tensor,
        fit: torch.Tensor,
        stage_start_pop: torch.Tensor,
        stage_start_fit: torch.Tensor,
        stage_start_stagnation: torch.Tensor,
        stagnation: torch.Tensor,
        intent_eff: torch.Tensor,
        initial_div: float,
    ) -> NichePlan:
        n = pop.shape[0]
        device = pop.device
        current_div = torch.std(pop, dim=0).mean().item()
        eps = max(1e-6, 0.5 * current_div + 0.5 * float(initial_div) * (0.08 + 0.20 * float(intent_eff[2].item())))

        cluster_ids_local, clusters = self._cluster(pop, fit, eps=eps)

        exploring = torch.zeros(n, dtype=torch.bool, device=device)
        refining = torch.zeros(n, dtype=torch.bool, device=device)
        converged = torch.zeros(n, dtype=torch.bool, device=device)
        guardian = torch.zeros(n, dtype=torch.bool, device=device)
        release = torch.zeros(n, dtype=torch.bool, device=device)
        rescue = torch.zeros(n, dtype=torch.bool, device=device)

        centers: List[torch.Tensor] = []
        converged_centers: List[torch.Tensor] = []
        active_centers: List[torch.Tensor] = []
        release_pool: List[int] = []
        rescue_pool: List[int] = []

        # Per-niche temporary metrics for logging.
        niche_tmp: List[Dict] = []

        guard_ratio = 0.05 + 0.20 * float(intent_eff[5].item())
        for local_cid, members in enumerate(clusters):
            m = members
            c_pop = pop[m]
            c_fit = fit[m]
            c_pop_start = stage_start_pop[m]
            c_fit_start = stage_start_fit[m]
            c_stag = stagnation[m]
            c_stag_start = stage_start_stagnation[m]

            center = torch.mean(c_pop, dim=0)
            centers.append(center)

            center_start = torch.mean(c_pop_start, dim=0)
            radius = torch.norm(c_pop - center.unsqueeze(0), dim=1).mean()
            radius_start = torch.norm(c_pop_start - center_start.unsqueeze(0), dim=1).mean()
            radius_rel = (radius / (initial_div + 1e-8)).item()
            radius_shrink = ((radius_start - radius) / (radius_start + 1e-8)).item()

            curr_best = torch.min(c_fit)
            start_best = torch.min(c_fit_start)
            best_improve_rel = ((start_best - curr_best) / (torch.abs(start_best) + 1e-8)).item()
            improve_ratio_stage = torch.mean((c_fit < c_fit_start).float()).item()
            stag_ratio = torch.mean((c_stag >= 10).float()).item()
            stag_growth = torch.mean((c_stag - c_stag_start).float()).item()

            is_converged = (
                radius_rel < 0.01
                and radius_shrink > 0.40
                and best_improve_rel < 5e-4
                and improve_ratio_stage < 0.20
            )
            is_refining = (not is_converged) and (
                radius_rel < 0.05
                and radius_shrink > 0.20
                and best_improve_rel < 2e-3
            )

            if is_converged:
                converged[m] = True
                converged_centers.append(center)
                k = max(1, int(len(m) * guard_ratio))
                local_order = torch.argsort(c_fit)
                g_idx = m[local_order[:k]]
                guardian[g_idx] = True
                for idx in m[local_order[k:]]:
                    release_pool.append(int(idx.item()))
                status = "converged"
            elif is_refining:
                refining[m] = True
                active_centers.append(center)
                status = "refining"
            else:
                exploring[m] = True
                active_centers.append(center)
                score = (c_stag - c_stag_start).float() + 0.5 * (c_stag >= 10).float()
                local_bad = m[torch.argsort(score, descending=True)]
                topk = max(1, int(len(m) * min(0.4, 0.2 + 0.2 * float(intent_eff[4].item()))))
                for idx in local_bad[:topk]:
                    rescue_pool.append(int(idx.item()))
                status = "exploring"

            niche_tmp.append(
                {
                    "local_cluster_id": local_cid,
                    "members": m,
                    "size": int(len(m)),
                    "status": status,
                    "center": center,
                    "center_norm": float(torch.norm(center).item()),
                    "radius": float(radius.item()),
                    "radius_rel": float(radius_rel),
                    "radius_shrink": float(radius_shrink),
                    "best_fit": float(curr_best.item()),
                    "best_improve_rel": float(best_improve_rel),
                    "accept_ratio_stage": float(improve_ratio_stage),
                    "improve_ratio_stage": float(improve_ratio_stage),
                    "stag_ratio": float(stag_ratio),
                    "stag_growth": float(stag_growth),
                }
            )

        stable_ids, stable_ages, n_new_niches = self._assign_stable_ids(centers, eps=eps)

        # Build stable per-particle cluster ids.
        cluster_ids = torch.full((n,), -1, dtype=torch.long, device=device)
        for t, row in enumerate(niche_tmp):
            cluster_ids[row["members"]] = stable_ids[t]

        # Budgeted release and rescue.
        release_budget = int(self.pop_size * float(intent_eff[3].item()) * (0.30 + 0.70 * float(intent_eff[0].item())))
        rescue_budget = int(self.pop_size * float(intent_eff[4].item()))

        if release_budget > 0 and len(release_pool) > 0:
            release_sel = release_pool[: min(release_budget, len(release_pool))]
            release[torch.tensor(release_sel, dtype=torch.long, device=device)] = True
        if rescue_budget > 0 and len(rescue_pool) > 0:
            rescue_sel = rescue_pool[: min(rescue_budget, len(rescue_pool))]
            rescue[torch.tensor(rescue_sel, dtype=torch.long, device=device)] = True

        rescue = rescue & (~guardian)
        release = release & (~guardian)

        archive_added = self._update_archive(converged_centers, min_dist=max(1e-6, 1.5 * eps))

        if len(active_centers) == 0:
            active_centers_t = torch.empty((0, self.dim), device=device)
        else:
            active_centers_t = torch.stack(active_centers, dim=0)

        niche_rows: List[Dict] = []
        for t, row in enumerate(niche_tmp):
            m = row["members"]
            center = row["center"]
            if self.archive_centers.numel() == 0:
                dist_archive_min = float("inf")
            else:
                dist_archive_min = float(torch.min(torch.norm(self.archive_centers - center.unsqueeze(0), dim=1)).item())

            release_count = int(release[m].sum().item())
            rescue_count = int(rescue[m].sum().item())
            guardian_count = int(guardian[m].sum().item())

            niche_rows.append(
                {
                    "niche_id": int(stable_ids[t]),
                    "niche_age": int(stable_ages[t]),
                    "size": int(row["size"]),
                    "status": row["status"],
                    "center_norm": row["center_norm"],
                    "radius": row["radius"],
                    "radius_rel": row["radius_rel"],
                    "radius_shrink": row["radius_shrink"],
                    "best_fit": row["best_fit"],
                    "best_improve_rel": row["best_improve_rel"],
                    "accept_ratio_stage": row["accept_ratio_stage"],
                    "stag_ratio": row["stag_ratio"],
                    "stag_growth": row["stag_growth"],
                    "guardian_count": guardian_count,
                    "release_count": release_count,
                    "rescue_count": rescue_count,
                    "dist_to_archive_min": dist_archive_min,
                }
            )

        return NichePlan(
            exploring_mask=exploring,
            refining_mask=refining,
            converged_mask=converged,
            guardian_mask=guardian,
            release_mask=release,
            rescue_mask=rescue,
            cluster_ids=cluster_ids,
            active_centers=active_centers_t,
            niche_rows=niche_rows,
            n_new_niches=n_new_niches,
            archive_added=archive_added,
        )

    def sample_release_positions(
        self,
        n: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        avoid_centers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if n <= 0:
            return torch.empty((0, self.dim), device=self.device)

        avoid = self.archive_centers
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
        min_dist = max(1e-6, 0.10 * span / (self.dim ** 0.5))

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
