from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class NichePlan:
    exploring_mask: torch.Tensor
    refining_mask: torch.Tensor
    completed_mask: torch.Tensor
    anchor_mask: torch.Tensor
    scout_mask: torch.Tensor
    guardian_mask: torch.Tensor
    release_mask: torch.Tensor
    rescue_mask: torch.Tensor
    cluster_ids: torch.Tensor
    overlap_group_ids: torch.Tensor
    active_centers: torch.Tensor
    anchor_centers: torch.Tensor
    completed_centers: torch.Tensor
    niche_rows: List[Dict]
    n_new_niches: int
    archive_added: int


class NicheManager:
    """Niche-level lifecycle planner with stable IDs, completion patience, overlap merge, and scout release."""

    def __init__(
        self,
        pop_size: int,
        dim: int,
        device: torch.device,
        r_small: float = 0.01,
        r_shrink: float = 0.40,
        improve_eps: float = 5e-4,
        improve_ratio_eps: float = 0.20,
        complete_patience: int = 2,
        merge_dist: float = 1.2,
        merge_fit_eps: float = 1e-2,
        max_anchor_niches: int = 2,
        anchor_scale: float = 0.20,
        release_exploring_stag: float = 16.0,
        completed_guardian_cap: int = 2,
    ):
        self.pop_size = pop_size
        self.dim = dim
        self.device = device
        self.r_small = r_small
        self.r_shrink = r_shrink
        self.improve_eps = improve_eps
        self.improve_ratio_eps = improve_ratio_eps
        self.complete_patience = complete_patience
        self.merge_dist = merge_dist
        self.merge_fit_eps = merge_fit_eps
        self.max_anchor_niches = max_anchor_niches
        self.anchor_scale = anchor_scale
        self.release_exploring_stag = release_exploring_stag
        self.completed_guardian_cap = completed_guardian_cap

        self.archive_centers = torch.empty((0, dim), device=device)
        self.archive_best_fit: List[float] = []
        self.archive_age: List[int] = []

        self.prev_centers = torch.empty((0, dim), device=device)
        self.prev_ids: List[int] = []
        self.prev_ages: List[int] = []
        self.prev_status_by_id: Dict[int, str] = {}
        self.prev_complete_counter: Dict[int, int] = {}
        self.next_niche_id = 0

    def _cluster(self, pop: torch.Tensor, fit: torch.Tensor, eps: float) -> Tuple[torch.Tensor, List[torch.Tensor]]:
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

    def _update_archive(
        self,
        centers: List[torch.Tensor],
        best_fits: List[float],
        ages: List[int],
        min_dist: float,
    ) -> int:
        added = 0
        for idx, c in enumerate(centers):
            if self.archive_centers.numel() == 0:
                self.archive_centers = c.unsqueeze(0)
                self.archive_best_fit.append(float(best_fits[idx]))
                self.archive_age.append(int(ages[idx]))
                added += 1
                continue
            d = torch.norm(self.archive_centers - c.unsqueeze(0), dim=1)
            if torch.min(d).item() > min_dist:
                self.archive_centers = torch.cat([self.archive_centers, c.unsqueeze(0)], dim=0)
                self.archive_best_fit.append(float(best_fits[idx]))
                self.archive_age.append(int(ages[idx]))
                added += 1
        return added

    def _assign_stable_ids(self, centers: List[torch.Tensor], eps: float) -> Tuple[List[int], List[int], int]:
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

    def _detect_overlap_groups(
        self,
        centers: torch.Tensor,
        best_fit: torch.Tensor,
        merge_dist: float,
        merge_fit_eps: float,
    ) -> torch.Tensor:
        if centers.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        n = centers.shape[0]
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        dist_mat = torch.cdist(centers, centers)
        for i in range(n):
            for j in range(i + 1, n):
                cond_a = float(dist_mat[i, j].item()) < merge_dist
                denom = max(abs(float(best_fit[i].item())), abs(float(best_fit[j].item())), 1e-8)
                fit_rel = abs(float(best_fit[i].item()) - float(best_fit[j].item())) / denom
                cond_b = fit_rel < merge_fit_eps
                if cond_a and cond_b:
                    union(i, j)

        root_to_gid: Dict[int, int] = {}
        next_gid = 0
        group_ids = torch.full((n,), -1, dtype=torch.long, device=self.device)
        for i in range(n):
            r = find(i)
            if r not in root_to_gid:
                root_to_gid[r] = next_gid
                next_gid += 1
            group_ids[i] = root_to_gid[r]
        return group_ids

    def _update_completion_counter(
        self,
        stable_ids: List[int],
        complete_signal: List[bool],
        patience: int,
    ) -> Tuple[List[bool], List[int]]:
        completed: List[bool] = []
        counters: List[int] = []
        alive_ids = set(stable_ids)

        for sid, signal in zip(stable_ids, complete_signal):
            prev = self.prev_complete_counter.get(sid, 0)
            now = prev + 1 if signal else 0
            self.prev_complete_counter[sid] = now
            counters.append(now)
            completed.append(now >= patience)

        stale_ids = [k for k in self.prev_complete_counter.keys() if k not in alive_ids]
        for sid in stale_ids:
            self.prev_complete_counter.pop(sid, None)

        return completed, counters

    def _select_anchor_niches(self, rows: List[Dict], max_anchor_niches: int) -> List[int]:
        candidates = [
            i
            for i, row in enumerate(rows)
            if row["status"] in ("refining", "completed") and not row.get("is_overlap_redundant", False)
        ]
        if not candidates:
            return []

        ranked = sorted(
            candidates,
            key=lambda i: (
                rows[i]["best_fit"],
                -rows[i]["niche_age"],
                rows[i]["radius_rel"],
                -rows[i]["best_improve_rel"],
            ),
        )
        return ranked[: max(1, max_anchor_niches)]

    def _build_scout_release_pool(self, rows: List[Dict], per_niche_guardians: List[torch.Tensor]) -> List[int]:
        pool: List[int] = []
        for i, row in enumerate(rows):
            members: torch.Tensor = row["members"]
            guardian_ids: torch.Tensor = per_niche_guardians[i]
            is_guardian = torch.zeros((members.shape[0],), dtype=torch.bool, device=members.device)
            if guardian_ids.numel() > 0:
                is_guardian = (members.unsqueeze(1) == guardian_ids.unsqueeze(0)).any(dim=1)

            non_guard = members[~is_guardian]
            if row["status"] == "completed":
                pool.extend([int(x.item()) for x in non_guard])
                continue
            if row.get("is_overlap_redundant", False):
                pool.extend([int(x.item()) for x in non_guard])
                continue
            if row["status"] == "exploring" and row["stag_ratio"] > 0.6 and row["stag_growth"] > self.release_exploring_stag:
                pool.extend([int(x.item()) for x in non_guard])

        seen = set()
        ordered = []
        for idx in pool:
            if idx in seen:
                continue
            seen.add(idx)
            ordered.append(idx)
        return ordered

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

        _, clusters = self._cluster(pop, fit, eps=eps)

        exploring = torch.zeros(n, dtype=torch.bool, device=device)
        refining = torch.zeros(n, dtype=torch.bool, device=device)
        completed = torch.zeros(n, dtype=torch.bool, device=device)
        anchor = torch.zeros(n, dtype=torch.bool, device=device)
        scout = torch.zeros(n, dtype=torch.bool, device=device)
        guardian = torch.zeros(n, dtype=torch.bool, device=device)
        release = torch.zeros(n, dtype=torch.bool, device=device)
        rescue = torch.zeros(n, dtype=torch.bool, device=device)
        overlap_group_ids = torch.full((n,), -1, dtype=torch.long, device=device)
        redundant_particles = torch.zeros(n, dtype=torch.bool, device=device)

        centers: List[torch.Tensor] = []
        active_centers: List[torch.Tensor] = []
        completed_centers: List[torch.Tensor] = []
        anchor_centers: List[torch.Tensor] = []
        rescue_pool: List[int] = []

        niche_tmp: List[Dict] = []
        complete_signal: List[bool] = []

        anchor_guard_ratio = 0.05 + 0.20 * float(intent_eff[5].item())
        completed_guard_ratio = min(0.10, 0.03 + 0.08 * float(intent_eff[5].item()))
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

            local_complete_signal = (
                radius_rel < self.r_small
                and radius_shrink > self.r_shrink
                and best_improve_rel < self.improve_eps
                and improve_ratio_stage < self.improve_ratio_eps
            )
            complete_signal.append(local_complete_signal)

            niche_tmp.append(
                {
                    "local_cluster_id": local_cid,
                    "members": m,
                    "size": int(len(m)),
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
        completed_flags, completion_counters = self._update_completion_counter(
            stable_ids=stable_ids,
            complete_signal=complete_signal,
            patience=self.complete_patience,
        )

        if len(centers) > 0:
            centers_t = torch.stack(centers, dim=0)
            best_fit_t = torch.tensor([row["best_fit"] for row in niche_tmp], device=device)
            merge_dist_eff = max(1e-6, float(self.merge_dist) * float(eps))
            group_ids = self._detect_overlap_groups(
                centers=centers_t,
                best_fit=best_fit_t,
                merge_dist=merge_dist_eff,
                merge_fit_eps=self.merge_fit_eps,
            )
        else:
            centers_t = torch.empty((0, self.dim), device=device)
            group_ids = torch.empty((0,), dtype=torch.long, device=device)

        keep_idx_by_gid: Dict[int, int] = {}
        for i, row in enumerate(niche_tmp):
            gid = int(group_ids[i].item()) if group_ids.numel() > 0 else -1
            if gid not in keep_idx_by_gid or row["best_fit"] < niche_tmp[keep_idx_by_gid[gid]]["best_fit"]:
                keep_idx_by_gid[gid] = i

        for i, row in enumerate(niche_tmp):
            row["niche_id"] = int(stable_ids[i])
            row["niche_age"] = int(stable_ages[i])
            row["completion_counter"] = int(completion_counters[i])
            row["is_completed_signal"] = bool(complete_signal[i])
            row["is_completed"] = bool(completed_flags[i])
            row["overlap_group_id"] = int(group_ids[i].item()) if group_ids.numel() > 0 else -1
            row["is_overlap_redundant"] = keep_idx_by_gid.get(row["overlap_group_id"], i) != i

        for row in niche_tmp:
            m = row["members"]
            if row["is_completed"]:
                status = "completed"
            else:
                is_refining = row["radius_rel"] < 0.05 and row["radius_shrink"] > 0.20 and row["best_improve_rel"] < 2e-3
                status = "refining" if is_refining else "exploring"
            row["status"] = status

            if status == "completed":
                completed[m] = True
                completed_centers.append(row["center"])
            elif status == "refining":
                refining[m] = True
                active_centers.append(row["center"])
            else:
                exploring[m] = True
                active_centers.append(row["center"])
                if not row["is_overlap_redundant"]:
                    c_stag = stagnation[m]
                    c_stag_start = stage_start_stagnation[m]
                    score = (c_stag - c_stag_start).float() + 0.5 * (c_stag >= 10).float()
                    local_bad = m[torch.argsort(score, descending=True)]
                    topk = max(1, int(len(m) * min(0.4, 0.2 + 0.2 * float(intent_eff[4].item()))))
                    for idx in local_bad[:topk]:
                        rescue_pool.append(int(idx.item()))

        per_niche_guardians: List[torch.Tensor] = [torch.empty((0,), dtype=torch.long, device=device) for _ in niche_tmp]
        for i, row in enumerate(niche_tmp):
            if row["status"] != "completed":
                continue
            m = row["members"]
            c_fit = fit[m]
            k = min(self.completed_guardian_cap, max(1, int(len(m) * completed_guard_ratio)))
            local_order = torch.argsort(c_fit)
            g_idx = m[local_order[:k]]
            per_niche_guardians[i] = g_idx
            guardian[g_idx] = True

        anchor_local_ids = self._select_anchor_niches(niche_tmp, self.max_anchor_niches)
        for i in anchor_local_ids:
            row = niche_tmp[i]
            m = row["members"]
            anchor[m] = True
            c_fit = fit[m]
            k_anchor = max(1, int(len(m) * anchor_guard_ratio))
            local_order = torch.argsort(c_fit)
            anchor_guard = m[local_order[:k_anchor]]
            if per_niche_guardians[i].numel() > 0:
                per_niche_guardians[i] = torch.unique(torch.cat([per_niche_guardians[i], anchor_guard], dim=0))
            else:
                per_niche_guardians[i] = anchor_guard
            guardian[per_niche_guardians[i]] = True
            anchor_centers.append(row["center"])
            if row["status"] == "exploring":
                exploring[m] = False
                refining[m] = True
                row["status"] = "refining"

        scout_pool = self._build_scout_release_pool(niche_tmp, per_niche_guardians)

        cluster_ids = torch.full((n,), -1, dtype=torch.long, device=device)
        for t, row in enumerate(niche_tmp):
            cluster_ids[row["members"]] = stable_ids[t]
            overlap_group_ids[row["members"]] = row["overlap_group_id"]
            if row["is_overlap_redundant"]:
                redundant_particles[row["members"]] = True

        release_budget = int(self.pop_size * float(intent_eff[3].item()) * (0.30 + 0.70 * float(intent_eff[0].item())))
        rescue_budget = int(self.pop_size * float(intent_eff[4].item()))

        if release_budget > 0 and len(scout_pool) > 0:
            release_sel = scout_pool[: min(release_budget, len(scout_pool))]
            release[torch.tensor(release_sel, dtype=torch.long, device=device)] = True
        if rescue_budget > 0 and len(rescue_pool) > 0:
            rescue_sel = rescue_pool[: min(rescue_budget, len(rescue_pool))]
            rescue[torch.tensor(rescue_sel, dtype=torch.long, device=device)] = True

        rescue = rescue & (~guardian) & exploring & (~anchor) & (~completed)
        release = release & (~guardian) & (~anchor)
        scout = (exploring & (~anchor) & (~guardian) & (~completed))
        scout = scout | (redundant_particles & (~anchor) & (~guardian) & (~completed))
        scout = scout | release

        completed_for_archive: List[torch.Tensor] = []
        completed_fit_for_archive: List[float] = []
        completed_age_for_archive: List[int] = []
        for row in niche_tmp:
            if row["status"] == "completed" and not row["is_overlap_redundant"]:
                completed_for_archive.append(row["center"])
                completed_fit_for_archive.append(float(row["best_fit"]))
                completed_age_for_archive.append(int(row["niche_age"]))

        archive_added = self._update_archive(
            completed_for_archive,
            completed_fit_for_archive,
            completed_age_for_archive,
            min_dist=max(1e-6, 1.5 * eps),
        )

        all_active_centers = active_centers + anchor_centers
        if len(all_active_centers) == 0:
            active_centers_t = torch.empty((0, self.dim), device=device)
        else:
            active_centers_t = torch.stack(all_active_centers, dim=0)
        anchor_centers_t = torch.stack(anchor_centers, dim=0) if len(anchor_centers) > 0 else torch.empty((0, self.dim), device=device)
        completed_centers_t = torch.stack(completed_centers, dim=0) if len(completed_centers) > 0 else torch.empty((0, self.dim), device=device)

        niche_rows: List[Dict] = []
        for row in niche_tmp:
            m = row["members"]
            center = row["center"]
            if self.archive_centers.numel() == 0:
                dist_archive_min = float("inf")
            else:
                dist_archive_min = float(torch.min(torch.norm(self.archive_centers - center.unsqueeze(0), dim=1)).item())

            release_count = int(release[m].sum().item())
            rescue_count = int(rescue[m].sum().item())
            guardian_count = int(guardian[m].sum().item())

            role = row["status"]
            if anchor[m].any().item():
                role = "anchor"
            elif row.get("is_overlap_redundant", False):
                role = "overlap_redundant"
            elif scout[m].any().item():
                role = "scout"

            niche_rows.append(
                {
                    "niche_id": int(row["niche_id"]),
                    "niche_age": int(row["niche_age"]),
                    "size": int(row["size"]),
                    "status": row["status"],
                    "role": role,
                    "center_norm": row["center_norm"],
                    "radius": row["radius"],
                    "radius_rel": row["radius_rel"],
                    "radius_shrink": row["radius_shrink"],
                    "best_fit": row["best_fit"],
                    "best_improve_rel": row["best_improve_rel"],
                    "accept_ratio_stage": row["accept_ratio_stage"],
                    "improve_ratio_stage": row["improve_ratio_stage"],
                    "stag_ratio": row["stag_ratio"],
                    "stag_growth": row["stag_growth"],
                    "guardian_count": guardian_count,
                    "release_count": release_count,
                    "rescue_count": rescue_count,
                    "dist_to_archive_min": dist_archive_min,
                    "overlap_group_id": int(row["overlap_group_id"]),
                    "is_overlap_redundant": bool(row["is_overlap_redundant"]),
                    "completion_counter": int(row["completion_counter"]),
                    "is_anchor": bool(anchor[m].any().item()),
                    "is_guardian": bool(guardian[m].any().item()),
                }
            )
            self.prev_status_by_id[int(row["niche_id"])] = row["status"]

        alive_ids = {int(row["niche_id"]) for row in niche_tmp}
        stale_status_ids = [k for k in self.prev_status_by_id.keys() if k not in alive_ids]
        for sid in stale_status_ids:
            self.prev_status_by_id.pop(sid, None)

        return NichePlan(
            exploring_mask=exploring,
            refining_mask=refining,
            completed_mask=completed,
            anchor_mask=anchor,
            scout_mask=scout,
            guardian_mask=guardian,
            release_mask=release,
            rescue_mask=rescue,
            cluster_ids=cluster_ids,
            overlap_group_ids=overlap_group_ids,
            active_centers=active_centers_t,
            anchor_centers=anchor_centers_t,
            completed_centers=completed_centers_t,
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
