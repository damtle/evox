import torch


class StateBuilder:
    """Build per-individual state features for black-box optimization.
    New State format for each individual i:
        s_i = [x_i, Δx_i, f_i, Δf_i, dist_to_mean_i, stagnation_i]
    """

    def build(self, x: torch.Tensor, prev_x: torch.Tensor, f: torch.Tensor, prev_f: torch.Tensor,
              pop_mean: torch.Tensor, pop_std: torch.Tensor, stagnation: torch.Tensor) -> torch.Tensor:
        dx = x - prev_x
        df = f - prev_f

        # 适应度 Symlog 平滑
        f_norm = torch.sign(f) * torch.log1p(torch.abs(f))
        df_norm = torch.sign(df) * torch.log1p(torch.abs(df))

        # 1. 【新增】多样性特征：计算当前粒子距离种群中心的相对距离
        # 距离越小，说明越拥挤，越需要全局跳出
        dist_to_mean = torch.norm(x - pop_mean, dim=1, keepdim=True) / (torch.norm(pop_std) + 1e-6)

        # 2. 【新增】停滞特征：该粒子连续多少代没有提升
        # 归一化处理 (假设 50 代不更新算很久)
        stag_norm = stagnation.unsqueeze(1) / 50.0

        return torch.cat([
            x / 100.0,
            dx / 100.0,
            f_norm[:, None],
            df_norm[:, None],
            dist_to_mean,  # [pop, 1]
            stag_norm  # [pop, 1]
        ], dim=1)