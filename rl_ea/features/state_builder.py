import torch


class StateBuilder:
    """Build per-individual state features for black-box optimization.

    State format for each individual i:
        s_i = [x_i, Δx_i, f_i, Δf_i]
    where Δx_i = x_i - x_i_prev and Δf_i = f_i - f_i_prev.

    For the first available state, callers should supply prev_x=x and prev_f=f so that
    the delta terms are zero.
    """

    def build(self, x: torch.Tensor, prev_x: torch.Tensor, f: torch.Tensor, prev_f: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x to have shape [pop, dim], got {tuple(x.shape)}")
        if prev_x.shape != x.shape:
            raise ValueError("prev_x must have the same shape as x")
        if f.ndim != 1 or prev_f.ndim != 1:
            raise ValueError("f and prev_f must be 1D tensors")

        dx = x - prev_x
        df = f - prev_f

        # 【关键修复】：对适应度特征进行 Symlog (Symmetric Log) 平滑处理，把 10^10 压到 20 左右的范围
        f_norm = torch.sign(f) * torch.log1p(torch.abs(f))
        df_norm = torch.sign(df) * torch.log1p(torch.abs(df))

        # 将 x 和 dx 除以 100 进行简单的尺度缩放 (CEC2022 范围通常是 -100 到 100)
        return torch.cat([x / 100.0, dx / 100.0, f_norm[:, None], df_norm[:, None]], dim=1)
