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
        return torch.cat([x, dx, f[:, None], df[:, None]], dim=1)
