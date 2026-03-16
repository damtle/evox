from typing import Optional
import torch
from evox.algorithms import CoDE
from rlec.algorithms.rlec_wrapper import RLECWrapper


def make_rlec_code(
        pop_size: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        device: Optional[torch.device] = None,
        stage_length: int = 20,
        update_stages: int = 5,
        enable_rl: bool = True
):
    """
    快速创建一个被 RLEC 控制器包裹的 CoDE 算法实例
    """
    device = torch.get_default_device() if device is None else device
    base_code = CoDE(pop_size, lb, ub, device=device)

    if not enable_rl:
        return base_code

    algo = RLECWrapper(
        base_algo=base_code,
        stage_length=stage_length,
        update_stages=update_stages,
        device=device
    )
    return algo