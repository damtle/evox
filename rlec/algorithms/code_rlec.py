from typing import Optional

import torch
from evox.algorithms import DE

from rlec.algorithms.rlec_wrapper import RLECWrapper


def make_rlec_code(
    pop_size: int,
    lb: torch.Tensor,
    ub: torch.Tensor,
    device: Optional[torch.device] = None,
    stage_length: int = 20,
    update_stages: int = 5,
    enable_rl: bool = True,
    func_id: Optional[int] = None,
    run_id: Optional[int] = None,
    log_dir: Optional[str] = None,
):
    """Build DE baseline or RLEC-wrapped DE."""
    device = torch.get_default_device() if device is None else device
    base_code = DE(pop_size, lb, ub, device=device)

    if not enable_rl:
        return base_code

    return RLECWrapper(
        base_algo=base_code,
        stage_length=stage_length,
        update_stages=update_stages,
        device=device,
        func_id=func_id,
        run_id=run_id,
        log_dir=log_dir,
    )
