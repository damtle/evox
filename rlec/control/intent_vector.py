import torch
from dataclasses import dataclass


@dataclass
class IntentVector:
    """
    RLEC 战略意图向量 (6D)
    所有值均被规范化到 [0, 1] 范围内
    """
    e_t: float  # Exploration pressure (探索压力): 鼓励大跨度搜索，容忍变差
    x_t: float  # Exploitation pressure (开发压力): 鼓励局部微调，严格要求提升
    d_t: float  # Diversity target (多样性目标): 0.0 代表允许坍缩，1.0 代表极度分散
    b_t: float  # Intervention budget (干预预算): 本阶段允许干预的最大个体比例
    r_t: float  # Rescue budget (脱困预算): 给停滞个体分配的重构资源比例
    p_t: float  # Elite protection (精英保护): 精英个体免受随机扰动的保护强度

    @classmethod
    def from_tensor(cls, action_tensor: torch.Tensor) -> 'IntentVector':
        """
        将 PPO 网络的输出（通常在 [-1, 1] 之间）映射到 [0, 1] 的物理意图区间
        """
        a = torch.clamp(action_tensor.squeeze(), 0.0, 1.0).tolist()

        return cls(
            e_t=a[0], x_t=a[1], d_t=a[2],
            b_t=a[3], r_t=a[4], p_t=a[5]
        )

    def __repr__(self):
        return (f"Intent[Explor:{self.e_t:.2f}, Exploit:{self.x_t:.2f}, Div:{self.d_t:.2f}, "
                f"Budget:{self.b_t:.2f}, Rescue:{self.r_t:.2f}, Elite:{self.p_t:.2f}]")