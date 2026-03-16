import torch


def compute_diversity(pop: torch.Tensor) -> torch.Tensor:
    return torch.std(pop, dim=0).mean()


def compute_fitness_skewness(fit: torch.Tensor) -> torch.Tensor:
    mean = fit.mean()
    std = fit.std() + 1e-8
    skewness = torch.mean(((fit - mean) / std) ** 3)
    return torch.clamp(skewness, -5.0, 5.0)


def compute_rank_entropy(fit: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    norm_fit = (fit - fit.mean()) / (fit.std() + 1e-8)
    probs = torch.softmax(-norm_fit / temperature, dim=0)
    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    max_entropy = torch.log(torch.tensor(float(fit.shape[0]), device=fit.device))
    return entropy / max_entropy


def compute_stagnation_ratio(stagnation: torch.Tensor, threshold: int = 10) -> torch.Tensor:
    stagnant_count = (stagnation >= threshold).sum().float()
    return stagnant_count / stagnation.shape[0]


# 【必须改 4】：新增聚类分离度特征，识别种群是否塌缩为单峰
def compute_elite_separation(pop: torch.Tensor, fit: torch.Tensor, q: float = 0.1) -> torch.Tensor:
    """计算精英簇与其他个体之间的分离度 (分离度越高，说明盆地越多/结构越清晰)"""
    N = pop.shape[0]
    k = max(1, int(N * q))

    _, indices = torch.sort(fit)
    elite_idx = indices[:k]
    non_elite_idx = indices[k:]

    elite_pop = pop[elite_idx]
    non_elite_pop = pop[non_elite_idx]

    # 找到精英簇的中心点
    elite_center = elite_pop.mean(dim=0)

    # 精英簇内部的紧凑度
    elite_spread = torch.norm(elite_pop - elite_center, dim=1).mean()
    # 大众个体离精英中心的平均距离
    separation = torch.norm(non_elite_pop - elite_center, dim=1).mean()

    # 比例越大，说明种群不仅有精英，还有大量个体在远方探索 (多模态结构)
    return separation / (elite_spread + 1e-8)