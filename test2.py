import torch
from evox.problems.numerical import CEC2022
from evox.workflows import StdWorkflow, EvalMonitor

from rl_ea.algorithms.pso_rl import make_rl_pso


def run_experiment(func_id: int, dim: int, enable_rl: bool, max_generations: int = 100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 根据 CEC2022 的常规标准，搜索边界通常设定为 [-100, 100]
    lb = -100.0 * torch.ones(dim, device=device)
    ub = 100.0 * torch.ones(dim, device=device)

    # 实例化算法 (利用 enable_rl 开关自动决定是否启用 RL)
    algorithm = make_rl_pso(
        pop_size=100,
        lb=lb,
        ub=ub,
        device=device,
        enable_rl=enable_rl,
        rl_candidate_ratio=0.3,
        train_after=256,
        batch_size=128
    )

    # 实例化当前循环对应的 CEC2022 问题
    problem = CEC2022(problem_number=func_id, dimension=dim, device=device)

    # 建立工作流
    monitor = EvalMonitor()
    workflow = StdWorkflow(algorithm, problem, monitor)

    # 迭代训练
    workflow.init_step()
    for gen in range(max_generations):
        workflow.step()

    # 获取最后一代的全局最优值
    best_val = algorithm.base.global_best_fit.item() if enable_rl else algorithm.global_best_fit.item()
    return best_val


def main():
    dim = 20  # 测试维度：推荐 10 或 20 (因为 D=2 不支持 F6-F8)
    max_generations = 1000  # 测试代数：正式写论文时通常需要设为 1000 或按计算次数(FEs)终止

    print(f"{'=' * 60}")
    print(f"🚀 开始 CEC2022 Benchmark 批量测试 (D={dim}, Gens={max_generations})")
    print(f"{'=' * 60}")
    print(f"{'Func':<6} | {'Pure PSO (Baseline)':<20} | {'RL-Enhanced PSO':<20} | {'Winner':<6}")
    print("-" * 60)

    results = []

    # 遍历 CEC2022 的 F1 到 F12
    for func_id in range(1, 13):
        # 1. 跑纯 EA 基准 (屏蔽 RL)
        best_baseline = run_experiment(func_id, dim, enable_rl=False, max_generations=max_generations)

        # 2. 跑带有 RL 增强的 EA
        best_rl = run_experiment(func_id, dim, enable_rl=True, max_generations=max_generations)

        results.append((func_id, best_baseline, best_rl))

        # 判断谁赢了 (因为是求最小值，所以数值越小越好)
        if best_rl < best_baseline:
            winner = "👑 RL"
        elif best_baseline < best_rl:
            winner = "👑 EA"
        else:
            winner = "Tie"

        # 格式化输出这一行的结果
        print(f"F{func_id:<5} | {best_baseline:<20.4f} | {best_rl:<20.4f} | {winner}")

    print(f"{'=' * 60}")
    print("✅ 全部 12 个函数测试完成！")


if __name__ == "__main__":
    main()