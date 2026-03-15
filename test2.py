import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 强制 PyTorch 全局默认使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_device(device)

from evox.problems.numerical import CEC2022
from evox.workflows import StdWorkflow, EvalMonitor
from rl_ea.algorithms.pso_rl import make_rl_pso


def run_experiment(func_id: int, dim: int, enable_rl: bool, max_generations: int = 500):
    lb = -100.0 * torch.ones(dim, device=device)
    ub = 100.0 * torch.ones(dim, device=device)
    warmup = int(max_generations * 0.1)
    algorithm = make_rl_pso(
        pop_size=100,
        lb=lb,
        ub=ub,
        device=device,
        enable_rl=enable_rl,
        rl_candidate_ratio=0.05,  # 5% 的算力给 RL
        train_after=256,
        batch_size=128,
        warmup_gens=warmup
    )

    problem = CEC2022(problem_number=func_id, dimension=dim, device=device)

    # 安全转移设备
    for attr in ['OShift', 'M', 'SS']:
        if hasattr(problem, attr) and getattr(problem, attr) is not None:
            setattr(problem, attr, getattr(problem, attr).to(device))

    monitor = EvalMonitor()
    workflow = StdWorkflow(algorithm, problem, monitor)

    history = []

    workflow.init_step()
    # 记录第 0 代的最优值
    best_val = algorithm.global_best_fit.item() if hasattr(algorithm, 'global_best_fit') else torch.min(
        algorithm.fit).item()
    history.append(best_val)

    for gen in range(max_generations):
        workflow.step()

        # 兼容性读取全局最优适应度
        if hasattr(algorithm, 'global_best_fit'):
            best_val = algorithm.global_best_fit.item()
        elif hasattr(algorithm, 'fit'):
            best_val = torch.min(algorithm.fit).item()
        else:
            raise AttributeError("无法从算法对象中获取最优适应度值！")

        history.append(best_val)

    return best_val, history


def main():
    dim = 20
    max_generations = (10000 * dim) // 100

    print(f"{'=' * 60}")
    print(f"🚀 开始 CEC2022 批量测试并独立绘制收敛曲线 (D={dim}, Gens={max_generations})")
    print(f"{'=' * 60}")

    # 【新增】：创建一个文件夹专门用来保存图片
    output_dir = "convergence_results"
    os.makedirs(output_dir, exist_ok=True)

    for func_id in range(1, 13):
        print(f"\n正在运行 F{func_id}...")
        # if func_id < 6:
        #     continue
        # 1. 跑纯 EA 基准
        best_baseline, hist_baseline = run_experiment(func_id, dim, enable_rl=False, max_generations=max_generations)
        # 2. 跑 RL 增强的 EA
        best_rl, hist_rl = run_experiment(func_id, dim, enable_rl=True, max_generations=max_generations)

        winner = "👑 RL" if best_rl < best_baseline else "👑 EA" if best_baseline < best_rl else "Tie"
        print(f"F{func_id:<2} | Base: {best_baseline:<15.4f} | RL: {best_rl:<15.4f} | {winner}")

        # ==== 独立绘制子图 ====
        # 每次循环都新建一个清晰度高、尺寸适中的画布
        fig, ax = plt.subplots(figsize=(8, 6))

        # 技巧：CEC2022 函数的适应度通常在数百到数千，下降幅度极大，用对数坐标轴 (log scale) 最能看清后期微小的差距
        ax.plot(hist_baseline, label='Pure SaDE', color='#1f77b4', linewidth=2)
        ax.plot(hist_rl, label='RL-SaDE', color='#d62728', linewidth=2, linestyle='--')

        ax.set_title(f'F{func_id} Convergence (D={dim})', fontsize=14, fontweight='bold')
        ax.set_yscale('log')  # 启用 Y轴对数刻度
        ax.set_xlabel('Generations', fontsize=12)
        ax.set_ylabel('Fitness (Log Scale)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout()

        # 拼接文件路径并保存
        plot_filename = os.path.join(output_dir, f"F{func_id}_convergence_{max_generations}gens.png")
        plt.savefig(plot_filename, dpi=300)

        # 【关键】：保存完毕后必须关闭画布，否则会发生内存泄漏，并且下一张图会和上一张图重叠
        plt.close(fig)

        print(f"📈 F{func_id} 收敛曲线图已实时保存至: {os.path.abspath(plot_filename)}")

    print(f"\n{'=' * 60}")
    print(f"✅ 全部测试完成！所有图片均已存入 '{output_dir}' 文件夹。")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()