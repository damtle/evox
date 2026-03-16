import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from datetime import datetime

# 强制 PyTorch 全局默认使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_device(device)

from evox.problems.numerical import CEC2022
from evox.workflows import StdWorkflow, EvalMonitor
# 注意：虽然你的文件名还叫 pso_rl，但里面的 base 已经可以随意换了
from rl_ea.algorithms.pso_rl import make_rl_pso


def run_experiment(func_id: int, dim: int, enable_rl: bool, max_generations: int = 500, seed: int = 42):
    # 为每次独立实验设置不同的随机种子，保证实验的严谨性
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    lb = -100.0 * torch.ones(dim, device=device)
    ub = 100.0 * torch.ones(dim, device=device)

    # 预热期：可以根据需要修改
    warmup = int(max_generations * 0.1)

    algorithm = make_rl_pso(
        pop_size=100,
        lb=lb,
        ub=ub,
        device=device,
        enable_rl=enable_rl,
        rl_candidate_ratio=0.01,
        train_after=256,
        batch_size=128,
        warmup_gens=warmup
    )

    # 【新增】：动态获取底层算法的真实名称 (比如 "CoDE", "SaDE", "PSO")
    if enable_rl:
        # 如果是 RL 包装器，取它 base 的类名
        algo_name = algorithm.base.__class__.__name__
    else:
        # 如果是纯 EA，直接取它自己的类名
        algo_name = algorithm.__class__.__name__

    problem = CEC2022(problem_number=func_id, dimension=dim, device=device)

    # 安全转移设备
    for attr in ['OShift', 'M', 'SS']:
        if hasattr(problem, attr) and getattr(problem, attr) is not None:
            setattr(problem, attr, getattr(problem, attr).to(device))

    optimum_values = {
        1: 300, 2: 400, 3: 600, 4: 800, 5: 900,
        6: 1800, 7: 2000, 8: 2200, 9: 2300, 10: 2400, 11: 2600, 12: 2700
    }
    f_star = optimum_values.get(func_id, 0.0)

    original_evaluate = problem.evaluate

    def shifted_evaluate(state: torch.Tensor) -> torch.Tensor:
        # 直接减去极小值，让最优值变成 0
        return original_evaluate(state) - f_star

    problem.evaluate = shifted_evaluate

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

    # 返回结果的同时，把动态获取的算法名字也传出去
    return best_val, history, algo_name


def setup_logger(log_file: str):
    """设置双通道 Logger：同时输出到控制台和 txt 文件"""
    logger = logging.getLogger("ExperimentLogger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(message)s')

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    dim = 20
    max_generations = (10000 * dim) // 100

    # 【核心配置】：独立运行的次数
    num_runs = 5

    # 创建一个带有当前时间戳的独立文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"convergence_results/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "experiment_log.txt")
    logger = setup_logger(log_file)

    logger.info(f"{'=' * 60}")
    logger.info(f"🚀 开始 CEC2022 严谨测试 (D={dim}, Gens={max_generations}, Runs={num_runs})")
    logger.info(f"📁 本次所有图表和日志将保存至: {output_dir}")
    logger.info(f"{'=' * 60}")

    for func_id in range(1, 13):
        if func_id < 6:
            continue
        logger.info(f"\n正在运行 F{func_id}...")

        histories_base = []
        best_vals_base = []
        histories_rl = []
        best_vals_rl = []

        base_algo_name = "EA"  # 默认占位符

        for run in range(num_runs):
            current_seed = 42 + run * 100

            # 1. 跑纯 EA 基准
            best_base, hist_base, base_algo_name = run_experiment(func_id, dim, enable_rl=False,
                                                                  max_generations=max_generations, seed=current_seed)
            histories_base.append(hist_base)
            best_vals_base.append(best_base)

            # 2. 跑 RL 增强的 EA
            best_rl, hist_rl, _ = run_experiment(func_id, dim, enable_rl=True, max_generations=max_generations,
                                                 seed=current_seed)
            histories_rl.append(hist_rl)
            best_vals_rl.append(best_rl)

            # 日志动态显示算法名
            logger.info(
                f"   - Run {run + 1}/{num_runs} | Pure {base_algo_name}: {best_base:<12.4f} | RL-{base_algo_name}: {best_rl:<12.4f}")

        # ==== 统计学计算 ====
        mean_best_base = np.mean(best_vals_base)
        std_best_base = np.std(best_vals_base)
        mean_best_rl = np.mean(best_vals_rl)
        std_best_rl = np.std(best_vals_rl)

        mat_hist_base = np.array(histories_base)
        mat_hist_rl = np.array(histories_rl)

        mean_hist_base = np.mean(mat_hist_base, axis=0)
        std_hist_base = np.std(mat_hist_base, axis=0)

        mean_hist_rl = np.mean(mat_hist_rl, axis=0)
        std_hist_rl = np.std(mat_hist_rl, axis=0)

        winner = "👑 RL" if mean_best_rl < mean_best_base else "👑 EA" if mean_best_base < mean_best_rl else "Tie"
        # 汇总日志也动态显示
        logger.info(
            f"⭐ F{func_id:<2} 汇总 | Pure {base_algo_name}: {mean_best_base:.4f} ± {std_best_base:.4f} | RL-{base_algo_name}: {mean_best_rl:.4f} ± {std_best_rl:.4f} | {winner}")

        # ==== 独立绘制带有波动区间的阴影子图 ====
        fig, ax = plt.subplots(figsize=(8, 6))
        generations = np.arange(len(mean_hist_base))

        # 【绘制 Base 曲线与阴影，图例动态显示名称】
        ax.plot(generations, mean_hist_base, label=f'Pure {base_algo_name} (Mean)', color='#1f77b4', linewidth=2)
        ax.fill_between(generations,
                        np.clip(mean_hist_base - std_hist_base, a_min=1e-5, a_max=None),
                        mean_hist_base + std_hist_base,
                        color='#1f77b4', alpha=0.2)

        # 【绘制 RL 曲线与阴影，图例动态显示名称】
        ax.plot(generations, mean_hist_rl, label=f'RL-{base_algo_name} (Mean)', color='#d62728', linewidth=2,
                linestyle='--')
        ax.fill_between(generations,
                        np.clip(mean_hist_rl - std_hist_rl, a_min=1e-5, a_max=None),
                        mean_hist_rl + std_hist_rl,
                        color='#d62728', alpha=0.2)

        ax.set_title(f'F{func_id} Convergence ({base_algo_name}, D={dim}, Runs={num_runs})', fontsize=14,
                     fontweight='bold')
        ax.set_yscale('log')
        ax.set_xlabel('Generations', fontsize=12)
        ax.set_ylabel('Fitness (Log Scale)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout()

        plot_filename = os.path.join(output_dir, f"F{func_id}_convergence_{num_runs}runs.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close(fig)

        logger.info(f"📈 F{func_id} 图表已保存: {os.path.abspath(plot_filename)}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"✅ 全部测试完成！所有图表和包含 Mean±Std 的日志均已存入:\n   '{os.path.abspath(output_dir)}'")


if __name__ == "__main__":
    main()