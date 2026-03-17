import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_device(device)

from evox.problems.numerical import CEC2022
from evox.workflows import StdWorkflow, EvalMonitor
from rlec.algorithms.code_rlec import make_rlec_code


def run_experiment(func_id: int, dim: int, enable_rl: bool, max_generations: int = 500, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    lb = -100.0 * torch.ones(dim, device=device)
    ub = 100.0 * torch.ones(dim, device=device)

    algorithm = make_rlec_code(
        pop_size=100, lb=lb, ub=ub, device=device,
        stage_length=10, update_stages=2, enable_rl=enable_rl
    )

    algo_name = algorithm.base.__class__.__name__ if enable_rl else algorithm.__class__.__name__
    problem = CEC2022(problem_number=func_id, dimension=dim, device=device)

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
        return original_evaluate(state) - f_star

    problem.evaluate = shifted_evaluate

    monitor = EvalMonitor()
    workflow = StdWorkflow(algorithm, problem, monitor)

    history = []
    # 【新增】：专门记录每一代的 6D 控制意图
    intent_history = []

    workflow.init_step()
    best_val = algorithm.global_best_fit.item() if hasattr(algorithm, 'global_best_fit') else torch.min(
        algorithm.fit).item()
    history.append(best_val)

    for gen in range(max_generations):
        workflow.step()

        best_val = algorithm.global_best_fit.item() if hasattr(algorithm, 'global_best_fit') else torch.min(
            algorithm.fit).item()
        history.append(best_val)

        # 记录当前活跃的 Intent 向量 (仅当启用了 RL)
        if enable_rl and hasattr(algorithm, 'last_intent'):
            intent_history.append(algorithm.last_intent.detach().cpu().numpy())

    return best_val, history, np.array(intent_history), algo_name


def setup_logger(log_file: str):
    logger = logging.getLogger("RLECLogger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
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
    num_runs = 5

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"convergence_results/rlec_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "rlec_experiment_log.txt")
    logger = setup_logger(log_file)

    logger.info(
        f"{'=' * 65}\n🚀 开始 RLEC (PPO-MLP 元进化控制框架) 严谨测试\n   D={dim}, Gens={max_generations}, Runs={num_runs}\n📁 保存至: {output_dir}\n{'=' * 65}")

    for func_id in range(1, 13):
        logger.info(f"\n正在运行 F{func_id}...")

        histories_base, best_vals_base = [], []
        histories_rl, best_vals_rl, intents_rl = [], [], []
        base_algo_name = "EA"

        for run in range(num_runs):
            current_seed = 42 + run * 100

            best_b, hist_b, _, base_algo_name = run_experiment(func_id, dim, False, max_generations, current_seed)
            histories_base.append(hist_b)
            best_vals_base.append(best_b)

            best_r, hist_r, intent_r, _ = run_experiment(func_id, dim, True, max_generations, current_seed)
            histories_rl.append(hist_r)
            best_vals_rl.append(best_r)
            intents_rl.append(intent_r)

            logger.info(
                f"   - Run {run + 1}/{num_runs} | Pure {base_algo_name}: {best_b:<12.4e} | RLEC-{base_algo_name}: {best_r:<12.4e}")

        mean_best_base, std_best_base = np.mean(best_vals_base), np.std(best_vals_base)
        mean_best_rl, std_best_rl = np.mean(best_vals_rl), np.std(best_vals_rl)

        mean_hist_base, std_hist_base = np.mean(histories_base, axis=0), np.std(histories_base, axis=0)
        mean_hist_rl, std_hist_rl = np.mean(histories_rl, axis=0), np.std(histories_rl, axis=0)

        winner = "👑 RLEC" if mean_best_rl < mean_best_base else "👑 EA" if mean_best_base < mean_best_rl else "Tie"
        logger.info(
            f"⭐ F{func_id:<2} 汇总 | Pure {base_algo_name}: {mean_best_base:.4e} ± {std_best_base:.4e} | RLEC-{base_algo_name}: {mean_best_rl:.4e} ± {std_best_rl:.4e} | {winner}")

        # =========================================================
        # 👑 双图表绘制：左图收敛曲线，右图 6D 控制意图时序演化！
        # =========================================================
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        generations = np.arange(len(mean_hist_base))

        # --- 图 1: 收敛曲线 ---
        ax1.plot(generations, mean_hist_base, label=f'Pure {base_algo_name}', color='#1f77b4', linewidth=2)
        ax1.fill_between(generations, np.clip(mean_hist_base - std_hist_base, a_min=1e-8, a_max=None),
                         mean_hist_base + std_hist_base, color='#1f77b4', alpha=0.2)
        ax1.plot(generations, mean_hist_rl, label=f'RLEC-{base_algo_name}', color='#d62728', linewidth=2,
                 linestyle='--')
        ax1.fill_between(generations, np.clip(mean_hist_rl - std_hist_rl, a_min=1e-8, a_max=None),
                         mean_hist_rl + std_hist_rl, color='#d62728', alpha=0.2)
        ax1.set_title(f'F{func_id} Convergence', fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        ax1.set_xlabel('Generations', fontsize=12)
        ax1.set_ylabel('Fitness Error (Log Scale)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, which="both", ls="--", alpha=0.5)

        # --- 图 2: 6D 搜索意图演化轨迹 ---
        # 计算 5 次运行的 Intent 均值轨迹 shape: [max_generations, 6]
        mean_intents = np.mean(intents_rl, axis=0)
        labels = ['Exploration (e)', 'Exploitation (x)', 'Diversity Target (d)', 'Budget (b)', 'Rescue (r)',
                  'Elite Protect (p)']
        colors = ['#ff7f0e', '#2ca02c', '#2bc1ff', '#d62728', '#9467bd', '#8c564b']

        # 为了折线图不至于太密集，我们做一点平滑处理或按阶段取样
        gens_intent = np.arange(len(mean_intents))
        for i in range(6):
            # 用一点指数滑动平均(EMA)让轨迹更平滑好看
            smoothed = []
            val = mean_intents[0, i]
            for v in mean_intents[:, i]:
                val = 0.9 * val + 0.1 * v
                smoothed.append(val)
            ax2.plot(gens_intent, smoothed, label=labels[i], color=colors[i], linewidth=2)

        ax2.set_title(f'F{func_id} 6D Intent Trajectory (RLEC)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Generations', fontsize=12)
        ax2.set_ylabel('Intent Intensity [0, 1]', fontsize=12)
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.25, 1))
        ax2.grid(True, ls="--", alpha=0.5)

        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"F{func_id}_analysis_{num_runs}runs.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"📈 F{func_id} 分析图表已保存: {os.path.abspath(plot_filename)}")

    logger.info(f"\n{'=' * 65}\n✅ 全部测试完成！\n{'=' * 65}")


if __name__ == "__main__":
    main()