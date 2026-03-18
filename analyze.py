from __future__ import annotations

import os
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 只需要改这里
# =========================
RESULT_DIR = "convergence_results/rlec_20260318_005538"


# =========================
# 基础工具
# =========================
STAGE_PATTERN = re.compile(r"F(?P<fid>\d+)_run(?P<rid>\d+)_stage_log\.csv$")
NICHE_PATTERN = re.compile(r"F(?P<fid>\d+)_run(?P<rid>\d+)_niche_log\.csv$")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"[WARN] 读取失败: {path.name} -> {e}")
        return None


def ema(series: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    if len(series) == 0:
        return series
    out = np.zeros_like(series, dtype=float)
    out[0] = float(series[0])
    for i in range(1, len(series)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * float(series[i])
    return out


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def sort_stage_df(df: pd.DataFrame) -> pd.DataFrame:
    for key in ["stage", "stage_id", "gen", "generation", "step"]:
        if key in df.columns:
            return df.sort_values(key).reset_index(drop=True)
    return df.reset_index(drop=True)


def sort_niche_df(df: pd.DataFrame) -> pd.DataFrame:
    sort_keys = [c for c in ["stage", "stage_id", "gen", "generation", "niche_id"] if c in df.columns]
    if sort_keys:
        return df.sort_values(sort_keys).reset_index(drop=True)
    return df.reset_index(drop=True)


# =========================
# 扫描文件
# =========================
def discover_runs(result_dir: Path) -> Dict[Tuple[int, int], Dict[str, Path]]:
    runs: Dict[Tuple[int, int], Dict[str, Path]] = {}

    for p in result_dir.glob("*.csv"):
        m1 = STAGE_PATTERN.match(p.name)
        m2 = NICHE_PATTERN.match(p.name)

        if m1:
            fid = int(m1.group("fid"))
            rid = int(m1.group("rid"))
            runs.setdefault((fid, rid), {})["stage"] = p
        elif m2:
            fid = int(m2.group("fid"))
            rid = int(m2.group("rid"))
            runs.setdefault((fid, rid), {})["niche"] = p

    return runs


# =========================
# 单个 run 的图
# =========================
def plot_single_run(fid: int, rid: int, stage_df: Optional[pd.DataFrame], niche_df: Optional[pd.DataFrame], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()

    # ---------- 图1：intent + gates ----------
    ax = axes[0]
    if stage_df is not None:
        stage_df = sort_stage_df(stage_df)
        xcol = pick_col(stage_df, ["stage", "stage_id", "gen", "generation", "step"])
        x = np.arange(len(stage_df)) if xcol is None else stage_df[xcol].to_numpy()

        intent_cols = {
            "e": pick_col(stage_df, ["intent_e", "e", "intent_exploration"]),
            "x": pick_col(stage_df, ["intent_x", "x", "intent_exploitation"]),
            "d": pick_col(stage_df, ["intent_d", "d", "intent_diversity"]),
            "b": pick_col(stage_df, ["intent_b", "b", "intent_budget"]),
            "r": pick_col(stage_df, ["intent_r", "r", "intent_rescue"]),
            "p": pick_col(stage_df, ["intent_p", "p", "intent_protect"]),
        }

        for name, col in intent_cols.items():
            if col is not None:
                y = stage_df[col].to_numpy(dtype=float)
                ax.plot(x, ema(y, 0.15), label=name)

        g_ref = pick_col(stage_df, ["g_refine", "gate_refine"])
        g_rec = pick_col(stage_df, ["g_recover", "gate_recover"])
        if g_ref is not None:
            ax.plot(x, ema(stage_df[g_ref].to_numpy(dtype=float), 0.15), linestyle="--", label="g_refine")
        if g_rec is not None:
            ax.plot(x, ema(stage_df[g_rec].to_numpy(dtype=float), 0.15), linestyle="--", label="g_recover")

        ax.set_title(f"F{fid} Run{rid} - Intent / Gates")
        ax.set_xlabel("Stage")
        ax.set_ylabel("Value")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(ncol=4, fontsize=9)
    else:
        ax.set_title(f"F{fid} Run{rid} - Intent / Gates (no stage log)")

    # ---------- 图2：niche数量 / release / rescue / guardians ----------
    ax = axes[1]
    if stage_df is not None:
        xcol = pick_col(stage_df, ["stage", "stage_id", "gen", "generation", "step"])
        x = np.arange(len(stage_df)) if xcol is None else stage_df[xcol].to_numpy()

        metric_candidates = {
            "n_niches": ["n_niches", "niche_count"],
            "n_exploring": ["n_exploring", "exploring_niches"],
            "n_refining": ["n_refining", "refining_niches"],
            "n_converged": ["n_converged", "converged_niches"],
            "n_release": ["n_release", "release_count", "release_particles"],
            "n_rescue": ["n_rescue", "rescue_count", "rescue_particles"],
            "n_guardians": ["n_guardians", "guardian_count", "guardian_particles"],
            "archive_size": ["archive_size"],
        }

        for label, candidates in metric_candidates.items():
            col = pick_col(stage_df, candidates)
            if col is not None:
                ax.plot(x, stage_df[col].to_numpy(dtype=float), label=label)

        ax.set_title(f"F{fid} Run{rid} - Niche Lifecycle Summary")
        ax.set_xlabel("Stage")
        ax.set_ylabel("Count")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(ncol=2, fontsize=9)
    else:
        ax.set_title(f"F{fid} Run{rid} - Niche Lifecycle Summary (no stage log)")

    # ---------- 图3：best_fit + reward相关 ----------
    ax = axes[2]
    if stage_df is not None:
        xcol = pick_col(stage_df, ["stage", "stage_id", "gen", "generation", "step"])
        x = np.arange(len(stage_df)) if xcol is None else stage_df[xcol].to_numpy()

        best_col = pick_col(stage_df, ["best_fit", "global_best_fit"])
        reward_col = pick_col(stage_df, ["reward", "stage_reward", "total_reward"])
        acc_col = pick_col(stage_df, ["accept_ratio", "last_accept_ratio"])
        rel_eff_col = pick_col(stage_df, ["release_efficiency", "release_eff"])
        guard_stab_col = pick_col(stage_df, ["guardian_stability"])

        if best_col is not None:
            ax.plot(x, stage_df[best_col].to_numpy(dtype=float), label="best_fit")
        if reward_col is not None:
            ax.plot(x, ema(stage_df[reward_col].to_numpy(dtype=float), 0.2), label="reward")
        if acc_col is not None:
            ax.plot(x, ema(stage_df[acc_col].to_numpy(dtype=float), 0.2), label="accept_ratio")
        if rel_eff_col is not None:
            ax.plot(x, ema(stage_df[rel_eff_col].to_numpy(dtype=float), 0.2), label="release_eff")
        if guard_stab_col is not None:
            ax.plot(x, ema(stage_df[guard_stab_col].to_numpy(dtype=float), 0.2), label="guardian_stability")

        ax.set_title(f"F{fid} Run{rid} - Best / Reward / Credit")
        ax.set_xlabel("Stage")
        ax.set_ylabel("Value")
        ax.grid(True, ls="--", alpha=0.4)
        ax.legend(fontsize=9)
    else:
        ax.set_title(f"F{fid} Run{rid} - Best / Reward / Credit (no stage log)")

    # ---------- 图4：niche 级别散点 / 寿命 ----------
    ax = axes[3]
    if niche_df is not None:
        niche_df = sort_niche_df(niche_df)

        age_col = pick_col(niche_df, ["niche_age", "age"])
        best_col = pick_col(niche_df, ["best_fit", "niche_best_fit"])
        radius_col = pick_col(niche_df, ["radius_rel", "niche_radius_rel", "radius"])
        status_col = pick_col(niche_df, ["status"])

        if age_col is not None and best_col is not None:
            if status_col is not None:
                for status, sub in niche_df.groupby(status_col):
                    ax.scatter(
                        sub[age_col].to_numpy(dtype=float),
                        sub[best_col].to_numpy(dtype=float),
                        s=16,
                        alpha=0.7,
                        label=str(status),
                    )
            else:
                ax.scatter(
                    niche_df[age_col].to_numpy(dtype=float),
                    niche_df[best_col].to_numpy(dtype=float),
                    s=16,
                    alpha=0.7,
                    label="niches",
                )

            ax.set_title(f"F{fid} Run{rid} - Niche Age vs Best Fit")
            ax.set_xlabel("Niche Age")
            ax.set_ylabel("Best Fit")
            ax.grid(True, ls="--", alpha=0.4)
            ax.legend(fontsize=9)
        elif radius_col is not None and best_col is not None:
            ax.scatter(
                niche_df[radius_col].to_numpy(dtype=float),
                niche_df[best_col].to_numpy(dtype=float),
                s=16,
                alpha=0.7,
            )
            ax.set_title(f"F{fid} Run{rid} - Niche Radius vs Best Fit")
            ax.set_xlabel("Radius")
            ax.set_ylabel("Best Fit")
            ax.grid(True, ls="--", alpha=0.4)
        else:
            ax.set_title(f"F{fid} Run{rid} - Niche Scatter (columns not found)")
    else:
        ax.set_title(f"F{fid} Run{rid} - Niche Scatter (no niche log)")

    plt.tight_layout()
    out_path = out_dir / f"F{fid}_run{rid}_csv_analysis.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 保存单run分析图: {out_path}")


# =========================
# 跨 run 汇总图
# =========================
def aggregate_stage_runs(run_items: List[Tuple[int, int, Path]]) -> Optional[pd.DataFrame]:
    dfs = []
    for fid, rid, p in run_items:
        df = safe_read_csv(p)
        if df is None:
            continue
        df = sort_stage_df(df).copy()
        df["func_id"] = fid
        df["run_id"] = rid
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def plot_func_summary(fid: int, stage_dfs: List[pd.DataFrame], out_dir: Path) -> None:
    if not stage_dfs:
        return

    # 对齐 stage 索引
    aligned = []
    min_len = min(len(df) for df in stage_dfs if df is not None and len(df) > 0)
    if min_len == 0:
        return

    for df in stage_dfs:
        df2 = sort_stage_df(df).iloc[:min_len].reset_index(drop=True)
        aligned.append(df2)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()
    x = np.arange(min_len)

    # ---- 1 best_fit mean±std
    ax = axes[0]
    best_col = None
    for cands in [["best_fit", "global_best_fit"]]:
        for df in aligned:
            best_col = pick_col(df, cands)
            if best_col is not None:
                break
        if best_col is not None:
            break

    if best_col is not None:
        ys = np.stack([df[best_col].to_numpy(dtype=float) for df in aligned], axis=0)
        mean_y = ys.mean(axis=0)
        std_y = ys.std(axis=0)
        ax.plot(x, mean_y, label="best_fit mean")
        ax.fill_between(x, mean_y - std_y, mean_y + std_y, alpha=0.2)
        ax.set_title(f"F{fid} - Best Fit Mean±Std")
        ax.set_xlabel("Stage")
        ax.set_ylabel("Best Fit")
        ax.grid(True, ls="--", alpha=0.4)

    # ---- 2 intents mean
    ax = axes[1]
    intent_map = {
        "e": ["intent_e", "e", "intent_exploration"],
        "x": ["intent_x", "x", "intent_exploitation"],
        "d": ["intent_d", "d", "intent_diversity"],
        "b": ["intent_b", "b", "intent_budget"],
        "r": ["intent_r", "r", "intent_rescue"],
        "p": ["intent_p", "p", "intent_protect"],
    }
    for label, cands in intent_map.items():
        col = None
        for df in aligned:
            col = pick_col(df, cands)
            if col is not None:
                break
        if col is not None:
            ys = np.stack([df[col].to_numpy(dtype=float) for df in aligned], axis=0)
            ax.plot(x, ema(ys.mean(axis=0), 0.15), label=label)
    ax.set_title(f"F{fid} - Intent Mean")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Value")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(ncol=3, fontsize=9)

    # ---- 3 niche counts mean
    ax = axes[2]
    count_map = {
        "n_niches": ["n_niches", "niche_count"],
        "n_refining": ["n_refining", "refining_niches"],
        "n_converged": ["n_converged", "converged_niches"],
        "n_release": ["n_release", "release_count", "release_particles"],
        "archive_size": ["archive_size"],
    }
    for label, cands in count_map.items():
        col = None
        for df in aligned:
            col = pick_col(df, cands)
            if col is not None:
                break
        if col is not None:
            ys = np.stack([df[col].to_numpy(dtype=float) for df in aligned], axis=0)
            ax.plot(x, ys.mean(axis=0), label=label)
    ax.set_title(f"F{fid} - Niche Metrics Mean")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Count")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)

    # ---- 4 reward/credit mean
    ax = axes[3]
    reward_map = {
        "reward": ["reward", "stage_reward", "total_reward"],
        "accept_ratio": ["accept_ratio", "last_accept_ratio"],
        "release_eff": ["release_efficiency", "release_eff"],
        "guardian_stability": ["guardian_stability"],
        "repeat_ratio": ["repeat_ratio"],
    }
    for label, cands in reward_map.items():
        col = None
        for df in aligned:
            col = pick_col(df, cands)
            if col is not None:
                break
        if col is not None:
            ys = np.stack([df[col].to_numpy(dtype=float) for df in aligned], axis=0)
            ax.plot(x, ema(ys.mean(axis=0), 0.2), label=label)
    ax.set_title(f"F{fid} - Reward / Credit Mean")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Value")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = out_dir / f"F{fid}_summary_analysis.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 保存函数汇总图: {out_path}")


# =========================
# 额外：从 experiment log 画总表
# =========================
SUMMARY_PATTERN = re.compile(
    r"⭐\s*F(?P<fid>\d+).*?Pure\s+(?P<base_name>[A-Za-z0-9_]+):\s*(?P<base_mean>[0-9eE+\-\.]+)\s*±\s*(?P<base_std>[0-9eE+\-\.]+)\s*\|\s*RLEC-[A-Za-z0-9_]+:\s*(?P<rlec_mean>[0-9eE+\-\.]+)\s*±\s*(?P<rlec_std>[0-9eE+\-\.]+)"
)


def parse_experiment_log(log_path: Path) -> Optional[pd.DataFrame]:
    if not log_path.exists():
        return None
    rows = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = SUMMARY_PATTERN.search(line)
            if m:
                rows.append(
                    {
                        "func_id": int(m.group("fid")),
                        "base_mean": float(m.group("base_mean")),
                        "base_std": float(m.group("base_std")),
                        "rlec_mean": float(m.group("rlec_mean")),
                        "rlec_std": float(m.group("rlec_std")),
                    }
                )
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("func_id").reset_index(drop=True)


def plot_experiment_summary(df: pd.DataFrame, out_dir: Path) -> None:
    x = df["func_id"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, df["base_mean"].to_numpy(), marker="o", label="Pure DE mean")
    ax.plot(x, df["rlec_mean"].to_numpy(), marker="o", label="RLEC-DE mean")
    ax.fill_between(
        x,
        df["base_mean"].to_numpy() - df["base_std"].to_numpy(),
        df["base_mean"].to_numpy() + df["base_std"].to_numpy(),
        alpha=0.15,
    )
    ax.fill_between(
        x,
        df["rlec_mean"].to_numpy() - df["rlec_std"].to_numpy(),
        df["rlec_mean"].to_numpy() + df["rlec_std"].to_numpy(),
        alpha=0.15,
    )
    ax.set_title("Experiment Summary from rlec_experiment_log.txt")
    ax.set_xlabel("Function ID")
    ax.set_ylabel("Fitness Error")
    ax.set_xticks(x)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()

    out_path = out_dir / "experiment_summary_from_log.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 保存实验总览图: {out_path}")


# =========================
# 主流程
# =========================
def main():
    result_dir = Path(RESULT_DIR)
    if not result_dir.exists():
        raise FileNotFoundError(f"结果文件夹不存在: {result_dir}")

    analysis_dir = result_dir / "csv_analysis"
    per_run_dir = analysis_dir / "per_run"
    per_func_dir = analysis_dir / "per_function"
    ensure_dir(per_run_dir)
    ensure_dir(per_func_dir)

    runs = discover_runs(result_dir)
    if not runs:
        print("[WARN] 没找到任何 F*_run*_stage_log.csv / niche_log.csv 文件")
        return

    print(f"[INFO] 发现 {len(runs)} 个 run 日志对/单文件")

    # 单 run 图
    stage_by_func: Dict[int, List[pd.DataFrame]] = {}

    for (fid, rid), files in sorted(runs.items()):
        stage_df = safe_read_csv(files["stage"]) if "stage" in files else None
        niche_df = safe_read_csv(files["niche"]) if "niche" in files else None

        plot_single_run(fid, rid, stage_df, niche_df, per_run_dir)

        if stage_df is not None:
            stage_by_func.setdefault(fid, []).append(stage_df)

    # 每个函数跨 run 汇总
    for fid, stage_dfs in sorted(stage_by_func.items()):
        plot_func_summary(fid, stage_dfs, per_func_dir)

    # 解析总实验日志
    log_path = result_dir / "rlec_experiment_log.txt"
    exp_df = parse_experiment_log(log_path)
    if exp_df is not None:
        plot_experiment_summary(exp_df, analysis_dir)
    else:
        print("[INFO] 未解析到 rlec_experiment_log.txt 的汇总行，跳过总览图")

    print(f"[DONE] 全部分析图已保存到: {analysis_dir}")


if __name__ == "__main__":
    main()