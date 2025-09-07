#!/usr/bin/env python3
"""
Hyperparameter sweep and plotting for single_objective_optimization_03.py

- Runs grid or random sweeps over GA hyperparameters.
- Saves results to CSV and figures (convergence curve, heatmap).
- Minimal dependence on the optimizer module (only requires main(..., return_log=True)).
"""

import argparse
import itertools
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import single_objective_optimization_03 as opt  # import your optimizer module


# ------------------ Config dataclass ------------------

@dataclass
class GAConfig:
    pop_size: int # Size of the population
    ngen: int # Number of generations to evolve
    cxpb: float # Crossover probability
    mutpb: float # Mutation probability
    seed: int # Random seed for reproducibility
    threads: int = 1 # Number of parallel worker threads
    halloffame_k: int = 5 # How many top solutions to keep in the Hall of Fame
    verbose: bool = False # Whether to print progress / logs

    def to_kwargs(self) -> Dict[str, Any]:
        return dict(
            pop_size=self.pop_size,
            ngen=self.ngen,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            seed=self.seed,
            halloffame_k=self.halloffame_k,
            verbose=self.verbose,
            threads=self.threads,
            return_log=True,  # request log for plotting
        )

# ------------------ Runner ------------------

def run_once(cfg: GAConfig) -> Tuple[Dict[str, Any], float]:
    """Run a single GA with the given config, return (result, runtime_sec)."""
    t0 = time.perf_counter()
    result = opt.main(**cfg.to_kwargs())
    dt = time.perf_counter() - t0
    return result, dt


def grid_configs(
    pop_sizes: List[int],
    ngens: List[int],
    cxpbs: List[float],
    mutpbs: List[float],
    repeats: int,
    base_seed: int,
    threads: int,
) -> List[GAConfig]:
    """Create a grid of configurations with multiple seeds."""
    configs = []
    for pop, ngen, cx, mu in itertools.product(pop_sizes, ngens, cxpbs, mutpbs):
        for r in range(repeats):
            seed = base_seed + r
            configs.append(GAConfig(pop_size=pop, ngen=ngen, cxpb=cx, mutpb=mu,
                                    seed=seed, threads=threads, verbose=False))
    return configs


def random_configs(
    n: int,
    pop_range: Tuple[int, int],
    ngen_range: Tuple[int, int],
    cxpb_range: Tuple[float, float],
    mutpb_range: Tuple[float, float],
    repeats: int,
    base_seed: int,
    threads: int,
) -> List[GAConfig]:
    """Sample random hyperparameters uniformly from given ranges."""
    rng = np.random.default_rng(base_seed)
    configs = []
    for i in range(n):
        pop = int(rng.integers(pop_range[0], pop_range[1] + 1))
        # Round pop to nearest 10 for sanity
        pop = int(round(pop / 10) * 10)
        ngen = int(rng.integers(ngen_range[0], ngen_range[1] + 1))
        cx = float(rng.uniform(cxpb_range[0], cxpb_range[1]))
        mu = float(rng.uniform(mutpb_range[0], mutpb_range[1]))
        for r in range(repeats):
            seed = base_seed + i * 100 + r  # spread seeds across configs
            configs.append(GAConfig(pop_size=pop, ngen=ngen, cxpb=cx, mutpb=mu,
                                    seed=seed, threads=threads, verbose=False))
    return configs


# ------------------ Plotting ------------------

def plot_convergence(log_rows: List[Dict[str, Any]], outpath: str, title: str = "Convergence"):
    """Plot min/avg/std vs generation from log rows."""
    if not log_rows:
        print("No log rows to plot.")
        return
    df = pd.DataFrame(log_rows)
    # Some DEAP logs may have multi-index stats; ensure columns exist.
    for col in ["min", "avg", "std"]:
        if col not in df.columns:
            raise ValueError(f"Log missing '{col}' column. Got: {df.columns.tolist()}")

    plt.figure(figsize=(8, 5))
    plt.plot(df["gen"], df["min"], label="min (best)", lw=2, color="#2ca02c")
    plt.plot(df["gen"], df["avg"], label="avg", lw=1.8, color="#1f77b4")
    plt.fill_between(df["gen"], df["avg"] - df["std"], df["avg"] + df["std"],
                     color="#1f77b4", alpha=0.15, label="avg ± std")
    plt.xlabel("Generation")
    plt.ylabel("ΔV (km/s)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print(f"[saved] {outpath}")


def plot_heatmap(df: pd.DataFrame, outpath: str, ngen_fixed: Optional[int] = None, cxpb_fixed: Optional[float] = None):
    """
    Heatmap of mean best_fitness vs (pop_size, mutpb) after filtering to a fixed ngen and cxpb.
    If not provided, uses the most frequent values in the dataframe.
    """
    if df.empty:
        print("Empty dataframe; skipping heatmap.")
        return

    if ngen_fixed is None:
        ngen_fixed = df["ngen"].value_counts().idxmax()
    if cxpb_fixed is None:
        cxpb_fixed = df["cxpb"].value_counts().idxmax()

    dff = df[(df["ngen"] == ngen_fixed) & (df["cxpb"] == cxpb_fixed)]

    if dff.empty:
        print(f"No rows for ngen={ngen_fixed}, cxpb={cxpb_fixed}; skipping heatmap.")
        return

    pivot = dff.pivot_table(
        index="pop_size", columns="mutpb", values="best_fitness", aggfunc="mean"
    ).sort_index(axis=0).sort_index(axis=1)

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "Best ΔV (km/s)"})
    plt.title(f"Mean Best ΔV vs pop_size × mutpb  (ngen={ngen_fixed}, cxpb={cxpb_fixed})")
    plt.xlabel("Mutation probability (mutpb)")
    plt.ylabel("Population size")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print(f"[saved] {outpath}")


def plot_sensitivity(df: pd.DataFrame, outpath: str):
    """
    Simple sensitivity plots: best_fitness vs each hyperparameter.
    """
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    sns.lineplot(ax=axes[0], data=df, x="pop_size", y="best_fitness", marker="o")
    axes[0].set_title("ΔV vs pop_size")
    axes[0].set_ylabel("Best ΔV (km/s)")
    axes[0].grid(alpha=0.3)

    sns.lineplot(ax=axes[1], data=df, x="cxpb", y="best_fitness", marker="o")
    axes[1].set_title("ΔV vs cxpb")
    axes[1].grid(alpha=0.3)

    sns.lineplot(ax=axes[2], data=df, x="mutpb", y="best_fitness", marker="o")
    axes[2].set_title("ΔV vs mutpb")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print(f"[saved] {outpath}")


# ------------------ Main sweep routine ------------------
def main():
    outdir = "runs/sweep"
    threads = 4
    repeats = 2
    base_seed = 42
    use_random = True   # set True if you want random sampling
    random_n = 20         # number of random configs (ignored if use_random=False)

    # Grid search values
    pops = [100, 200]
    ngens = [20, 30]
    cxpbs = [0.6, 0.8]
    mutpbs = [0.2, 0.3, 0.5]

    os.makedirs(outdir, exist_ok=True)

    # Build configs
    if use_random and random_n > 0:
        cfgs = random_configs(
            n=random_n,
            pop_range=(80, 500),
            ngen_range=(15, 40),
            cxpb_range=(0.5, 0.9),
            mutpb_range=(0.1, 0.6),
            repeats=repeats,
            base_seed=base_seed,
            threads=threads,
        )
    else:
        cfgs = grid_configs(
            pop_sizes=pops,
            ngens=ngens,
            cxpbs=cxpbs,
            mutpbs=mutpbs,
            repeats=repeats,
            base_seed=base_seed,
            threads=threads,
        )

    print(f"Planned runs: {len(cfgs)}")

    # Run sweep
    rows = []
    best_log = None
    best_row_for_conv = None
    best_fitness_global = float("inf")

    for idx, cfg in enumerate(cfgs, start=1):
        print(f"[{idx}/{len(cfgs)}] Running: {cfg}")
        result, dt = run_once(cfg)

        row = {
            "pop_size": cfg.pop_size,
            "ngen": cfg.ngen,
            "cxpb": round(cfg.cxpb, 3),
            "mutpb": round(cfg.mutpb, 3),
            "seed": cfg.seed,
            "threads": cfg.threads,
            "runtime_sec": dt,
            "best_fitness": result.get("best_fitness", float("inf")),
            "dep_offset_days": result.get("dep_offset_days"),
            "tof_days": result.get("tof_days"),
            "best_departure": result.get("best_departure"),
            "best_arrival": result.get("best_arrival"),
        }
        rows.append(row)

        # Save best run log
        bf = row["best_fitness"]
        if bf < best_fitness_global and "log" in result:
            best_fitness_global = bf
            best_log = result["log"]
            best_row_for_conv = row

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    # Plots
    if best_log:
        title = (f"Convergence (best: ΔV={best_fitness_global:.3f} km/s, "
                 f"pop={best_row_for_conv['pop_size']}, ngen={best_row_for_conv['ngen']}, "
                 f"cxpb={best_row_for_conv['cxpb']}, mutpb={best_row_for_conv['mutpb']})")
        plot_convergence(best_log, os.path.join(outdir, "convergence.png"), title=title)

    plot_heatmap(df, os.path.join(outdir, "heatmap.png"))
    plot_sensitivity(df.sort_values("best_fitness"), os.path.join(outdir, "sensitivity.png"))


if __name__ == "__main__":
    main()