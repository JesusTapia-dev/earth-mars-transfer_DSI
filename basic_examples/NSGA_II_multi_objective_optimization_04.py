"""
multi_objective_optimization_04.py

Multi-objective optimization for Earth->Mars transfer.
- Objectives:
    1. Minimize total Delta-V (heliocentric impulsive estimate)
    2. Minimize Time of Flight
- Method: NSGA-II Genetic Algorithm via DEAP
- Decision variables: departure date (days offset) and time of flight (days
"""

import random
from datetime import datetime, timedelta
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools, algorithms

# Assuming earth_mars_transfer_02.py is in the same directory
from earth_mars_transfer_02 import run_transfer

# --------- Configurable search window ---------
START_DATE = datetime(2026, 8, 1)
END_DATE   = datetime(2029, 2, 1)
TOF_MIN = 140.0
TOF_MAX = 320.0
# -----------------------------------------------------------


def decode_individual(ind: List[float]):
    """Decode individual -> (departure_datetime, arrival_datetime)."""
    dep_offset_days = float(ind[0])
    tof_days = float(ind[1])
    dep_date = START_DATE + timedelta(days=dep_offset_days)
    arr_date = dep_date + timedelta(days=tof_days)
    return dep_date, arr_date


def objective(individual: List[float]):
    """
    Objective: minimize both total DV (km/s) and time of flight (days).
    """
    dep_offset, tof_days = float(individual[0]), float(individual[1])
    window_days = float((END_DATE - START_DATE).days)

    # Short-circuit infeasible candidates
    if not (0.0 <= dep_offset <= window_days and TOF_MIN <= tof_days <= TOF_MAX):
        # Return a large penalty for both objectives
        return (1e6, 1e6)

    dep_dt, arr_dt = decode_individual(individual)
    dep_tuple = (dep_dt.year, dep_dt.month, dep_dt.day, dep_dt.hour, dep_dt.minute, dep_dt.second)
    arr_tuple = (arr_dt.year, arr_dt.month, arr_dt.day, arr_dt.hour, arr_dt.minute, arr_dt.second)

    try:
        res = run_transfer(dep_tuple, arr_tuple)
        dv_total = float(res['dv_total'])
    except Exception as e:
        print(f"[objective] run_transfer failed for dep={dep_tuple}, arr={arr_tuple}: {e}")
        dv_total = 1e6

    return (dv_total, tof_days)


def setup_deap(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if 'FitnessMulti' not in creator.__dict__:
        creator.create('FitnessMulti', base.Fitness, weights=(-1.0, -1.0))
    if 'Individual' not in creator.__dict__:
        creator.create('Individual', list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    window_days = float((END_DATE - START_DATE).days)

    toolbox.register('attr_dep', random.uniform, 0.0, window_days)
    toolbox.register('attr_tof', random.uniform, TOF_MIN, TOF_MAX)
    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.attr_dep, toolbox.attr_tof), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('evaluate', objective)

    # Operators
    low = [0.0, TOF_MIN]
    up  = [window_days, TOF_MAX]
    toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=low, up=up, eta=15.0)
    toolbox.register('mutate', tools.mutPolynomialBounded, low=low, up=up, eta=20.0, indpb=0.5)
    
    # *** Selection operator for NSGA-II ***
    toolbox.register('select', tools.selNSGA2)

    return toolbox

def plot_pareto_front(pareto_front, outpath="runs/nsga_ii/pareto_front.png"):
    """
    Plots the Pareto front solutions.

    Args:
        pareto_front (list): A list of DEAP individual objects from the hall of fame.
        outpath (str): The path to save the output image file.
    """
    if not pareto_front:
        print("Pareto front is empty, skipping plot.")
        return

    # Extract fitness values (Delta-V and TOF) from each individual
    delta_v_vals = [ind.fitness.values[0] for ind in pareto_front]
    tof_vals = [ind.fitness.values[1] for ind in pareto_front]

    # Create a new figure
    plt.figure(figsize=(9, 6))
    plt.scatter(delta_v_vals, tof_vals, c="#1f77b4", marker="o", label="Solutions")

    # Add labels and title for clarity
    plt.title("Pareto Front: Earth-Mars Transfer Optimization", fontsize=16)
    plt.xlabel("Total Mission ΔV (km/s)", fontsize=12)
    plt.ylabel("Time of Flight (days)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"\n[plot saved] Pareto front visualization saved to '{outpath}'")

def main(
    pop_size: int = 40,
    ngen: int = 8,
    cxpb: float = 0.6,
    mutpb: float = 0.2,
    seed: int = 42,
    verbose: bool = True,
    threads: int = 1
):
    toolbox = setup_deap(seed=seed)

    pool = None
    if threads and threads > 1:
        import multiprocessing as mp
        pool = mp.Pool(processes=threads)
        toolbox.register("map", pool.map)

    try:
        pop = toolbox.population(n=pop_size)
        hof = tools.ParetoFront()

        stats_dv = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_tof = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats = tools.MultiStatistics(delta_v=stats_dv, time_of_flight=stats_tof)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        pop, logbook = algorithms.eaMuPlusLambda(
            population=pop,
            toolbox=toolbox,
            mu=pop_size,      # Number of individuals to select for the next generation
            lambda_=pop_size, # Number of children to produce at each generation
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=ngen,
            stats=stats,
            halloffame=hof,
            verbose=verbose
        )

        print("\n=== Optimization Result: Pareto Front ===")
        print(f"Found {len(hof)} non-dominated solutions.")
        print("--------------------------------------------------------------------------------")
        print("  Delta-V (km/s)  |  Time of Flight (days)  |  Departure Date")
        print("--------------------------------------------------------------------------------")
        
        sorted_hof = sorted(list(hof), key=lambda ind: ind.fitness.values[0])

        for ind in sorted_hof:
            dv, tof = ind.fitness.values
            dep_dt, _ = decode_individual(ind)
            print(f"  {dv:^16.3f} |  {tof:^21.1f} |  {dep_dt.strftime('%Y-%m-%d')}")
        print("--------------------------------------------------------------------------------")
        plot_pareto_front(sorted_hof)
        return sorted_hof

    finally:
        if pool is not None:
            pool.close()
            pool.join()

if __name__ == "__main__":
    results = main(pop_size=200,ngen=20,threads=4)