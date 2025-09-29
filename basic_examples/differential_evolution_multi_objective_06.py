"""
de_optimization_multi_parallel.py

Multi-objective optimization for Earth->Mars transfer using Differential Evolution (MODE).
- Decision variables: departure date (days offset) and time of flight (days)
- Objectives:
    1. Minimize total Delta-V (heliocentric impulsive estimate)
    2. Minimize Time of Flight
- Method: Multi-Objective Differential Evolution (DE/rand/1/bin) via DEAP
"""
import random
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import List

import numpy as np
from deap import base, creator, tools
import matplotlib.pyplot as plt
from earth_mars_transfer_02 import run_transfer

# --------- Configurable search window ---------
START_DATE = datetime(2026, 8, 1)
END_DATE   = datetime(2029, 2, 1)
TOF_MIN = 140.0
TOF_MAX = 320.0
# ---------------------------------------------


def decode_individual(ind: List[float]):
    """Decode individual -> (departure_datetime, arrival_datetime)."""
    dep_offset_days = float(ind[0])
    tof_days = float(ind[1])
    dep_date = START_DATE + timedelta(days=dep_offset_days)
    arr_date = dep_date + timedelta(days=tof_days)
    return dep_date, arr_date


def objective(individual: List[float]):
    """
    Multi-objective function.
    Returns a tuple: (total_dv, time_of_flight)
    """
    dep_offset, tof_days = float(individual[0]), float(individual[1])
    window_days = float((END_DATE - START_DATE).days)

    if not (0.0 <= dep_offset <= window_days and TOF_MIN <= tof_days <= TOF_MAX):
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


def setup_deap_mode(seed: int = 42):
    """Setup DEAP toolbox for Multi-Objective Differential Evolution."""
    random.seed(seed)
    np.random.seed(seed)

    # Use 'if not hasattr' to prevent errors if the script is run multiple times in an interactive session
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    window_days = float((END_DATE - START_DATE).days)

    toolbox.register("attr_dep", random.uniform, 0.0, window_days)
    toolbox.register("attr_tof", random.uniform, TOF_MIN, TOF_MAX)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_dep, toolbox.attr_tof), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective)

    return toolbox


def plot_pareto_front(pareto_front, outpath: str):
    """Plots the Pareto front: ΔV vs. Time of Flight."""
    tof_values = [ind.fitness.values[1] for ind in pareto_front]
    dv_values = [ind.fitness.values[0] for ind in pareto_front]

    plt.figure(figsize=(9, 6))
    plt.scatter(tof_values, dv_values, c="#007acc", alpha=0.8, s=50, edgecolors='k', linewidth=0.5)
    
    plt.title("Pareto Front for Earth-Mars Transfer (Parallel MODE)")
    plt.xlabel("Time of Flight (days)")
    plt.ylabel("Total ΔV (km/s)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print(f"\n[saved] Pareto front plot to {outpath}")


def main(
    pop_size: int = 150,
    ngen: int = 75,
    cxpb: float = 0.9,
    f: float = 0.7,
    seed: int = 42,
    threads: int = 4,
    verbose: bool = True,
    enable_prints: bool = False,
    enable_plots: bool = True,
):
    """
    Main Multi-Objective DE algorithm loop with a corrected parallel evaluation structure.
    
    Args:
        pop_size (int): Population size.
        ngen (int): Number of generations.
        cxpb (float): Crossover probability.
        f (float): Mutation factor (Differential weight).
        seed (int): Random seed.
        threads (int): Number of parallel processes to use.
        verbose (bool): Whether to print logbook updates.
        enable_plots (bool): Whether to generate the Pareto front plot.
    """
    toolbox = setup_deap_mode(seed=seed)

    pool = None
    if threads and threads > 1:
        print(f"Running in parallel with {threads} threads...")
        pool = mp.Pool(processes=threads)
        toolbox.register("map", pool.map)
    else:
        print("Running in serial mode...")
        toolbox.register("map", map) 

    try:
        pop = toolbox.population(n=pop_size)
        hof = tools.ParetoFront()
        
        stats_dv = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats_tof = tools.Statistics(key=lambda ind: ind.fitness.values[1])
        stats = tools.MultiStatistics(dv=stats_dv, tof=stats_tof)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        # Initial evaluation of the population using toolbox.map (runs in parallel)
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "dv", "tof"
        logbook.chapters["dv"].header = "min", "avg"
        logbook.chapters["tof"].header = "min", "avg"

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(pop), **record)
        if verbose:
            print(logbook.stream)
        
        window_days = float((END_DATE - START_DATE).days)
        min_bound = [0.0, TOF_MIN]
        max_bound = [window_days, TOF_MAX]

        # Main evolution loop
        for g in range(1, ngen + 1):
            # 1. Create a list to hold all new trial individuals for this generation
            offspring = []
            for i in range(pop_size):
                target = pop[i]
                
                # --- Mutation ---
                candidates = [ind for j, ind in enumerate(pop) if j != i]
                a, b, c = random.sample(candidates, 3)
                mutant = toolbox.clone(a)
                for k in range(len(mutant)):
                    mutant[k] += f * (b[k] - c[k])

                # --- Crossover ---
                trial = toolbox.clone(target)
                rand_j = random.randrange(len(trial))
                for k in range(len(trial)):
                    if random.random() < cxpb or k == rand_j:
                        trial[k] = mutant[k]
                
                # --- Enforce Bounds ---
                for j, val in enumerate(trial):
                    if val > max_bound[j]:
                        trial[j] = max_bound[j]
                    elif val < min_bound[j]:
                        trial[j] = min_bound[j]
                
                # Add the unevaluated trial individual to our list
                offspring.append(trial)

            # 2. Evaluate the entire batch of offspring in parallel
            fitnesses = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit
            
            # 3. Perform selection using the newly evaluated offspring
            for i in range(pop_size):
                if offspring[i].fitness.dominates(pop[i].fitness):
                    pop[i] = offspring[i]
                # Optional: Replace if non-dominated to promote diversity
                elif not pop[i].fitness.dominates(offspring[i].fitness):
                    pop[i] = offspring[i]

            # Update the hall of fame and logbook with the new population
            hof.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=g, evals=len(pop), **record)
            if verbose:
                print(logbook.stream)
        
        # --- Results Processing ---
        print("\n=== Optimization Result (Multi-Objective DE) ===")
        print(f"Found {len(hof)} non-dominated solutions.")
        sorted_hof = sorted(list(hof), key=lambda ind: ind.fitness.values[0])
        if enable_prints:
            print("--------------------------------------------------------------------------------")
            print("  Delta-V (km/s)  |  Time of Flight (days)  |  Departure Date")
            print("--------------------------------------------------------------------------------")
            for ind in sorted_hof:
                dv, tof = ind.fitness.values
                dep_dt, _ = decode_individual(ind)
                print(f"  {dv:^16.3f} |  {tof:^21.1f} |  {dep_dt.strftime('%Y-%m-%d')}")
            print("--------------------------------------------------------------------------------")


        if enable_plots:
            plot_pareto_front(hof, "runs/multi_objective_differential_evolution/mode_pareto_front_parallel.png")
        return hof

    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    import time
    start_time = time.time()    
    # Run with verbose=False to keep the final output clean
    pareto_solutions = main(verbose=False, enable_prints=True, threads=4) 
    end_time = time.time()
    
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds.")