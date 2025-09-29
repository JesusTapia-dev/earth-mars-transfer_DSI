"""
de_optimization_05.py

Single-objective optimization for Earth->Mars transfer using Differential Evolution (DE).
- Decision variables: departure date (days offset) and time of flight (days)
- Objective: minimize total Delta-V (heliocentric impulsive estimate)
- Method: Differential Evolution (DE/rand/1/bin) via DEAP
"""
import random
from datetime import datetime, timedelta
from typing import List

import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt 
from earth_mars_transfer_02 import run_transfer


# --------- Configurable search window  ---------
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
    """Objective: minimize total DV (km/s)."""
    dep_offset, tof_days = float(individual[0]), float(individual[1])
    window_days = float((END_DATE - START_DATE).days)

    if not (0.0 <= dep_offset <= window_days and TOF_MIN <= tof_days <= TOF_MAX):
        return (1e6,)

    dep_dt, arr_dt = decode_individual(individual)
    dep_tuple = (dep_dt.year, dep_dt.month, dep_dt.day, dep_dt.hour, dep_dt.minute, dep_dt.second)
    arr_tuple = (arr_dt.year, arr_dt.month, arr_dt.day, arr_dt.hour, arr_dt.minute, arr_dt.second)

    try:
        res = run_transfer(dep_tuple, arr_tuple)
        dv_total = float(res['dv_total'])
    except Exception as e:
        print(f"[objective] run_transfer failed for dep={dep_tuple}, arr={arr_tuple}: {e}")
        dv_total = 1e6

    return (dv_total,)


def check_bounds(min_b, max_b):
    """Decorator to ensure individuals remain within bounds."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for i, child in enumerate(offspring):
                for j, val in enumerate(child):
                    if val > max_b[j]:
                        child[j] = max_b[j]
                    elif val < min_b[j]:
                        child[j] = min_b[j]
            return offspring
        return wrapper
    return decorator


def setup_deap_de(seed: int = 42):
    """Setup DEAP toolbox for Differential Evolution."""
    random.seed(seed)
    np.random.seed(seed)

    if 'FitnessMin' not in creator.__dict__:
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    if 'Individual' not in creator.__dict__:
        creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    window_days = float((END_DATE - START_DATE).days)

    toolbox.register('attr_dep', random.uniform, 0.0, window_days)
    toolbox.register('attr_tof', random.uniform, TOF_MIN, TOF_MAX)
    
    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.attr_dep, toolbox.attr_tof), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    toolbox.register('evaluate', objective)

    return toolbox


def main(
    pop_size: int = 200,
    ngen: int = 20,
    cr: float = 0.8,
    f: float = 0.8,
    seed: int = 42,
    halloffame_k: int = 5,
    verbose: bool = True
):
    """Main DE algorithm loop."""
    toolbox = setup_deap_de(seed=seed)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(halloffame_k)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("std", np.std)

    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "avg", "std"

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    if verbose:
        print(logbook.stream)

    for g in range(1, ngen + 1):

        window_days = float((END_DATE - START_DATE).days)
        min_bound = [0.0, TOF_MIN]
        max_bound = [window_days, TOF_MAX]  

        for i in range(pop_size):
            target = pop[i]
            
            candidates = [ind for j, ind in enumerate(pop) if j != i]
            a, b, c = random.sample(candidates, 3)
            
            mutant = toolbox.clone(a)
            for k in range(len(mutant)):
                mutant[k] += f * (b[k] - c[k])
####review
            trial = toolbox.clone(target)
            rand_j = random.randrange(len(trial))
            for k in range(len(trial)):
                if random.random() < cr or k == rand_j:
                    trial[k] = mutant[k]
            for j, val in enumerate(trial):
                if val > max_bound[j]:
                    trial[j] = max_bound[j]
                elif val < min_bound[j]:
                    trial[j] = min_bound[j]            
####review
            
            trial.fitness.values = toolbox.evaluate(trial)
            if trial.fitness >= target.fitness:
                pop[i] = trial

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(pop), **record)
        if verbose:
            print(logbook.stream)

    print("\n=== Optimization Result (Differential Evolution) ===")
    best = hof[0]
    dep_dt, arr_dt = decode_individual(best)
    result = {
        "best_individual": list(best),
        "best_fitness": float(best.fitness.values[0]),
        "best_departure": dep_dt.isoformat(),
        "best_arrival": arr_dt.isoformat(),
        "dep_offset_days": float(best[0]),
        "tof_days": float(best[1]),
    }
    for k, v in result.items():
        print(f"{k}: {v}")
    
    plot_convergence(logbook, "de_convergence.png")

    return result

def plot_convergence(logbook, outpath: str):
    """
    Plots the convergence of the algorithm from the logbook.
    """
    gen = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")
    std_fitness = logbook.select("std")

    plt.figure(figsize=(8, 5))
    
    # Plot the minimum and average fitness
    plt.plot(gen, min_fitness, color="#2ca02c", lw=2, label="Best (min) Fitness")
    plt.plot(gen, avg_fitness, color="#1f77b4", lw=1.8, label="Average Fitness")
    
    # Add a shaded region for standard deviation
    avg_plus_std = [a + s for a, s in zip(avg_fitness, std_fitness)]
    avg_minus_std = [a - s for a, s in zip(avg_fitness, std_fitness)]
    plt.fill_between(gen, avg_minus_std, avg_plus_std, color="#1f77b4", alpha=0.15, label="avg ± std")

    plt.title("DE Algorithm Convergence")
    plt.xlabel("Generation")
    plt.ylabel("ΔV (km/s)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print(f"\n[saved] Convergence plot to {outpath}")


if __name__ == "__main__":
    result = main()