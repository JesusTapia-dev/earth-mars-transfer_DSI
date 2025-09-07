"""
single_objective_optimization_03.py

Single-objective optimization prototype for Earth->Mars transfer:
- Decision variables: departure date (days offset) and time of flight (days)
- Objective: minimize total Delta-V (heliocentric impulsive estimate)
- Method: Genetic Algorithm via DEAP

Patched improvements:
- Use bounded SBX crossover and bounded polynomial mutation (no manual repair needed).
- Short-circuit infeasible individuals (consistent units; avoids wasted Lambert calls).
- Better exception logging.
- Optional parallel evaluation with --threads.
- Guard empty Hall of Fame.
"""

import argparse
import random
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List

import numpy as np
from deap import base, creator, tools, algorithms

from earth_mars_transfer_02 import run_transfer

# --------- Configurable search window (edit to taste) ---------
# We'll center a window around a plausible Earth->Mars opportunity.
START_DATE = datetime(2026, 8, 1)
END_DATE   = datetime(2029, 2, 1)

# Time of flight bounds (days)
TOF_MIN = 140.0
TOF_MAX = 320.0
# --------------------------------------------------------------


def clamp(x: float, lo: float, hi: float):
    return max(lo, min(hi, x))


def decode_individual(ind: List[float]):
    """Decode individual -> (departure_datetime, arrival_datetime)."""
    dep_offset_days = float(ind[0])
    tof_days = float(ind[1])
    dep_date = START_DATE + timedelta(days=dep_offset_days)
    arr_date = dep_date + timedelta(days=tof_days)
    return dep_date, arr_date


def objective(individual: List[float]):
    """
    Objective: minimize total DV (km/s).
    Short-circuit infeasible candidates to avoid mixing penalty units with ΔV
    and avoid unnecessary ephemeris/Lambert calls.
    """
    dep_offset, tof_days = float(individual[0]), float(individual[1])
    window_days = float((END_DATE - START_DATE).days)

    # Short-circuit infeasible candidates (fast + consistent units)
    if not (0.0 <= dep_offset <= window_days and TOF_MIN <= tof_days <= TOF_MAX):
        return (1e6,)

    dep_dt, arr_dt = decode_individual(individual)
    dep_tuple = (dep_dt.year, dep_dt.month, dep_dt.day, dep_dt.hour, dep_dt.minute, dep_dt.second)
    arr_tuple = (arr_dt.year, arr_dt.month, arr_dt.day, arr_dt.hour, arr_dt.minute, arr_dt.second)

    try:
        res = run_transfer(dep_tuple, arr_tuple)
        dv_total = float(res['dv_total'])
    except Exception as e:
        # If Lambert fails or ephemeris issue, return large penalty
        print(f"[objective] run_transfer failed for dep={dep_tuple}, arr={arr_tuple}: {e}")
        dv_total = 1e6

    return (dv_total,)


def setup_deap(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    # Fitness and Individual
    if 'FitnessMin' not in creator.__dict__:
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    if 'Individual' not in creator.__dict__:
        creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    window_days = float((END_DATE - START_DATE).days)

    # Decision variables:
    #   x0: departure offset in days (0 .. window_days)
    #   x1: time of flight days (TOF_MIN .. TOF_MAX)
    toolbox.register('attr_dep', random.uniform, 0.0, window_days)
    toolbox.register('attr_tof', random.uniform, TOF_MIN, TOF_MAX)
    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.attr_dep, toolbox.attr_tof), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # Base evaluator
    toolbox.register('evaluate', objective)

    # --- Bounded operators (recommended for continuous bounded domains) ---
    low = [0.0, TOF_MIN]
    up  = [window_days, TOF_MAX]

    # Simulated Binary Crossover (SBX), bounded
    # eta ~ 10-20 typical; higher = offspring closer to parents
    toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=low, up=up, eta=15.0)

    # Polynomial mutation, bounded
    toolbox.register('mutate', tools.mutPolynomialBounded, low=low, up=up, eta=20.0, indpb=0.5)

    # Selection
    toolbox.register('select', tools.selTournament, tournsize=3)

    return toolbox


def main(
    pop_size: int = 200,
    ngen: int = 20,
    cxpb: float = 0.7,
    mutpb: float = 0.3,
    seed: int = 42,
    halloffame_k: int = 5,
    verbose: bool = True,
    threads: int = 1,
    return_log: bool = False
):
    toolbox = setup_deap(seed=seed)

    # Optional parallelism
    pool = None
    if threads and threads > 1:
        import multiprocessing as mp
        pool = mp.Pool(processes=threads)
        toolbox.register("map", pool.map)

    try:
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(halloffame_k)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("std", np.std)

        # CAPTURE the logbook returned by eaSimple
        pop, logbook = algorithms.eaSimple(
            pop, toolbox,
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=ngen,
            stats=stats,
            halloffame=hof,
            verbose=verbose
        )

        if len(hof) == 0:
            print("No valid individuals found.")
            result = {
                "best_individual": None,
                "best_fitness": float('inf'),
                "best_departure": None,
                "best_arrival": None,
                "dep_offset_days": None,
                "tof_days": None,
            }
            if return_log:
                # Convert logbook to rows so it's JSON/CSV friendly
                result["log"] = [dict(r) for r in logbook]
            return result

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

        if return_log:
            result["log"] = [dict(r) for r in logbook]

        print("\n=== Optimization Result ===")
        for k, v in result.items():
            if k != "log":
                print(f"{k}: {v}")

        return result

    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == "__main__":
    # Direct call with parameters
    result = main(
        pop_size=200,
        ngen=30,
        cxpb=0.7,
        mutpb=0.5,
        seed=42,
        halloffame_k=5,
        verbose=True,
        threads=4,
        return_log=False
    )