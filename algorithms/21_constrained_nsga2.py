"""
Algorithm 21: Constrained NSGA-II
Proposed by: Deb et al. (2002)
Family: Constrained MOEA (Part 4, slides 6-12)

Core idea:
Extends NSGA-II to handle constraints rigorously using Deb's feasibility-first rules:
    Rule 1: Feasible beats infeasible, always (regardless of objective values).
    Rule 2: Both feasible -> ordinary Pareto dominance on objectives.
    Rule 3: Both infeasible -> lower total constraint violation CV wins.

Key modifications over standard NSGA-II:
1. Constraint violation CV(x) = sum max(0, g_i(x)) + sum |h_j(x)| is explicitly calculated.
2. In crowded tournament selection and elitist survivor selection, Deb's constrained
   dominance is used everywhere.
3. Infeasible individuals are ranked strictly below all feasible non-dominated fronts.
"""

import numpy as np
import moo_utils as U
import operators as ops


def initialize(problem, pop_size=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return problem.random_solutions(pop_size, rng)


def evaluate(problem, X):
    return problem.evaluate(X)


def compute_constrained_ranks_and_crowding(F, CV):
    """
    Compute ranks and crowding distance under Deb's feasibility rules.
    """
    N = len(F)
    fronts = U.non_dominated_sort(F, CV)
    ranks = np.zeros(N, dtype=int)
    crowding = np.zeros(N, dtype=float)

    for rank_idx, front in enumerate(fronts):
        ranks[front] = rank_idx
        if len(front) > 0:
            cd = U.crowding_distance(F[front])
            crowding[front] = cd

    return ranks, crowding, fronts


def select_mating_pool(X, ranks, crowding, rng):
    N = len(X)
    return ops.crowded_tournament(ranks, crowding, N, rng)


def make_offspring(problem, parents, pc=0.9, pm=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(parents)
    offspring = []

    for i in range(0, N, 2):
        p1 = parents[i]
        p2 = parents[(i + 1) % N]
        c1, c2 = ops.sbx_crossover(p1, p2, problem.lower, problem.upper, rng, pc=pc)
        c1 = ops.polynomial_mutation(c1, problem.lower, problem.upper, rng, pm=pm)
        c2 = ops.polynomial_mutation(c2, problem.lower, problem.upper, rng, pm=pm)
        offspring.append(c1)
        if len(offspring) < N:
            offspring.append(c2)

    X_off = np.array(offspring[:N])
    if problem.var_type == "int":
        X_off = np.round(X_off)
    return X_off


def constrained_survivor_selection(X_comb, F_comb, CV_comb, pop_size):
    fronts = U.non_dominated_sort(F_comb, CV_comb)
    survivors = []

    for front in fronts:
        if len(survivors) + len(front) <= pop_size:
            survivors.extend(front)
        else:
            needed = pop_size - len(survivors)
            cd = U.crowding_distance(F_comb[front])
            sorted_front_idx = np.argsort(-cd)
            chosen = [front[i] for i in sorted_front_idx[:needed]]
            survivors.extend(chosen)
            break

    survivors = np.array(survivors, dtype=int)
    return X_comb[survivors], F_comb[survivors], CV_comb[survivors]


def step(problem, X, F, CV, pc=0.9, pm=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    ranks, crowding, _ = compute_constrained_ranks_and_crowding(F, CV)
    mating_idx = select_mating_pool(X, ranks, crowding, rng)
    parents = X[mating_idx]

    X_off = make_offspring(problem, parents, pc=pc, pm=pm, rng=rng)
    F_off, CV_off = evaluate(problem, X_off)

    X_comb = np.vstack([X, X_off])
    F_comb = np.vstack([F, F_off])
    CV_comb = np.concatenate([CV, CV_off])

    return constrained_survivor_selection(X_comb, F_comb, CV_comb, N)


def run(problem, pop_size=100, n_gen=50, pc=0.9, pm=None, seed=42):
    rng = np.random.default_rng(seed)
    X = initialize(problem, pop_size=pop_size, rng=rng)
    F, CV = evaluate(problem, X)

    for gen in range(n_gen):
        X, F, CV = step(problem, X, F, CV, pc=pc, pm=pm, rng=rng)

    nd_idx = U.pareto_front(F, CV)
    return X[nd_idx], F[nd_idx]
