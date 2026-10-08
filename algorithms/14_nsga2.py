"""
Algorithm 14: NSGA-II (Non-Dominated Sorting Genetic Algorithm II)
Proposed by: Deb, Pratap, Agarwal & Meyarivan (2002)
Family: Elitist MOEA (Part 3, slides 4-11)

Core idea:
Fast non-dominated sorting + crowding distance + (μ + λ) parent-offspring elitism.
Solutions are layered into Pareto fronts in O(k N^2) time, and a parameterless
crowding distance preserves diversity along each front without needing a niche radius:

    1. Initialize population P_0 of size N randomly, evaluate all objectives.
    2. Fast non-dominated sort ranks solutions into fronts F_1, F_2, ...
    3. Crowding distance computed for each solution in a front:
       d_i = sum_{m=1}^k [ f_m(i+1) - f_m(i-1) ] / [ f_m^max - f_m^min ]
       boundary solutions get d = inf.
    4. Binary tournament selection using crowded comparison operator:
       prefer lower rank (better front); if ranks equal, prefer larger crowding distance.
    5. Crossover (SBX) and mutation (polynomial) create offspring population Q_t (size N).
    6. Combine R_t = P_t U Q_t (size 2N).
    7. Elitist selection: sort R_t into fronts; fill P_{t+1} front-by-front.
       In the last partially accepted front, select individuals with largest crowding distance.
    8. Repeat until termination.
"""

import numpy as np
import moo_utils as U
import operators as ops


def initialize(problem, pop_size=100, rng=None):
    """
    Initialize random population of size pop_size within problem bounds.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    return problem.random_solutions(pop_size, rng)


def evaluate(problem, X):
    """
    Evaluate objective vectors and constraint violations.
    """
    return problem.evaluate(X)


def compute_ranks_and_crowding(F, CV=None):
    """
    Compute front ranks and crowding distance for all individuals.
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
    """
    Binary tournament selection based on crowded comparison operator.
    """
    N = len(X)
    return ops.crowded_tournament(ranks, crowding, N, rng)


def make_offspring(problem, parents, pc=0.9, pm=None, rng=None):
    """
    Generate N offspring using SBX crossover and polynomial mutation.
    """
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


def elitist_survivor_selection(X_combined, F_combined, CV_combined, pop_size):
    """
    Select best pop_size individuals from 2N combined pool using fronts and crowding distance.
    """
    fronts = U.non_dominated_sort(F_combined, CV_combined)
    survivors = []

    for front in fronts:
        if len(survivors) + len(front) <= pop_size:
            survivors.extend(front)
        else:
            # Need partial selection from this front based on crowding distance
            needed = pop_size - len(survivors)
            cd = U.crowding_distance(F_combined[front])
            # Sort front descending by crowding distance
            sorted_front_idx = np.argsort(-cd)
            chosen = [front[i] for i in sorted_front_idx[:needed]]
            survivors.extend(chosen)
            break

    survivors = np.array(survivors, dtype=int)
    return X_combined[survivors], F_combined[survivors], CV_combined[survivors]


def step(problem, X, F, CV, pc=0.9, pm=None, rng=None):
    """
    Perform one full generation of NSGA-II.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    # 1. Rank and crowding distance of current population
    ranks, crowding, _ = compute_ranks_and_crowding(F, CV)

    # 2. Select mating pool
    mating_idx = select_mating_pool(X, ranks, crowding, rng)
    parents = X[mating_idx]

    # 3. Create offspring
    X_off = make_offspring(problem, parents, pc=pc, pm=pm, rng=rng)
    F_off, CV_off = evaluate(problem, X_off)

    # 4. Combine parents and offspring
    X_comb = np.vstack([X, X_off])
    F_comb = np.vstack([F, F_off])
    CV_comb = np.concatenate([CV, CV_off])

    # 5. Elitist survivor selection
    return elitist_survivor_selection(X_comb, F_comb, CV_comb, N)


def run(problem, pop_size=100, n_gen=50, pc=0.9, pm=None, seed=42):
    """
    Run NSGA-II optimization.
    Returns:
        X_front : Pareto-optimal decision vectors
        F_front : corresponding objective vectors
    """
    rng = np.random.default_rng(seed)
    X = initialize(problem, pop_size=pop_size, rng=rng)
    F, CV = evaluate(problem, X)

    for gen in range(n_gen):
        X, F, CV = step(problem, X, F, CV, pc=pc, pm=pm, rng=rng)

    nd_idx = U.pareto_front(F, CV)
    return X[nd_idx], F[nd_idx]
