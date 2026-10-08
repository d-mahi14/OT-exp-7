"""
Algorithm 10: VEGA (Vector Evaluated Genetic Algorithm)
Proposed by: Schaffer (1985)
Family: Non-Elitist MOEA (Part 2, slides 8-13)

Core idea:
Split the mating pool into k equal sub-populations (where k is the number of objectives).
Each sub-population is selected using ONE objective only, then all sub-populations are
shuffled together before crossover and mutation:

    1. Evaluate every objective f_1, ..., f_k for each individual.
    2. Split the mating pool into k equal sub-populations of size N / k.
    3. Select sub-population m using objective f_m only (proportional / tournament selection).
    4. Shuffle all sub-populations back into one mating pool.
    5. Apply crossover and mutation to create the new population.
    6. Non-elitist: parents are completely replaced by offspring.

Pros: Simple extension of single-objective GA; fast O(kN) selection.
Cons: Suffers from speciation / extreme solutions bias; lacks elitism; loses balanced trade-offs.
"""

import numpy as np
import moo_utils as U
import operators as ops


def initialize(problem, pop_size=100, rng=None):
    """
    Initialize random population within problem bounds.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    X = problem.random_solutions(pop_size, rng)
    return X


def evaluate(problem, X):
    """
    Evaluate objective vectors and constraint violations for population X.
    """
    return problem.evaluate(X)


def select_subpopulations(X, F, CV, k_sub, rng):
    """
    Select sub-populations where each sub-population is selected using one objective.
    In minimization form, smaller f is better.
    """
    N, k = F.shape
    sub_size = N // k
    mating_indices = []

    for m in range(k):
        # Fitness for objective m (lower is better, add penalty for CV)
        scores = F[:, m] + 1e6 * CV
        # Tournament selection of size 2 for objective m
        for _ in range(sub_size):
            i, j = rng.integers(N, size=2)
            winner = i if scores[i] <= scores[j] else j
            mating_indices.append(winner)

    # If N is not divisible by k, fill remaining slots
    while len(mating_indices) < N:
        m = rng.integers(k)
        scores = F[:, m] + 1e6 * CV
        i, j = rng.integers(N, size=2)
        winner = i if scores[i] <= scores[j] else j
        mating_indices.append(winner)

    rng.shuffle(mating_indices)
    return np.array(mating_indices, dtype=int)


def step(problem, X, F, CV, pc=0.9, pm=None, rng=None):
    """
    Perform one VEGA generation (non-elitist).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)
    k = F.shape[1]

    # Select parents via sub-populations
    parent_idx = select_subpopulations(X, F, CV, k, rng)
    parents = X[parent_idx]

    # Crossover & mutation to produce N offspring
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

    X_next = np.array(offspring[:N])
    if problem.var_type == "int":
        X_next = np.round(X_next)
    F_next, CV_next = evaluate(problem, X_next)
    return X_next, F_next, CV_next


def run(problem, pop_size=100, n_gen=50, pc=0.9, pm=None, seed=42):
    """
    Run VEGA algorithm for n_gen generations.
    Returns:
        X_front : non-dominated decision vectors from final generation (or archive)
        F_front : corresponding objective vectors
    """
    rng = np.random.default_rng(seed)
    X = initialize(problem, pop_size=pop_size, rng=rng)
    F, CV = evaluate(problem, X)

    # VEGA is historically non-elitist, but we track all non-dominated solutions found
    # to return the best approximation front at the end
    archive_X, archive_F = X.copy(), F.copy()

    for gen in range(n_gen):
        X, F, CV = step(problem, X, F, CV, pc=pc, pm=pm, rng=rng)
        # Update external tracking
        archive_X = np.vstack([archive_X, X])
        archive_F = np.vstack([archive_F, F])
        nd_idx = U.pareto_front(archive_F)
        archive_X = archive_X[nd_idx]
        archive_F = archive_F[nd_idx]

    # Final non-dominated set
    nd_final = U.pareto_front(archive_F)
    return archive_X[nd_final], archive_F[nd_final]
