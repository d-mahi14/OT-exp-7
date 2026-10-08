"""
Algorithm 11: MOGA (Multi-Objective Genetic Algorithm)
Proposed by: Fonseca & Fleming (1993)
Family: Non-Elitist MOEA (Part 2, slides 14-20)

Core idea:
Rank each individual by dominance count: rank(i) = 1 + |{j : j dominates i}|.
Assign raw fitness inversely related to rank, then apply fitness sharing
across normalized objective space so clustered individuals have their fitness
penalized:

    1. Evaluate all objectives for every individual.
    2. rank(i) = 1 + (number of individuals that dominate i).
    3. Assign raw fitness: F_raw(i) = 1 / rank(i).
    4. Normalize objectives to [0, 1] using min-max scaling.
    5. Compute pairwise distances d_ij and sharing function:
       sh(d_ij) = 1 - (d_ij / sigma_share)^2 if d_ij < sigma_share else 0
       Niche count NC(i) = sum_j sh(d_ij)
    6. Shared fitness: F_shared(i) = F_raw(i) / NC(i)
    7. Selection via roulette wheel or tournament on F_shared.
    8. Crossover and mutation to produce the next generation.
"""

import numpy as np
import moo_utils as U
import operators as ops


def compute_moga_ranks(F, CV=None):
    """
    Fonseca-Fleming dominance rank:
    rank(i) = 1 + (number of individuals that dominate i).
    """
    N = len(F)
    D = U.dominance_matrix(F, CV)  # D[j, i] == True iff j dominates i
    # Count how many j dominate i (sum along columns of D)
    dominance_counts = np.sum(D, axis=0)
    ranks = 1 + dominance_counts
    return ranks


def compute_shared_fitness(F, ranks, sigma_share=0.5):
    """
    Compute shared fitness using niche counts in normalized objective space.
    """
    raw_fitness = 1.0 / ranks.astype(float)
    Fn = ops.minmax_normalize(F)
    nc = ops.niche_counts(Fn, sigma_share=sigma_share)
    shared_fitness = raw_fitness / nc
    return shared_fitness, nc


def initialize(problem, pop_size=100, rng=None):
    """
    Initialize random population.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    return problem.random_solutions(pop_size, rng)


def evaluate(problem, X):
    """
    Evaluate objective vectors and constraint violations.
    """
    return problem.evaluate(X)


def step(problem, X, F, CV, sigma_share=0.5, pc=0.9, pm=None, rng=None):
    """
    Perform one MOGA generation.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    # 1. Dominance ranks
    ranks = compute_moga_ranks(F, CV)

    # 2. Fitness sharing
    shared_fit, _ = compute_shared_fitness(F, ranks, sigma_share=sigma_share)

    # 3. Selection
    selected_idx = ops.roulette_wheel(shared_fit, N, rng)
    parents = X[selected_idx]

    # 4. Variation
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


def run(problem, pop_size=100, n_gen=50, sigma_share=0.5, pc=0.9, pm=None, seed=42):
    """
    Run MOGA optimization.
    """
    rng = np.random.default_rng(seed)
    X = initialize(problem, pop_size=pop_size, rng=rng)
    F, CV = evaluate(problem, X)

    archive_X, archive_F = X.copy(), F.copy()

    for gen in range(n_gen):
        X, F, CV = step(problem, X, F, CV, sigma_share=sigma_share, pc=pc, pm=pm, rng=rng)
        archive_X = np.vstack([archive_X, X])
        archive_F = np.vstack([archive_F, F])
        nd_idx = U.pareto_front(archive_F)
        archive_X = archive_X[nd_idx]
        archive_F = archive_F[nd_idx]

    nd_final = U.pareto_front(archive_F)
    return archive_X[nd_final], archive_F[nd_final]
