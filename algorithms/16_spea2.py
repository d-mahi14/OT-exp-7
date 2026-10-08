"""
Algorithm 16: SPEA2 (Strength Pareto Evolutionary Algorithm 2)
Proposed by: Zitzler, Laumanns & Thiele (2001)
Family: Elitist, Archive-Based MOEA (Part 3, slides 20-27)

Core idea:
Maintains a fixed-size external archive of non-dominated solutions.
Fitness combines a dominance-based "strength" term with a k-th nearest-neighbor density term:

    1. Strength S(i) = |{ j in P U A : i dominates j }|
    2. Raw Fitness R(i) = sum_{j in P U A : j dominates i} S(j)
    3. Density D(i) = 1 / (sigma_i^k + 2),  where k = sqrt(N + N_archive)
    4. Final Fitness F(i) = R(i) + D(i)  (lower is better; F < 1 <=> non-dominated)
    5. Environmental Selection:
       - Copy all non-dominated (F < 1) into archive.
       - If under-full: fill with lowest-F dominated individuals.
       - If over-full: truncate by nearest-neighbor distance.
    6. Mating selection on archive; variation produces offspring population P.
"""

import numpy as np
import moo_utils as U
import operators as ops


def compute_spea2_fitness(F_all, CV_all=None, k_nn=None):
    """
    Compute SPEA2 Strength, Raw Fitness, Density, and Final Fitness.
    """
    M_total = len(F_all)
    if k_nn is None:
        k_nn = int(np.sqrt(M_total))
    k_nn = max(1, min(k_nn, M_total - 1))

    # 1. Pairwise dominance matrix: D[i, j] == True iff i dominates j
    D = U.dominance_matrix(F_all, CV_all)

    # Strength S(i) = how many j are dominated by i
    S = np.sum(D, axis=1)

    # Raw fitness R(i) = sum of S(j) for all j that dominate i
    # (j dominates i is D[j, i])
    R = np.zeros(M_total, dtype=float)
    for i in range(M_total):
        dominators = np.where(D[:, i])[0]
        R[i] = np.sum(S[dominators])

    # Density D(i) = 1 / (sigma_i^k + 2) in normalized objective space
    Fn = ops.minmax_normalize(F_all)
    dist_matrix = np.linalg.norm(Fn[:, None, :] - Fn[None, :, :], axis=2)
    # Sort distances to neighbors for each individual
    sorted_dists = np.sort(dist_matrix, axis=1)
    sigma_k = sorted_dists[:, k_nn]
    density = 1.0 / (sigma_k + 2.0)

    fitness = R + density
    return S, R, density, fitness, dist_matrix


def truncate_archive(archive_indices, dist_matrix, target_size):
    """
    Truncate over-full archive using nearest-neighbor distances.
    """
    active = list(archive_indices)
    while len(active) > target_size:
        # Submatrix of distances among active archive members
        sub_dist = dist_matrix[np.ix_(active, active)].copy()
        # Replace diagonal (0 distance to self) with large finite number
        np.fill_diagonal(sub_dist, 1e12)
        # Sort distances for each member
        sorted_sub = np.sort(sub_dist, axis=1)

        # Lexicographically find the individual with smallest distance to another member
        # (i.e. most crowded)
        worst_idx = 0
        for i in range(1, len(active)):
            # Compare rows sorted_sub[i] vs sorted_sub[worst_idx]
            diff = sorted_sub[i] - sorted_sub[worst_idx]
            non_zero = diff[np.abs(diff) > 1e-12]
            if len(non_zero) > 0 and non_zero[0] < 0:
                worst_idx = i

        del active[worst_idx]

    return np.array(active, dtype=int)


def environmental_selection(X_all, F_all, CV_all, fitness, dist_matrix, archive_size):
    """
    Select archive_size survivors into next archive.
    """
    M_total = len(fitness)
    non_dom = np.where(fitness < 1.0)[0]

    if len(non_dom) == archive_size:
        chosen = non_dom
    elif len(non_dom) < archive_size:
        # Fill with best dominated individuals (lowest fitness)
        dominated = np.where(fitness >= 1.0)[0]
        sorted_dom = dominated[np.argsort(fitness[dominated])]
        needed = archive_size - len(non_dom)
        chosen = np.concatenate([non_dom, sorted_dom[:needed]])
    else:
        # Truncate over-full non-dominated set
        chosen = truncate_archive(non_dom, dist_matrix, archive_size)

    return X_all[chosen], F_all[chosen], CV_all[chosen]


def initialize(problem, pop_size=100, archive_size=None, rng=None):
    """
    Initialize population and empty initial archive.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if archive_size is None:
        archive_size = pop_size
    X_pop = problem.random_solutions(pop_size, rng)
    X_arch = problem.random_solutions(archive_size, rng)
    return X_pop, X_arch


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X_pop, F_pop, CV_pop, X_arch, F_arch, CV_arch, pc=0.9, pm=None, rng=None):
    """
    Perform one SPEA2 generation.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    pop_size = len(X_pop)
    archive_size = len(X_arch)

    # 1. Merge population and archive
    X_all = np.vstack([X_pop, X_arch])
    F_all = np.vstack([F_pop, F_arch])
    CV_all = np.concatenate([CV_pop, CV_arch])

    # 2. Fitness evaluation
    _, _, _, fitness, dist_matrix = compute_spea2_fitness(F_all, CV_all)

    # 3. Environmental selection -> next archive
    X_arch_next, F_arch_next, CV_arch_next = environmental_selection(
        X_all, F_all, CV_all, fitness, dist_matrix, archive_size
    )

    # 4. Mating selection from next archive (binary tournament on fitness)
    # Recompute fitness within archive
    _, _, _, arch_fit, _ = compute_spea2_fitness(F_arch_next, CV_arch_next)
    mating_idx = np.empty(pop_size, dtype=int)
    for i in range(pop_size):
        a, b = rng.integers(archive_size, size=2)
        mating_idx[i] = a if arch_fit[a] <= arch_fit[b] else b
    parents = X_arch_next[mating_idx]

    # 5. Variation -> offspring population
    offspring = []
    for i in range(0, pop_size, 2):
        p1 = parents[i]
        p2 = parents[(i + 1) % pop_size]
        c1, c2 = ops.sbx_crossover(p1, p2, problem.lower, problem.upper, rng, pc=pc)
        c1 = ops.polynomial_mutation(c1, problem.lower, problem.upper, rng, pm=pm)
        c2 = ops.polynomial_mutation(c2, problem.lower, problem.upper, rng, pm=pm)
        offspring.append(c1)
        if len(offspring) < pop_size:
            offspring.append(c2)

    X_pop_next = np.array(offspring[:pop_size])
    if problem.var_type == "int":
        X_pop_next = np.round(X_pop_next)
    F_pop_next, CV_pop_next = evaluate(problem, X_pop_next)

    return X_pop_next, F_pop_next, CV_pop_next, X_arch_next, F_arch_next, CV_arch_next


def run(problem, pop_size=100, archive_size=None, n_gen=50, pc=0.9, pm=None, seed=42):
    """
    Run SPEA2 optimization.
    """
    rng = np.random.default_rng(seed)
    if archive_size is None:
        archive_size = pop_size
    X_pop, X_arch = initialize(problem, pop_size=pop_size, archive_size=archive_size, rng=rng)
    F_pop, CV_pop = evaluate(problem, X_pop)
    F_arch, CV_arch = evaluate(problem, X_arch)

    for gen in range(n_gen):
        X_pop, F_pop, CV_pop, X_arch, F_arch, CV_arch = step(
            problem, X_pop, F_pop, CV_pop, X_arch, F_arch, CV_arch, pc=pc, pm=pm, rng=rng
        )

    nd_idx = U.pareto_front(F_arch, CV_arch)
    return X_arch[nd_idx], F_arch[nd_idx]
