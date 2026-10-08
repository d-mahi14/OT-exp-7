"""
Algorithm 34: HypE (Hypervolume Estimation Algorithm)
Proposed by: Bader & Zitzler (2011)
Family: Indicator-Based MOEA (Part 5E, slides 17-19)

Core idea:
Monte-Carlo sampling to estimate hypervolume contributions in many-objective problems:
1. Define a bounding box between the per-objective ideal point z_ideal and a dominated reference point r.
2. Draw S random sample points uniformly inside the bounding box.
3. For every sample point:
   Count which individuals in the candidate pool dominate it:
   If dominated by d individuals (d >= 1), each dominator receives an equal 1/d share of that sample.
4. Sum each individual's accumulated share across all S samples -> estimated HypE fitness.
5. Environmental selection:
   Repeatedly remove the individual with the lowest estimated contribution,
   re-estimating until the population size reaches the target N.
"""

import numpy as np
import moo_utils as U
import operators as ops


def estimate_hype_fitness(F_pool, ref_point, ideal_point=None, n_samples=500, rng=None):
    """
    Monte-Carlo estimate of hypervolume contribution for each individual.
    Fully vectorized across all samples and candidates.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N, k = F_pool.shape
    if ideal_point is None:
        ideal_point = np.min(F_pool, axis=0)

    box_diff = np.maximum(ref_point - ideal_point, 1e-6)
    samples = rng.uniform(ideal_point, ideal_point + box_diff, size=(n_samples, k))

    # Vectorized dominance check: dom[i, s] is True if solution i dominates sample s
    dom = np.all(F_pool[:, None, :] <= samples[None, :, :], axis=2)  # (N, S)
    d_count = np.sum(dom, axis=0)  # (S,)
    weights = np.where(d_count > 0, 1.0 / np.maximum(d_count, 1), 0.0)
    shares = dom @ weights  # (N,)

    box_vol = np.prod(box_diff)
    fitness = shares * (box_vol / n_samples)
    return fitness


def hype_environmental_selection(X_comb, F_comb, CV_comb, target_size, ref_point, n_samples=500, rng=None):
    """
    Iteratively remove individuals with lowest HypE fitness until target_size is reached.
    Uses vectorized dominance matrix and efficient rank-1 updates.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N_comb = len(X_comb)
    if N_comb <= target_size:
        return X_comb, F_comb, CV_comb

    survivors = np.arange(N_comb)
    ideal_point = np.min(F_comb, axis=0)
    box_diff = np.maximum(ref_point - ideal_point, 1e-6)
    samples = rng.uniform(ideal_point, ideal_point + box_diff, size=(n_samples, F_comb.shape[1]))

    # Precompute dominance matrix: dom[i, s] == True if solution i dominates sample s
    dom = np.all(F_comb[:, None, :] <= samples[None, :, :], axis=2)  # (N_comb, S)
    d_count = np.sum(dom, axis=0).astype(int)

    # Constraint penalties
    cv_penalties = 1e6 * CV_comb.copy()

    while len(survivors) > target_size:
        weights = np.where(d_count > 0, 1.0 / np.maximum(d_count, 1), 0.0)
        fitness = dom @ weights - cv_penalties

        worst = int(np.argmin(fitness))
        # Update d_count for removed individual
        d_count -= dom[worst].astype(int)
        dom = np.delete(dom, worst, axis=0)
        cv_penalties = np.delete(cv_penalties, worst)
        survivors = np.delete(survivors, worst)

    return X_comb[survivors], F_comb[survivors], CV_comb[survivors]


def initialize(problem, pop_size=40, ref_point=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    X = problem.random_solutions(pop_size, rng)
    F, CV = problem.evaluate(X)
    if ref_point is None:
        ref_point = np.max(F, axis=0) + 1.0
    return X, F, CV, ref_point


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, F, CV, ref_point, n_samples=500, pc=0.9, pm=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    # Variation: create N offspring
    offspring = []
    for i in range(0, N, 2):
        p1 = X[i]
        p2 = X[(i + 1) % N]
        c1, c2 = ops.sbx_crossover(p1, p2, problem.lower, problem.upper, rng, pc=pc)
        c1 = ops.polynomial_mutation(c1, problem.lower, problem.upper, rng, pm=pm)
        c2 = ops.polynomial_mutation(c2, problem.lower, problem.upper, rng, pm=pm)
        offspring.append(c1)
        if len(offspring) < N:
            offspring.append(c2)

    X_off = np.array(offspring[:N])
    F_off, CV_off = evaluate(problem, X_off)

    X_comb = np.vstack([X, X_off])
    F_comb = np.vstack([F, F_off])
    CV_comb = np.concatenate([CV, CV_off])

    return hype_environmental_selection(X_comb, F_comb, CV_comb, target_size=N,
                                       ref_point=ref_point, n_samples=n_samples, rng=rng)


def run(problem, pop_size=40, n_gen=30, n_samples=500, ref_point=None, pc=0.9, pm=None, seed=42):
    rng = np.random.default_rng(seed)
    X, F, CV, ref_point = initialize(problem, pop_size=pop_size, ref_point=ref_point, rng=rng)

    for gen in range(n_gen):
        X, F, CV = step(problem, X, F, CV, ref_point, n_samples=n_samples, pc=pc, pm=pm, rng=rng)

    nd_idx = U.pareto_front(F, CV)
    return X[nd_idx], F[nd_idx]
