r"""
Algorithm 33: SMS-EMOA (S-Metric Selection Evolutionary Multi-Objective Algorithm)
Proposed by: Beume, Naujoks & Emmerich (2007)
Family: Indicator-Based MOEA (Part 5E, slides 14-16)

Core idea:
Steady-state (μ + 1) evolutionary multiobjective algorithm directly optimizing hypervolume (S-metric):
1. Fix a dominated reference point r.
2. Initialize population of size μ.
3. Generate ONE offspring via crossover and mutation; add to pool (size μ + 1).
4. Non-dominated sort into fronts F_1, F_2, ...
5. Identify the worst front F_worst:
   - If |F_worst| == 1: delete that single individual directly.
   - If |F_worst| > 1: compute hypervolume contribution:
         Delta HV(i) = HV(F_worst) - HV(F_worst \ {i})
     for each member of the worst front, and delete the one with the smallest Delta HV.
6. Repeat until the evaluation budget or generation count is reached.
"""

import numpy as np
import moo_utils as U
import operators as ops


def compute_hypervolume_contribution_2d(front_F, ref_point):
    """
    Exact hypervolume contribution for each individual in a 2D non-dominated front.
    Both objectives are minimized.
    """
    N = len(front_F)
    if N == 1:
        return np.array([float(np.prod(ref_point - front_F[0]))])

    # Sort front ascending by f1 (so f2 is strictly descending)
    order = np.argsort(front_F[:, 0])
    sorted_F = front_F[order]

    contributions = np.zeros(N)
    for i in range(N):
        # Left bound for f1
        left_f1 = sorted_F[i - 1, 0] if i > 0 else sorted_F[i, 0]
        # Right bound for f2
        right_f2 = sorted_F[i + 1, 1] if i < N - 1 else ref_point[1]

        # For an extreme point or interior point:
        width = (sorted_F[i + 1, 0] - sorted_F[i, 0]) if i < N - 1 else (ref_point[0] - sorted_F[i, 0])
        height = (ref_point[1] - sorted_F[i, 1]) if i == 0 else (sorted_F[i - 1, 1] - sorted_F[i, 1])
        # Direct box contribution:
        box_w = (ref_point[0] - sorted_F[i, 0]) if i == N - 1 else (sorted_F[i + 1, 0] - sorted_F[i, 0])
        box_h = (ref_point[1] - sorted_F[i, 1]) if i == 0 else (sorted_F[i - 1, 1] - sorted_F[i, 1])
        contributions[order[i]] = max(0.0, box_w * box_h)

    return contributions


def compute_hv_contributions(F_sub, ref_point):
    """
    Compute hypervolume contribution Delta HV(i) for each point in F_sub.
    """
    N, k = F_sub.shape
    if k == 2:
        return compute_hypervolume_contribution_2d(F_sub, ref_point)

    # For 3+ objectives, use full HV differences
    total_hv = U.hypervolume(F_sub, ref_point)
    contributions = np.zeros(N)
    for i in range(N):
        subset = np.delete(F_sub, i, axis=0)
        sub_hv = U.hypervolume(subset, ref_point)
        contributions[i] = max(0.0, total_hv - sub_hv)
    return contributions


def steady_state_step(problem, X, F, CV, ref_point, pc=0.9, pm=None, rng=None):
    """
    Generate 1 offspring, add to population (size μ + 1), and remove worst.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    mu = len(X)

    # Pick 2 parents at random
    i, j = rng.integers(mu, size=2)
    c1, _ = ops.sbx_crossover(X[i], X[j], problem.lower, problem.upper, rng, pc=pc)
    y = ops.polynomial_mutation(c1, problem.lower, problem.upper, rng, pm=pm)
    if problem.var_type == "int":
        y = np.round(y)

    fy, cvy = problem.evaluate(y)
    fy = fy[0] if fy.ndim > 1 else fy
    cvy = cvy[0] if np.ndim(cvy) > 0 else cvy

    X_pool = np.vstack([X, y])
    F_pool = np.vstack([F, fy])
    CV_pool = np.append(CV, cvy)

    # Non-dominated sort
    fronts = U.non_dominated_sort(F_pool, CV_pool)
    worst_front = fronts[-1]

    if len(worst_front) == 1:
        discard_idx = worst_front[0]
    else:
        # Normalize worst front objectives before computing HV contributions
        F_worst = F_pool[worst_front]
        contributions = compute_hv_contributions(F_worst, ref_point)
        # Point with minimum contribution is dropped
        min_contrib_local = np.argmin(contributions)
        discard_idx = worst_front[min_contrib_local]

    X_next = np.delete(X_pool, discard_idx, axis=0)
    F_next = np.delete(F_pool, discard_idx, axis=0)
    CV_next = np.delete(CV_pool, discard_idx, axis=0)

    return X_next, F_next, CV_next


def initialize(problem, pop_size=30, ref_point=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    X = problem.random_solutions(pop_size, rng)
    F, CV = problem.evaluate(X)
    if ref_point is None:
        ref_point = np.max(F, axis=0) + 1.0
    return X, F, CV, ref_point


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, F, CV, ref_point, pc=0.9, pm=None, rng=None):
    return steady_state_step(problem, X, F, CV, ref_point, pc=pc, pm=pm, rng=rng)


def run(problem, pop_size=30, n_evals=100, ref_point=None, pc=0.9, pm=None, seed=42):
    rng = np.random.default_rng(seed)
    X, F, CV, ref_point = initialize(problem, pop_size=pop_size, ref_point=ref_point, rng=rng)

    for ev in range(n_evals):
        X, F, CV = step(problem, X, F, CV, ref_point, pc=pc, pm=pm, rng=rng)

    nd_idx = U.pareto_front(F, CV)
    return X[nd_idx], F[nd_idx]
