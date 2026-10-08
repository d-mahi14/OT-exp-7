"""
Algorithm 3: Benson's Method
Family: Classical (Part 1, slides 36-37)

Core mechanism:
Maximize weighted distance past a reference point (target), summed across objectives:

    max sum_{i=1}^k lambda_i * s_i(x)
    subject to s_i(x) >= 0,  i = 1, ..., k
               g_j(x) <= 0
               h_l(x) == 0

where in minimization convention:
    s_i(x) = z*_i - f_i(x)  (achievement: how far below target z*_i)

Equivalently (as a minimization problem):
    min -sum_{i=1}^k lambda_i * (z*_i - f_i(x)) + penalty * sum max(0, f_i(x) - z*_i)

Lecture steps:
1. Choose a reference point z* (targets in original/min objective units).
2. Define achievement s_i = z*_i - f_i(x) >= 0 (how far past target in good direction).
3. Choose direction weights lambda_i >= 0, sum lambda_i = 1.
4. Maximize weighted achievement subject to s_i >= 0 and feasibility.
5. Vary weights lambda to trace out non-convex Pareto regions.

Default parameters:
- lambda weights: evenly spread simplex grid
- reference point z*: nadir point or slightly worse than ideal
- penalty_factor: 1e6 for violating target or constraints
"""

import numpy as np
import moo_utils as U
import operators as ops


def benson_achievement(objectives, ref_point):
    """
    Compute achievements s_i = z*_i - f_i.
    Positive value means strictly better than target.
    """
    objectives = np.asarray(objectives, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    return ref_point - objectives


def benson_fitness(objectives, ref_point, weights, cv=0.0, penalty_factor=1e6):
    """
    Scalarized cost to minimize.
    cost = - sum(weights * s) + penalty * (target_violations + cv)
    """
    s = benson_achievement(objectives, ref_point)
    target_violations = np.sum(np.maximum(0.0, -s))
    weighted_achieve = float(np.dot(weights, s))
    return -weighted_achieve + penalty_factor * (target_violations + cv)


def initialize(problem, ref_point=None, n_weights=11, candidates=None, rng=None):
    """
    Initialize reference point, weight vectors, and candidate solutions.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if ref_point is None:
        # Default reference point: nadir point (or nadir + margin)
        ref_point = problem.nadir.copy()
        if np.allclose(ref_point, problem.ideal):
            ref_point = problem.ideal + 1.0
    weights_grid = ops.weight_grid(problem.n_obj, n_weights=n_weights)
    if candidates is None:
        candidates = problem.random_solutions(1000, rng)
    return ref_point, weights_grid, candidates


def evaluate(problem, x, ref_point, weights, penalty_factor=1e6):
    """
    Evaluate candidate x under Benson's scalarization.
    """
    f, cv = problem.evaluate(x)
    f = f[0] if f.ndim > 1 else f
    cv = cv[0] if np.ndim(cv) > 0 else cv
    score = benson_fitness(f, ref_point, weights, cv=cv, penalty_factor=penalty_factor)
    return score, f, cv


def solve_benson(problem, ref_point, weights, candidates, penalty_factor=1e6):
    """
    Find best candidate for a given weight setting.
    """
    best_x = None
    best_f = None
    best_score = np.inf

    for x in candidates:
        score, f, _ = evaluate(problem, x, ref_point, weights, penalty_factor)
        if score < best_score:
            best_score = score
            best_x = np.asarray(x).copy()
            best_f = f.copy()

    return best_x, best_f, best_score


def step(problem, ref_point, weights, candidates):
    """
    Perform one iteration (solve for one weight vector).
    """
    return solve_benson(problem, ref_point, weights, candidates)


def run(problem, ref_point=None, weights_grid=None, n_weights=11, candidates=None, n_candidates=1000, seed=42):
    """
    Run Benson's method over a grid of direction weights.
    """
    rng = np.random.default_rng(seed)
    if ref_point is None or weights_grid is None:
        ref_point, default_weights, _ = initialize(problem, ref_point=ref_point, n_weights=n_weights, rng=rng)
        if weights_grid is None:
            weights_grid = default_weights
    if candidates is None:
        candidates = problem.random_solutions(n_candidates, rng)

    X_list, F_list = [], []
    for w in weights_grid:
        bx, bf, score = solve_benson(problem, ref_point, w, candidates)
        if bx is not None and score < 1e5:
            X_list.append(bx)
            F_list.append(bf)

    if not X_list:
        bx, bf, _ = solve_benson(problem, ref_point, weights_grid[0], candidates)
        return np.array([bx]), np.array([bf])

    X_arr = np.array(X_list)
    F_arr = np.array(F_list)
    nd_idx = U.pareto_front(F_arr)
    return X_arr[nd_idx], F_arr[nd_idx]
