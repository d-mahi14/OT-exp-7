"""
Algorithm 4: Weight Metric Method (p-norm, Tchebycheff)
Family: Classical (Part 1, slide 38)
Task C Deep-Dive Algorithm for Roll No. 17.

Core idea:
Minimize the weighted distance from the objective vector to an ideal reference point z*,
using a p-norm:

    min ( sum_{i=1}^k w_i * |f_i(x) - z*_i|^p )^(1/p)     for 1 <= p < infinity
    min max_{i=1..k} [ w_i * |f_i(x) - z*_i| ]            for p = infinity (Tchebycheff)

where:
    z* = ideal point (best achievable value per objective alone)
    w_i >= 0, sum w_i = 1 (weights)
    p = 1 (Manhattan / linear metric)
    p = 2 (Euclidean metric)
    p = inf (Tchebycheff metric, reaches non-convex fronts!)

Lecture steps (Part 1, slide 38):
1. Choose the ideal point z* (best achievable value of each objective alone).
2. Choose the norm order: p = 1 (Manhattan), p = 2 (Euclidean), p -> inf (Tchebycheff).
3. Choose weights w_i and minimize the resulting distance.
4. Vary weights w_i to sweep out the Pareto front.

Complexity:
- O(k) per evaluation / candidate check.
- With N candidates or optimization solves: O(N * k).
"""

import numpy as np
import moo_utils as U
import operators as ops


def weight_metric(objectives, ideal_point, weights, p=2):
    """
    Compute the weighted p-norm distance to the ideal point.

    Parameters
    ----------
    objectives : array-like
        Objective values [f1, ..., fk] (minimization form).
    ideal_point : array-like
        Ideal reference point z* [z*1, ..., z*k].
    weights : array-like
        Weights [w1, ..., wk], non-negative.
    p : int, float, or str / np.inf
        Order of the norm: 1, 2, np.inf, or 'inf'/'chebyshev'/'tchebycheff'.

    Returns
    -------
    float
        Distance metric value.
    """
    f = np.asarray(objectives, dtype=float)
    z = np.asarray(ideal_point, dtype=float)
    w = np.asarray(weights, dtype=float)

    if f.ndim != 1 or z.ndim != 1 or w.ndim != 1:
        raise ValueError("objectives, ideal_point, and weights must be 1-D arrays")
    if len(f) != len(z) or len(f) != len(w):
        raise ValueError("objectives, ideal_point, and weights must have matching dimensions")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")

    diff = np.abs(f - z)

    if p == np.inf or p == "inf" or str(p).lower() in ("chebyshev", "tchebycheff"):
        return float(np.max(w * diff))
    elif p == 1:
        return float(np.sum(w * diff))
    else:
        p_val = float(p)
        if p_val <= 0:
            raise ValueError("p-norm order p must be positive (>= 1)")
        # Weighted p-norm: (sum w_i * |f_i - z_i|^p)^(1/p)
        return float((np.sum(w * (diff ** p_val))) ** (1.0 / p_val))


def initialize(problem, ideal_point=None, p=2, n_weights=11, candidates=None, rng=None):
    """
    Initialize ideal point, weights grid, and candidate pool.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if ideal_point is None:
        ideal_point = problem.ideal.copy()
    weights_grid = ops.weight_grid(problem.n_obj, n_weights=n_weights)
    if candidates is None:
        candidates = problem.random_solutions(1000, rng)
    return ideal_point, weights_grid, candidates


def evaluate(problem, x, ideal_point, weights, p=2, penalty_factor=1e6):
    """
    Evaluate candidate solution x under the weighted metric.
    """
    f, cv = problem.evaluate(x)
    f = f[0] if f.ndim > 1 else f
    cv = cv[0] if np.ndim(cv) > 0 else cv
    dist = weight_metric(f, ideal_point, weights, p=p)
    fitness = dist + penalty_factor * cv
    return fitness, f, cv


def solve_weight_metric(problem, ideal_point, weights, p=2, candidates=None, penalty_factor=1e6):
    """
    Solve for best candidate for a single weight setting and norm order p.
    """
    best_x = None
    best_f = None
    best_score = np.inf

    for x in candidates:
        score, f, _ = evaluate(problem, x, ideal_point, weights, p=p, penalty_factor=penalty_factor)
        if score < best_score:
            best_score = score
            best_x = np.asarray(x).copy()
            best_f = f.copy()

    return best_x, best_f, best_score


def step(problem, ideal_point, weights, p=2, candidates=None):
    """
    Perform one iteration (solve for one weight vector).
    """
    return solve_weight_metric(problem, ideal_point, weights, p=p, candidates=candidates)


def run(problem, ideal_point=None, p=np.inf, weights_grid=None, n_weights=11, candidates=None, n_candidates=1000, seed=42):
    """
    Run weight metric method across a weight grid.

    Returns:
        X : (M, n_var) array of best solutions
        F : (M, n_obj) array of objective vectors
    """
    rng = np.random.default_rng(seed)
    if ideal_point is None or weights_grid is None:
        ideal_point, default_weights, _ = initialize(problem, ideal_point=ideal_point, p=p, n_weights=n_weights, rng=rng)
        if weights_grid is None:
            weights_grid = default_weights
    if candidates is None:
        candidates = problem.random_solutions(n_candidates, rng)

    X_list, F_list = [], []
    for w in weights_grid:
        bx, bf, score = solve_weight_metric(problem, ideal_point, w, p=p, candidates=candidates)
        if bx is not None and score < 1e5:
            X_list.append(bx)
            F_list.append(bf)

    if not X_list:
        bx, bf, _ = solve_weight_metric(problem, ideal_point, weights_grid[0], p=p, candidates=candidates)
        return np.array([bx]), np.array([bf])

    X_arr = np.array(X_list)
    F_arr = np.array(F_list)
    nd_idx = U.pareto_front(F_arr)
    return X_arr[nd_idx], F_arr[nd_idx]
