"""
Algorithm 1: Weighted Sum Method

Multiobjective optimization is converted into a single-objective
optimization problem using a weighted sum of objectives.

All objectives are assumed to be minimization objectives.

Lecture idea:
    min F(x) = w1*f1(x) + w2*f2(x) + ... + wm*fm(x)

where:
    wi >= 0
    sum(wi) = 1
"""

import numpy as np


def weighted_sum(objectives, weights):
    """
    Calculate the weighted-sum scalar fitness.

    Parameters
    ----------
    objectives : array-like
        Objective values [f1, f2, ..., fm].

    weights : array-like
        Non-negative weights [w1, w2, ..., wm].
        They should sum to 1.

    Returns
    -------
    float
        Weighted-sum objective value.
    """

    objectives = np.asarray(objectives, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if objectives.ndim != 1:
        raise ValueError("objectives must be a 1-D array")

    if weights.ndim != 1:
        raise ValueError("weights must be a 1-D array")

    if len(objectives) != len(weights):
        raise ValueError(
            "objectives and weights must have the same length"
        )

    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("weights must sum to 1")

    return float(np.dot(objectives, weights))


def solve_weighted_sum(problem, weights, candidates):
    """
    Select the best candidate using the weighted-sum method.

    Parameters
    ----------
    problem : problem object
        Problem containing an evaluate(x) method.

    weights : array-like
        Objective weights.

    candidates : array-like
        Candidate decision vectors.

    Returns
    -------
    best_x : ndarray
        Decision vector of the best candidate.

    best_f : ndarray
        Objective values of the best candidate.

    best_value : float
        Weighted-sum scalar value.
    """

    best_x = None
    best_f = None
    best_value = np.inf

    for x in candidates:
        res = problem.evaluate(x)
        if isinstance(res, tuple):
            f, cv = res
            f = f[0] if np.ndim(f) > 1 else f
            cv = cv[0] if np.ndim(cv) > 0 else cv
        else:
            f, cv = res, 0.0

        f = np.asarray(f, dtype=float)
        value = weighted_sum(f, weights) + 1e6 * cv

        if value < best_value:
            best_value = value
            best_x = np.asarray(x).copy()
            best_f = f.copy()

    return best_x, best_f, best_value


def initialize(problem, n_weights=11, candidates=None, rng=None):
    """
    Initialize weights grid and candidate solutions if needed.
    """
    import operators as ops
    if rng is None:
        rng = np.random.default_rng(42)
    weights = ops.weight_grid(problem.n_obj, n_weights=n_weights)
    if candidates is None:
        candidates = problem.random_solutions(1000, rng)
    return weights, candidates


def evaluate(problem, x, weights):
    """
    Evaluate candidate x and compute weighted sum score.
    """
    f, cv = problem.evaluate(x)
    f = f[0] if f.ndim > 1 else f
    cv = cv[0] if np.ndim(cv) > 0 else cv
    score = weighted_sum(f, weights) + 1e6 * cv
    return score, f, cv


def step(problem, weights, candidates):
    """
    Perform one iteration (solve for one weight vector).
    """
    return solve_weighted_sum(problem, weights, candidates)


def run(problem, weights_grid=None, n_weights=11, candidates=None, n_candidates=1000, seed=42):
    """
    Run weighted sum over a grid of weights.
    Returns:
        X : (M, n_var) array of best solutions found
        F : (M, n_obj) array of objective vectors
    """
    import moo_utils as U
    rng = np.random.default_rng(seed)
    if weights_grid is None:
        import operators as ops
        weights_grid = ops.weight_grid(problem.n_obj, n_weights=n_weights)
    if candidates is None:
        candidates = problem.random_solutions(n_candidates, rng)

    X_list, F_list = [], []
    for w in weights_grid:
        bx, bf, _ = solve_weighted_sum(problem, w, candidates)
        if bx is not None:
            X_list.append(bx)
            F_list.append(bf)

    if not X_list:
        return np.empty((0, problem.n_var)), np.empty((0, problem.n_obj))

    X_arr = np.array(X_list)
    F_arr = np.array(F_list)
    # Filter to non-dominated set
    nd_idx = U.pareto_front(F_arr)
    return X_arr[nd_idx], F_arr[nd_idx]


if __name__ == "__main__":
    # Small standalone demonstration
    f = np.array([2.0, 5.0])
    w = np.array([0.4, 0.6])
    result = weighted_sum(f, w)
    print("Objectives:", f)
    print("Weights:", w)
    print("Weighted-sum value:", result)